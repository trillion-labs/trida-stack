import copy
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from PIL import Image
from tokenizers import Tokenizer

from ocr.engine.cancellation import CancellationCheck, check_cancelled
from ocr.engine.generation import load_generation_settings
from ocr.layout.assets import resolve_coreml_layout
from ocr.layout.detector import CoreMLDocLayoutDetector
from ocr.layout.regions import LayoutRegion, expand_box
from ocr.recognition.decoder_contract import BlockDiffusionConfig
from ocr.recognition.model import (
    MlxDecodePrefix,
    decode_speculative_mlx_many_from_prefix,
    encode_image_mlx,
    load_mlx_glm_ocr,
    prefill_mlx,
    to_mlx,
)
from ocr.recognition.preprocess import preprocess_image
from ocr.recognition.prompt import build_ocr_prompt_tensors

TASK_PROMPTS = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}


@dataclass
class PreparedRecognition:
    task: str
    visual_tokens: int
    input_ids: mx.array
    image_grid: mx.array
    patches: mx.array
    profile: dict
    # Area-based ceiling on generated tokens; None means the task limit.
    token_cap: int | None = None


# A crop cannot hold more text than fits in its pixels. Measured on 529 real
# text regions (2026-09-28), everything above 20 tokens per 1,000 px² at the
# app's 1600 px page scale was a hallucination: a one-line crop counting
# "1024, 2048, 4096, …" to 2,048 tokens, or "(96) … (97) …". Legible 8 px
# Korean peaks near 16. 30 leaves ~2x headroom; the floor keeps short labels.
REGION_TOKENS_PER_KILOPIXEL = 30
REGION_TOKEN_FLOOR = 128
PAGE_LONG_SIDE = 1600


def region_token_cap(width: int, height: int, page_long_side: int) -> int:
    scale = PAGE_LONG_SIDE / max(page_long_side, 1)
    area = max(width, 1) * max(height, 1) * scale * scale
    return max(REGION_TOKEN_FLOOR, int(area * REGION_TOKENS_PER_KILOPIXEL / 1_000))


# Every region in one cooperative decode keeps its KV cache (~64 KB per
# token) alive until the slowest region finishes. A 300-region newspaper page
# reached a 14 GB footprint that way on an 18 GB Mac, so structured pages run
# in bounded batches and the allocator cache is released between them.
REGION_BATCH_MAX_ITEMS = 16
REGION_BATCH_MAX_VISUAL_TOKENS = 8_192


def estimate_visual_tokens(width: int, height: int, max_long_side: int) -> int:
    """Upper-bound estimate of a region's visual tokens (28 px per merged token)."""
    scale = min(1.0, max_long_side / max(width, height, 1))
    return max(1, math.ceil(width * scale / 28) * math.ceil(height * scale / 28))


def plan_region_batches(
    token_estimates: list[int],
    *,
    max_items: int = REGION_BATCH_MAX_ITEMS,
    max_tokens: int = REGION_BATCH_MAX_VISUAL_TOKENS,
) -> list[range]:
    """Split regions, in reading order, into contiguous bounded batches.

    A single region larger than the token budget still gets its own batch.
    """
    batches: list[range] = []
    start = 0
    tokens = 0
    for index, estimate in enumerate(token_estimates):
        if index > start and (index - start >= max_items or tokens + estimate > max_tokens):
            batches.append(range(start, index))
            start, tokens = index, 0
        tokens += estimate
    if start < len(token_estimates):
        batches.append(range(start, len(token_estimates)))
    return batches


class LocalOcrEngine:
    # Extra context around each layout box, as a fraction of the page's long
    # side (about 5 px on the app's 1600 px pages); see `expand_box`. On
    # OmniDocBench demo18, 0.003 cut page CER from 0.445 to 0.417, and 0.012
    # pulled in neighbours and raised it to 0.488
    # (results/crop-margin-page-cer.json).
    crop_margin: float = 0.003

    def __init__(
        self,
        checkpoint: Path,
        *,
        structured: bool,
        max_long_side: int,
        coreml_layout_model: str | Path | None = None,
        crop_margin: float = crop_margin,
    ) -> None:
        self.crop_margin = crop_margin
        self.generation = load_generation_settings(checkpoint)
        self.config = BlockDiffusionConfig(
            block_size=self.generation.block_size,
            mask_token_id=self.generation.mask_token_id,
        )
        # The Rust tokenizer loads in ~12ms against ~700ms for pulling transformers in
        # just to reach the same tokenizer.json.
        tokenizer_file = checkpoint / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Checkpoint has no tokenizer.json: {checkpoint}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.markup_token_ids = frozenset(
            token_id for token, token_id in self.tokenizer.get_vocab().items() if "<" in token or ">" in token
        )
        self.model = load_mlx_glm_ocr(
            checkpoint,
            quantization_bits=6,
            quantization_group_size=64,
            quantization_scope="text-layers",
        )
        self.default_structured = structured
        self.coreml_layout_model = coreml_layout_model
        self.layout = self.build_layout() if structured else None
        self.max_long_side = max_long_side
        self.result_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._layout_lock = threading.Lock()

    def build_layout(self, compute_units: str = "ALL") -> CoreMLDocLayoutDetector:
        """The Core ML layout detector (ADR 0006), off the Neural Engine if it must.

        The release passes the bundled package; development resolves the
        pinned download. A package that fails to compile for the Neural
        Engine is retried on the CPU alone.
        """
        package = self.coreml_layout_model or resolve_coreml_layout()
        if package is None:
            raise RuntimeError("The Core ML layout model is not available")
        try:
            return CoreMLDocLayoutDetector(package, compute_units=compute_units)
        except Exception as error:
            if compute_units == "CPU_ONLY":
                raise
            print(
                f"Core ML layout failed to load, using CPU only: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            return CoreMLDocLayoutDetector(package, compute_units="CPU_ONLY")

    def release_memory(self) -> None:
        """Return MLX's cached buffers after a page.

        MLX keeps freed buffers for reuse; without this the process holds the
        heaviest page's peak until the next page.
        """
        mx.clear_cache()

    def detect_layout(self, image: Image.Image):
        """Layout detection that survives a Core ML runtime failure.

        The Apple Neural Engine can fail an inference intermittently (seen
        with two processes using Core ML at once). Retry once, then move this
        process to CPU-only Core ML for good rather than fail the page.
        """
        try:
            return self.layout.detect_profiled(image)
        except Exception as first:
            if self.layout.compute_units == "CPU_ONLY":
                raise
            try:
                return self.layout.detect_profiled(image)
            except Exception:
                print(
                    f"Core ML layout failed at runtime, switching to CPU only: {type(first).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
                with self._layout_lock:
                    self.layout = self.build_layout(compute_units="CPU_ONLY")
                return self.layout.detect_profiled(image)

    def ensure_layout(self) -> None:
        # The prewarm thread can race the request loop here, and building the detector
        # twice would double its memory.
        with self._layout_lock:
            if self.layout is None:
                self.layout = self.build_layout()

    def prepare_recognition(
        self,
        image: Image.Image,
        task: str = "text",
        *,
        should_cancel: CancellationCheck = None,
        token_cap: int | None = None,
    ) -> PreparedRecognition:
        check_cancelled(should_cancel)
        resize_started = time.perf_counter()
        if max(image.size) > self.max_long_side:
            scale = self.max_long_side / max(image.size)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        resize_ms = (time.perf_counter() - resize_started) * 1_000
        preprocess_started = time.perf_counter()
        processed = preprocess_image(image)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1_000
        image_tokens = int(processed.image_grid_thw.prod().item() // 4)
        input_ids, _ = build_ocr_prompt_tensors(
            self.tokenizer,
            image_tokens,
            TASK_PROMPTS[task],
        )
        transfer_started = time.perf_counter()
        input_ids_mx = to_mlx(input_ids)
        grid_mx = to_mlx(processed.image_grid_thw)
        patches_mx = to_mlx(processed.pixel_values, mx.bfloat16)
        mx.synchronize()
        check_cancelled(should_cancel)
        transfer_ms = (time.perf_counter() - transfer_started) * 1_000
        return PreparedRecognition(
            task=task,
            visual_tokens=image_tokens,
            input_ids=input_ids_mx,
            image_grid=grid_mx,
            patches=patches_mx,
            token_cap=token_cap,
            profile={
                "task": task,
                "token_cap": token_cap,
                "resized_size": list(image.size),
                "visual_tokens": image_tokens,
                "resize_ms": resize_ms,
                "preprocess_ms": preprocess_ms,
                "transfer_ms": transfer_ms,
            },
        )

    def prefill_prepared(
        self,
        prepared: PreparedRecognition,
        features: mx.array,
        *,
        should_cancel: CancellationCheck = None,
    ) -> tuple[MlxDecodePrefix, float]:
        check_cancelled(should_cancel)
        prefill_started = time.perf_counter()
        cache, seed_logits, rope_delta = prefill_mlx(
            self.model,
            prepared.input_ids,
            features,
            prepared.image_grid,
        )
        mx.synchronize()
        check_cancelled(should_cancel)
        prefill_ms = (time.perf_counter() - prefill_started) * 1_000
        return MlxDecodePrefix(
            input_ids=prepared.input_ids,
            cache=cache,
            seed_logits=seed_logits,
            rope_delta=rope_delta,
            max_new_tokens=min(
                self.generation.token_limit(prepared.task),
                prepared.token_cap or self.generation.token_limit(prepared.task),
            ),
            # Prose stops at any degenerate loop. Tables stop only on a loop
            # inside one cell, since rows of empty or 0/1 cells legitimately
            # repeat. Formulas never stop (0 & 0 & 0 is a real matrix).
            stop_on_repetition=prepared.task in {"text", "table"},
            repetition_markup_ids=self.markup_token_ids if prepared.task == "table" else None,
        ), prefill_ms

    def finish_decoded(
        self,
        prepared: PreparedRecognition,
        generated,
        *,
        vision_ms: float,
        prefill_ms: float,
        decode_ms: float,
    ) -> tuple[str, dict]:
        detokenize_started = time.perf_counter()
        text = self.tokenizer.decode(
            generated.token_ids,
            skip_special_tokens=True,
        ).strip()
        detokenize_ms = (time.perf_counter() - detokenize_started) * 1_000
        return text, {
            **prepared.profile,
            "output_tokens": len(generated.token_ids),
            "repetition_stopped": generated.repetition_stopped,
            "decode_rounds": generated.rounds,
            "text_forwards": generated.text_forwards,
            "vision_ms": vision_ms,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "detokenize_ms": detokenize_ms,
        }

    def recognize_many_profiled(
        self,
        items: list[tuple[Image.Image, str]],
        *,
        should_cancel: CancellationCheck = None,
    ) -> tuple[list[tuple[str, dict]], dict]:
        check_cancelled(should_cancel)
        if not items:
            return [], {
                "vision_mode": "isolated_multi_image_batch",
                "vision_batch_ms": 0.0,
                "batch_assembly_ms": 0.0,
                "batch_visual_tokens": 0,
                "decode_mode": "cooperative_eval_barriers",
                "decode_batch_ms": 0.0,
            }
        prepared = []
        for image, task, *cap in items:
            prepared.append(
                self.prepare_recognition(
                    image,
                    task,
                    should_cancel=should_cancel,
                    token_cap=cap[0] if cap else None,
                )
            )
        assembly_started = time.perf_counter()
        all_patches = mx.concatenate(
            [item.patches for item in prepared],
            axis=0,
        )
        all_grids = mx.concatenate(
            [item.image_grid for item in prepared],
            axis=0,
        )
        mx.eval(all_patches, all_grids)
        mx.synchronize()
        check_cancelled(should_cancel)
        assembly_ms = (time.perf_counter() - assembly_started) * 1_000
        vision_started = time.perf_counter()
        all_features = encode_image_mlx(
            self.model,
            all_patches,
            all_grids,
        )
        mx.synchronize()
        check_cancelled(should_cancel)
        vision_ms = (time.perf_counter() - vision_started) * 1_000
        total_tokens = sum(item.visual_tokens for item in prepared)
        features_by_item = []
        offset = 0
        for item in prepared:
            features = all_features[offset : offset + item.visual_tokens]
            offset += item.visual_tokens
            features_by_item.append(features)
        prefills = []
        for item, features in zip(prepared, features_by_item, strict=True):
            prefills.append(
                self.prefill_prepared(
                    item,
                    features,
                    should_cancel=should_cancel,
                )
            )
        decode_started = time.perf_counter()
        generated = decode_speculative_mlx_many_from_prefix(
            self.model,
            [prefix for prefix, _ in prefills],
            config=self.config,
            eos_token_ids=set(self.generation.eos_token_ids),
            compiled_accept=True,
            should_cancel=should_cancel,
        )
        mx.synchronize()
        check_cancelled(should_cancel)
        decode_ms = (time.perf_counter() - decode_started) * 1_000
        total_forwards = sum(result.text_forwards for result in generated)
        results = []
        for item, result, (_, prefill_ms) in zip(
            prepared,
            generated,
            prefills,
            strict=True,
        ):
            attributed_vision_ms = vision_ms * item.visual_tokens / total_tokens
            attributed_decode_ms = (
                decode_ms * result.text_forwards / total_forwards if total_forwards else 0.0
            )
            results.append(
                self.finish_decoded(
                    item,
                    result,
                    vision_ms=attributed_vision_ms,
                    prefill_ms=prefill_ms,
                    decode_ms=attributed_decode_ms,
                )
            )
        return results, {
            "vision_mode": "isolated_multi_image_batch",
            "vision_graph": ("compiled_single_image" if len(prepared) == 1 else "eager_isolated_multi_image"),
            "vision_batch_ms": vision_ms,
            "batch_assembly_ms": assembly_ms,
            "batch_visual_tokens": total_tokens,
            "decode_mode": ("cooperative_eval_barriers" if len(prepared) > 1 else "scalar"),
            "decode_batch_ms": decode_ms,
        }

    def recognize_regions_bounded(
        self,
        items: list[tuple[Image.Image, str]],
        *,
        should_cancel: CancellationCheck = None,
    ) -> tuple[list[tuple[str, dict]], dict]:
        """`recognize_many_profiled` with peak memory bounded per batch."""
        batches = plan_region_batches(
            [estimate_visual_tokens(item[0].width, item[0].height, self.max_long_side) for item in items],
            max_items=REGION_BATCH_MAX_ITEMS,
            max_tokens=REGION_BATCH_MAX_VISUAL_TOKENS,
        )
        if len(batches) <= 1:
            return self.recognize_many_profiled(items, should_cancel=should_cancel)
        recognized: list[tuple[str, dict]] = []
        merged: dict = {}
        for batch in batches:
            results, profile = self.recognize_many_profiled(
                [items[index] for index in batch],
                should_cancel=should_cancel,
            )
            recognized.extend(results)
            for key, value in profile.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    merged[key] = merged.get(key, 0) + value
                else:
                    merged.setdefault(key, value)
            del results
            # The finished batch's KV caches are garbage now; return them to
            # the system instead of keeping them in MLX's buffer cache.
            mx.clear_cache()
        merged["region_batches"] = len(batches)
        return recognized, merged

    def recognize_profiled(
        self,
        image: Image.Image,
        task: str = "text",
    ) -> tuple[str, dict]:
        results, _ = self.recognize_many_profiled([(image, task)])
        return results[0]

    def recognize_region(
        self,
        path: Path,
        box: tuple[int, int, int, int],
        *,
        label: str,
        task: str,
        should_cancel: CancellationCheck = None,
    ) -> dict:
        """Read one region of a page again, as the task the user picked.

        The app uses this for a region the layout skipped or read as the wrong
        kind. `box` is in page pixels, as the page result reported it.
        """
        if task not in {"text", "table", "formula"}:
            raise ValueError(f"unsupported region task: {task}")
        check_cancelled(should_cancel)
        image = Image.open(path).convert("RGB")
        x0, y0, x1, y1 = (round(value) for value in box)
        x0, x1 = max(0, min(image.width, x0)), max(0, min(image.width, x1))
        y0, y1 = max(0, min(image.height, y0)), max(0, min(image.height, y1))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("region box is empty inside the page")
        cap = region_token_cap(x1 - x0, y1 - y0, max(image.width, image.height))
        margin = round(max(image.width, image.height) * self.crop_margin)
        crop_box = expand_box((x0, y0, x1, y1), margin, (image.width, image.height))
        recognized, _ = self.recognize_many_profiled(
            [(image.crop(crop_box), task, cap)],
            should_cancel=should_cancel,
        )
        text, profile = recognized[0]
        region = LayoutRegion(label=label, task=task, score=1.0, box=(x0, y0, x1, y1), query_order=0)
        return {
            "text": text,
            "markdown": format_region(region, text),
            "task": task,
            "box": [x0, y0, x1, y1],
            "truncated": profile["output_tokens"] >= self.generation.token_limit(task),
        }

    def process(
        self,
        path: Path,
        structured: bool | None = None,
        *,
        should_cancel: CancellationCheck = None,
    ) -> dict:
        check_cancelled(should_cancel)
        started = time.perf_counter()
        use_structured = self.default_structured if structured is None else structured
        stat = path.stat()
        cache_key = (
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            use_structured,
            self.max_long_side,
            self.crop_margin,
        )
        cached = self.result_cache.get(cache_key)
        if cached is not None:
            check_cancelled(should_cancel)
            self.result_cache.move_to_end(cache_key)
            result = copy.deepcopy(cached)
            result["elapsed_ms"] = (time.perf_counter() - started) * 1_000
            result["cache_hit"] = True
            result["profile"]["cache_hit"] = True
            return result

        image = Image.open(path).convert("RGB")
        check_cancelled(should_cancel)
        image_width, image_height = image.size
        layout_ms = 0.0
        layout_profile = {
            "layout_preprocess_ms": 0.0,
            "layout_transfer_ms": 0.0,
            "layout_forward_ms": 0.0,
            "layout_postprocess_ms": 0.0,
            "layout_region_ms": 0.0,
        }
        if not use_structured:
            recognized, vision_batch = self.recognize_many_profiled(
                [(image, "text")],
                should_cancel=should_cancel,
            )
            text, region_profile = recognized[0]
            regions = [
                {
                    "id": "region-0000",
                    "reading_order": 0,
                    "label": "full_page",
                    "task": "text",
                    "score": 1.0,
                    "box": [0, 0, image_width, image_height],
                    "text": text,
                }
            ]
            region_profiles = [region_profile]
        else:
            self.ensure_layout()
            check_cancelled(should_cancel)
            layout_started = time.perf_counter()
            detected, layout_profile = self.detect_layout(image)
            check_cancelled(should_cancel)
            layout_ms = (time.perf_counter() - layout_started) * 1_000

            pieces = []
            regions = []
            readable = [region for region in detected if region.task not in {"image", "skip"}]
            page_long_side = max(image_width, image_height)
            margin = round(page_long_side * self.crop_margin)
            boxes = [region.box for region in readable]
            items = [
                (
                    image.crop(
                        expand_box(
                            region.box,
                            margin,
                            (image_width, image_height),
                            [other for other in boxes if other is not region.box],
                        )
                    ),
                    region.task,
                    region_token_cap(
                        region.box[2] - region.box[0],
                        region.box[3] - region.box[1],
                        page_long_side,
                    ),
                )
                for region in readable
            ]
            recognized, vision_batch = self.recognize_regions_bounded(
                items,
                should_cancel=should_cancel,
            )
            region_profiles = [profile for _, profile in recognized]
            outputs = {id(region): output for region, (output, _) in zip(readable, recognized, strict=True)}
            recognized_region_ids = []
            for reading_order, region in enumerate(detected):
                region_id = f"region-{reading_order:04d}"
                output = outputs.get(id(region), "")
                if region.task not in {"image", "skip"}:
                    recognized_region_ids.append(region_id)
                pieces.append(format_region(region, output, region_id=region_id))
                regions.append(
                    {
                        "id": region_id,
                        "reading_order": reading_order,
                        "label": region.label,
                        "task": region.task,
                        "score": region.score,
                        "box": list(region.box),
                        "text": output,
                    }
                )
            text = "\n\n".join(piece for piece in pieces if piece)
        aggregate_profile = {
            "layout_ms": layout_ms,
            **layout_profile,
            "regions": len(region_profiles),
            **vision_batch,
        }
        for key in (
            "resize_ms",
            "preprocess_ms",
            "transfer_ms",
            "vision_ms",
            "prefill_ms",
            "decode_ms",
            "detokenize_ms",
        ):
            aggregate_profile[key] = sum(profile[key] for profile in region_profiles)
        result = {
            "path": str(path.resolve()),
            "text": text,
            "regions": regions,
            "image_width": image_width,
            "image_height": image_height,
            "elapsed_ms": (time.perf_counter() - started) * 1_000,
            "structured": use_structured,
            "cache_hit": False,
            "profile": aggregate_profile,
            "region_profiles": region_profiles,
        }
        # Profiles exist only for recognized regions; figure regions have none.
        profiled_ids = recognized_region_ids if use_structured else [region["id"] for region in regions]
        truncated_region_ids = [
            profiled_ids[index]
            for index, profile in enumerate(region_profiles)
            if index < len(profiled_ids)
            and profile["output_tokens"] >= self.generation.token_limit(profile.get("task", "text"))
        ]
        result["truncated"] = bool(truncated_region_ids)
        # Hitting the area cap cuts a hallucination, not document content, so
        # it is reported separately and does not mark the page truncated.
        density_capped_ids = [
            profiled_ids[index]
            for index, profile in enumerate(region_profiles)
            if index < len(profiled_ids)
            and profile.get("token_cap") is not None
            and profile["token_cap"] < self.generation.token_limit(profile.get("task", "text"))
            and profile["output_tokens"] >= profile["token_cap"]
        ]
        repetition_stopped_ids = [
            profiled_ids[index]
            for index, profile in enumerate(region_profiles)
            if index < len(profiled_ids) and profile.get("repetition_stopped")
        ]
        result["generation"] = {
            "blockSize": self.generation.block_size,
            "eosTokenIDs": sorted(self.generation.eos_token_ids),
            "maxNewTokens": dict(self.generation.max_new_tokens),
            "truncatedRegionIDs": truncated_region_ids,
            "repetitionStoppedRegionIDs": repetition_stopped_ids,
            "densityCappedRegionIDs": density_capped_ids,
        }
        result["profile"]["cache_hit"] = False
        self.result_cache[cache_key] = copy.deepcopy(result)
        while len(self.result_cache) > 8:
            self.result_cache.popitem(last=False)
        return result


# Titles become Markdown headings; the rest of the hierarchy is not detected.
HEADING_LEVELS = {"doc_title": 1, "paragraph_title": 2}


def format_region(region: LayoutRegion, text: str, *, region_id: str = "") -> str:
    if region.task == "skip":
        return ""
    level = HEADING_LEVELS.get(region.label)
    if level is not None:
        title = " ".join(text.split()).lstrip("#").strip()
        return f"{'#' * level} {title}" if title else ""
    if region.task == "image":
        # The app replaces this with a link to the cropped file.
        return f"![{region.label}](gravity-figure:{region_id})"
    if region.task == "formula":
        return f"$$\n{text}\n$$"
    return text


def start_parent_watchdog(parent_pid: int | None) -> None:
    if parent_pid is None:
        return

    def watch() -> None:
        while True:
            try:
                os.kill(parent_pid, 0)
            except ProcessLookupError:
                os._exit(0)
            except PermissionError:
                pass
            time.sleep(1)

    threading.Thread(
        target=watch,
        name="local-ocr-parent-watchdog",
        daemon=True,
    ).start()
