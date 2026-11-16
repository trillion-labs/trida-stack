import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from ocr.engine.cancellation import CancellationCheck, check_cancelled
from ocr.recognition.decoder_contract import BlockDiffusionConfig
from ocr.recognition.mlx_vlm_import import load_glm_ocr_symbols

# The glm_ocr processor is what patched AutoProcessor on import; the cheap loader never
# reaches it, so no save/restore dance is needed here any more.
KVCache, Model, ModelConfig = load_glm_ocr_symbols()


_MAX_COMPILED_VISION_GRAPHS = 4
_COMPILED_SINGLE_IMAGE_VISION: OrderedDict[tuple, object] = OrderedDict()


@dataclass
class MlxGenerationResult:
    token_ids: list[int]
    text_forwards: int
    rounds: int
    accepted_counts: list[int]
    repetition_stopped: bool = False

    @property
    def tokens_per_forward(self) -> float:
        return len(self.token_ids) / self.text_forwards if self.text_forwards else 0.0


@dataclass
class MlxDecodePrefix:
    input_ids: mx.array
    cache: list[KVCache]
    seed_logits: mx.array
    rope_delta: int
    max_new_tokens: int
    stop_on_repetition: bool = False
    # Set for tables: only loops without markup (inside one cell) stop.
    repetition_markup_ids: frozenset[int] | None = None


@dataclass
class _CooperativeDecodeState:
    prefix: MlxDecodePrefix
    cached_length: int
    pending: mx.array | None
    committed: list[int]
    accepted_counts: list[int]
    forwards: int = 0
    finished: bool = False
    repetition_stopped: bool = False


REPETITION_MAX_PERIOD = 64
REPETITION_MIN_SPAN = 64
REPETITION_MIN_REPEATS = 4


def repetition_cut(tokens: list[int], markup_ids: frozenset[int] | None = None) -> int | None:
    """Length to keep when the output ends in a degenerate loop, else None.

    A loop is a unit of at most 64 tokens repeated at least 4 times and
    over at least 64 tokens at the very end (e.g. "하기를 하기를 …",
    "0000…"). One copy of the unit is kept. Checked on token IDs after
    every committed block; each period test stops at its first mismatch.

    With `markup_ids` (tables), a loop only counts when its unit contains no
    markup token: text repeating inside one cell ("확인합니다. 확인합니다.")
    is degenerate, while rows of `<td>0</td>` legitimately repeat.
    """
    length = len(tokens)
    for period in range(1, REPETITION_MAX_PERIOD + 1):
        repeats = max(REPETITION_MIN_REPEATS, -(-REPETITION_MIN_SPAN // period))
        span = period * repeats
        if span > length:
            break
        base = length - span
        if all(tokens[index] == tokens[index + period] for index in range(base, length - period)):
            if markup_ids is not None and any(token in markup_ids for token in tokens[-period:]):
                continue
            start = base
            while start > 0 and tokens[start - 1] == tokens[start - 1 + period]:
                start -= 1
            return start + period
    return None


def load_mlx_glm_ocr(
    checkpoint: Path,
    *,
    quantization_bits: int | None = None,
    quantization_group_size: int = 64,
    quantization_scope: str = "all",
) -> Model:
    raw = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    raw["vision_config"].pop("dtype", None)
    model = Model(ModelConfig.from_dict(raw))
    weights = model.sanitize(mx.load(str(checkpoint / "model.safetensors")))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    if quantization_bits is not None:
        quantize_mlx_glm_ocr(
            model,
            bits=quantization_bits,
            group_size=quantization_group_size,
            scope=quantization_scope,
        )
    model.eval()
    return model


def quantize_mlx_glm_ocr(
    model: Model,
    *,
    bits: int,
    group_size: int = 64,
    scope: str = "all",
) -> None:
    if scope not in {"all", "text-layers"}:
        raise ValueError(f"Unknown MLX quantization scope {scope}")
    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=(
            None
            if scope == "all"
            else lambda path, module: (
                path.startswith("language_model.model.layers") and isinstance(module, nn.Linear)
            )
        ),
    )
    mx.eval(model.parameters())
    mx.clear_cache()


def to_mlx(values, dtype=None) -> mx.array:
    """numpy (or anything exposing __array__, e.g. a torch CPU tensor) to MLX."""
    if hasattr(values, "detach"):  # torch tensors in experiments
        values = values.detach().cpu().numpy()
    array = mx.array(np.asarray(values))
    return array.astype(dtype) if dtype is not None else array


def encode_image_mlx(
    model: Model,
    pixel_values: mx.array,
    image_grid_thw: mx.array,
) -> mx.array:
    if image_grid_thw.shape[0] == 1:
        return encode_single_image_compiled_mlx(
            model,
            pixel_values,
            image_grid_thw,
        )
    dtype = model.vision_tower.patch_embed.proj.weight.dtype
    features = model.vision_tower(
        pixel_values.astype(dtype),
        image_grid_thw,
        output_hidden_states=False,
    )
    mx.eval(features)
    return features


def _rotate_half_vision(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def _single_image_vision_attention(
    attention,
    hidden_states: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> mx.array:
    sequence_length = hidden_states.shape[0]
    qkv = attention.qkv(hidden_states)
    qkv = qkv.reshape(sequence_length, 3, attention.num_heads, -1)
    qkv = qkv.transpose(1, 0, 2, 3)
    query, key, value = mx.split(qkv, 3, axis=0)
    query = attention.q_norm(query.squeeze(0))
    key = attention.k_norm(key.squeeze(0))
    value = value.squeeze(0)

    query_dtype = query.dtype
    key_dtype = key.dtype
    cos = mx.expand_dims(cos, axis=-2).astype(mx.float32)
    sin = mx.expand_dims(sin, axis=-2).astype(mx.float32)
    query_float = query.astype(mx.float32)
    key_float = key.astype(mx.float32)
    query = (query_float * cos + _rotate_half_vision(query_float) * sin).astype(query_dtype)
    key = (key_float * cos + _rotate_half_vision(key_float) * sin).astype(key_dtype)

    query = query.transpose(1, 0, 2)[None, ...]
    key = key.transpose(1, 0, 2)[None, ...]
    value = value.transpose(1, 0, 2)[None, ...]
    attended = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=attention.scale,
        mask=None,
    )
    attended = attended.transpose(0, 2, 1, 3).reshape(
        sequence_length,
        -1,
    )
    return attention.proj(attended)


def _make_single_image_vision_graph(vision_tower):
    def forward(
        pixel_values: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        dtype = vision_tower.patch_embed.proj.weight.dtype
        hidden_states = vision_tower.patch_embed(pixel_values.astype(dtype))
        for block in vision_tower.blocks:
            hidden_states = hidden_states + _single_image_vision_attention(
                block.attn,
                block.norm1(hidden_states),
                cos,
                sin,
            )
            hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
        hidden_states = vision_tower.post_layernorm(hidden_states)
        hidden_states = hidden_states.reshape(
            -1,
            vision_tower.spatial_merge_size,
            vision_tower.spatial_merge_size,
            hidden_states.shape[-1],
        )
        hidden_states = vision_tower.downsample(hidden_states).reshape(
            -1,
            vision_tower.config.out_hidden_size,
        )
        return vision_tower.merger(hidden_states)

    return mx.compile(forward, inputs=vision_tower.state)


def encode_single_image_compiled_mlx(
    model: Model,
    pixel_values: mx.array,
    image_grid_thw: mx.array,
) -> mx.array:
    vision_tower = model.vision_tower
    position_embeddings, _ = vision_tower.rot_pos_emb(image_grid_thw)
    cos, sin = position_embeddings
    key = (
        id(vision_tower),
        tuple(pixel_values.shape),
        tuple(cos.shape),
        str(pixel_values.dtype),
    )
    compiled = _COMPILED_SINGLE_IMAGE_VISION.get(key)
    if compiled is None:
        compiled = _make_single_image_vision_graph(vision_tower)
        _COMPILED_SINGLE_IMAGE_VISION[key] = compiled
        while len(_COMPILED_SINGLE_IMAGE_VISION) > _MAX_COMPILED_VISION_GRAPHS:
            _COMPILED_SINGLE_IMAGE_VISION.popitem(last=False)
    else:
        _COMPILED_SINGLE_IMAGE_VISION.move_to_end(key)
    features = compiled(pixel_values, cos, sin)
    mx.eval(features)
    return features


def compiled_vision_cache_entries() -> int:
    return len(_COMPILED_SINGLE_IMAGE_VISION)


def make_cache(model: Model) -> list[KVCache]:
    return [KVCache() for _ in model.layers]


def truncate_cache(cache: list[KVCache], length: int) -> None:
    for layer in cache:
        if length < 0 or length > layer.offset:
            raise ValueError("Invalid MLX KV Cache truncate length")
        layer.trim(layer.offset - length)


def causal_mask(query_length: int, cached_length: int) -> mx.array:
    queries = mx.arange(query_length)[:, None]
    keys = mx.arange(cached_length + query_length)[None, :]
    return ((keys < cached_length) | ((keys - cached_length) <= queries))[None, None]


def draft_mask(
    query_length: int,
    cached_length: int,
    pending_length: int,
) -> mx.array:
    queries = mx.arange(query_length)[:, None]
    keys = mx.arange(cached_length + query_length)[None, :]
    return ((keys < cached_length) | (queries >= pending_length) | ((keys - cached_length) <= queries))[
        None, None
    ]


def response_positions(start: int, length: int) -> mx.array:
    values = mx.arange(start, start + length)[None, None, :]
    return mx.broadcast_to(values, (3, 1, length))


@mx.compile
def first_mismatch_index(draft: mx.array, target: mx.array) -> mx.array:
    positions = mx.arange(draft.shape[0])
    sentinel = mx.full(positions.shape, draft.shape[0], dtype=positions.dtype)
    return mx.min(mx.where(draft != target, positions, sentinel))


def first_mismatch_index_eager(draft: mx.array, target: mx.array) -> mx.array:
    positions = mx.arange(draft.shape[0])
    sentinel = mx.full(positions.shape, draft.shape[0], dtype=positions.dtype)
    return mx.min(mx.where(draft != target, positions, sentinel))


def text_logits(
    model: Model,
    input_ids: mx.array,
    *,
    cache: list[KVCache],
    position_ids: mx.array,
    mask: mx.array | None = None,
    inputs_embeds: mx.array | None = None,
    last_positions: int | None = None,
) -> mx.array:
    hidden = model.language_model.model(
        input_ids,
        inputs_embeds=inputs_embeds,
        cache=cache,
        position_ids=position_ids,
        mask=mask,
    )
    if last_positions is not None:
        hidden = hidden[:, -last_positions:]
    return model.language_model.lm_head(hidden)


def prefill_mlx(
    model: Model,
    input_ids: mx.array,
    image_features: mx.array,
    image_grid_thw: mx.array,
) -> tuple[list[KVCache], mx.array, int]:
    embeddings = model.language_model.model.embed_tokens(input_ids)
    embeddings = model.merge_input_ids_with_image_features(
        model.config.image_token_id,
        model.config.video_token_id,
        image_features,
        embeddings,
        input_ids,
    )
    positions, rope_delta = model.language_model.get_rope_index(
        input_ids,
        image_grid_thw,
        None,
        None,
    )
    cache = make_cache(model)
    logits = text_logits(
        model,
        input_ids,
        cache=cache,
        position_ids=positions,
        mask=causal_mask(input_ids.shape[1], 0),
        inputs_embeds=embeddings,
        last_positions=1,
    )
    mx.eval(logits)
    return cache, logits, int(rope_delta.item())


def generate_ar_mlx(
    model: Model,
    input_ids: mx.array,
    image_features: mx.array,
    image_grid_thw: mx.array,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> MlxGenerationResult:
    cache, logits, rope_delta = prefill_mlx(
        model,
        input_ids,
        image_features,
        image_grid_thw,
    )
    return decode_ar_mlx_from_prefix(
        model,
        input_ids,
        cache,
        logits,
        rope_delta,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
    )


def decode_ar_mlx_from_prefix(
    model: Model,
    input_ids: mx.array,
    cache: list[KVCache],
    logits: mx.array,
    rope_delta: int,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> MlxGenerationResult:
    generated: list[int] = []
    forwards = 0
    next_token = int(mx.argmax(logits[:, -1], axis=-1).item())
    while len(generated) < max_new_tokens and next_token not in eos_token_ids:
        generated.append(next_token)
        token = mx.array([[next_token]], dtype=input_ids.dtype)
        position = response_positions(cache[0].offset + rope_delta, 1)
        logits = text_logits(
            model,
            token,
            cache=cache,
            position_ids=position,
            last_positions=1,
        )
        forwards += 1
        next_token = int(mx.argmax(logits[:, -1], axis=-1).item())
    return MlxGenerationResult(
        token_ids=generated,
        text_forwards=forwards,
        rounds=len(generated),
        accepted_counts=[1] * len(generated),
    )


def generate_speculative_mlx(
    model: Model,
    input_ids: mx.array,
    image_features: mx.array,
    image_grid_thw: mx.array,
    *,
    max_new_tokens: int,
    config: BlockDiffusionConfig,
    eos_token_ids: set[int],
    compiled_accept: bool = False,
) -> MlxGenerationResult:
    cache, seed_logits, rope_delta = prefill_mlx(
        model,
        input_ids,
        image_features,
        image_grid_thw,
    )
    return decode_speculative_mlx_from_prefix(
        model,
        input_ids,
        cache,
        seed_logits,
        rope_delta,
        max_new_tokens=max_new_tokens,
        config=config,
        eos_token_ids=eos_token_ids,
        compiled_accept=compiled_accept,
    )


def decode_speculative_mlx_from_prefix(
    model: Model,
    input_ids: mx.array,
    cache: list[KVCache],
    seed_logits: mx.array,
    rope_delta: int,
    *,
    max_new_tokens: int,
    config: BlockDiffusionConfig,
    eos_token_ids: set[int],
    compiled_accept: bool = False,
    should_cancel: CancellationCheck = None,
    stop_on_repetition: bool = False,
    repetition_markup_ids: frozenset[int] | None = None,
) -> MlxGenerationResult:
    prefix_length = input_ids.shape[1]
    cached_length = prefix_length
    pending: mx.array | None = None
    committed: list[int] = []
    accepted_counts: list[int] = []
    forwards = 0
    repetition_stopped = False

    while len(committed) < max_new_tokens:
        check_cancelled(should_cancel)
        pending_length = 0 if pending is None else 1
        block = mx.full((1, config.block_size), config.mask_token_id, dtype=input_ids.dtype)
        query_ids = block if pending is None else mx.concatenate((pending, block), axis=1)
        query_length = query_ids.shape[1]
        response_offset = cached_length - prefix_length
        positions = response_positions(
            prefix_length + rope_delta + response_offset,
            query_length,
        )

        truncate_cache(cache, cached_length)
        raw_draft = text_logits(
            model,
            query_ids,
            cache=cache,
            position_ids=positions,
            mask=draft_mask(query_length, cached_length, pending_length),
        )
        forwards += 1
        shifted_draft = (
            mx.concatenate((seed_logits, raw_draft[:, :-1]), axis=1) if pending is None else raw_draft[:, :-1]
        )
        draft = mx.argmax(shifted_draft, axis=-1)[0]
        mx.eval(draft)
        check_cancelled(should_cancel)
        query_ids = draft[None] if pending is None else mx.concatenate((pending, draft[None]), axis=1)

        truncate_cache(cache, cached_length)
        raw_verify = text_logits(
            model,
            query_ids,
            cache=cache,
            position_ids=positions,
            mask=causal_mask(query_length, cached_length),
        )
        forwards += 1
        shifted_verify = (
            mx.concatenate((seed_logits, raw_verify[:, :-1]), axis=1)
            if pending is None
            else raw_verify[:, :-1]
        )
        ar_prediction = mx.argmax(shifted_verify, axis=-1)[0]
        mx.eval(ar_prediction)
        check_cancelled(should_cancel)
        accept_kernel = first_mismatch_index if compiled_accept else first_mismatch_index_eager
        accepted_prefix = int(accept_kernel(draft, ar_prediction).item())
        new_token_array = (
            mx.concatenate(
                (
                    draft[:accepted_prefix],
                    ar_prediction[accepted_prefix : accepted_prefix + 1],
                )
            )
            if accepted_prefix < config.block_size
            else draft
        )
        new_tokens = new_token_array.tolist()
        new_tokens = new_tokens[: max_new_tokens - len(committed)]
        eos_at = next(
            (i for i, token in enumerate(new_tokens) if token in eos_token_ids),
            None,
        )
        if eos_at is not None:
            new_tokens = new_tokens[: eos_at + 1]
        count = len(new_tokens)
        if not count:
            break

        truncate_cache(cache, cached_length + pending_length + count - 1)
        cached_length += pending_length + count - 1
        pending = mx.array([[new_tokens[-1]]], dtype=input_ids.dtype)
        committed.extend(new_tokens)
        accepted_counts.append(count)
        if eos_at is not None:
            break
        if stop_on_repetition and (cut := repetition_cut(committed, repetition_markup_ids)) is not None:
            del committed[cut:]
            repetition_stopped = True
            break

    if committed and committed[-1] in eos_token_ids:
        committed.pop()
    return MlxGenerationResult(
        token_ids=committed,
        text_forwards=forwards,
        rounds=len(accepted_counts),
        accepted_counts=accepted_counts,
        repetition_stopped=repetition_stopped,
    )


def decode_speculative_mlx_many_from_prefix(
    model: Model,
    prefixes: list[MlxDecodePrefix],
    *,
    config: BlockDiffusionConfig,
    eos_token_ids: set[int],
    compiled_accept: bool = False,
    should_cancel: CancellationCheck = None,
) -> list[MlxGenerationResult]:
    """Decode independent sequences while sharing host synchronization points.

    This is deliberately not a tensor batch. Each sequence keeps the same
    batch-1 kernels, masks, cache and acceptance rule as the scalar decoder.
    Only `mx.eval` barriers are coalesced across active regions.
    """
    if not prefixes:
        return []
    check_cancelled(should_cancel)
    if len(prefixes) == 1:
        prefix = prefixes[0]
        return [
            decode_speculative_mlx_from_prefix(
                model,
                prefix.input_ids,
                prefix.cache,
                prefix.seed_logits,
                prefix.rope_delta,
                max_new_tokens=prefix.max_new_tokens,
                config=config,
                eos_token_ids=eos_token_ids,
                compiled_accept=compiled_accept,
                should_cancel=should_cancel,
                stop_on_repetition=prefix.stop_on_repetition,
                repetition_markup_ids=prefix.repetition_markup_ids,
            )
        ]

    states = [
        _CooperativeDecodeState(
            prefix=prefix,
            cached_length=prefix.input_ids.shape[1],
            pending=None,
            committed=[],
            accepted_counts=[],
        )
        for prefix in prefixes
    ]
    accept_kernel = first_mismatch_index if compiled_accept else first_mismatch_index_eager

    while active := [state for state in states if not state.finished]:
        check_cancelled(should_cancel)
        rounds = []
        for state in active:
            prefix = state.prefix
            prefix_length = prefix.input_ids.shape[1]
            pending_length = 0 if state.pending is None else 1
            block = mx.full(
                (1, config.block_size),
                config.mask_token_id,
                dtype=prefix.input_ids.dtype,
            )
            query_ids = block if state.pending is None else mx.concatenate((state.pending, block), axis=1)
            query_length = query_ids.shape[1]
            positions = response_positions(
                prefix_length + prefix.rope_delta + state.cached_length - prefix_length,
                query_length,
            )
            truncate_cache(prefix.cache, state.cached_length)
            raw_draft = text_logits(
                model,
                query_ids,
                cache=prefix.cache,
                position_ids=positions,
                mask=draft_mask(
                    query_length,
                    state.cached_length,
                    pending_length,
                ),
            )
            state.forwards += 1
            shifted_draft = (
                mx.concatenate((prefix.seed_logits, raw_draft[:, :-1]), axis=1)
                if state.pending is None
                else raw_draft[:, :-1]
            )
            draft = mx.argmax(shifted_draft, axis=-1)[0]
            rounds.append(
                {
                    "state": state,
                    "pending_length": pending_length,
                    "query_length": query_length,
                    "positions": positions,
                    "draft": draft,
                }
            )
        mx.eval(*[round_state["draft"] for round_state in rounds])
        check_cancelled(should_cancel)

        for round_state in rounds:
            state = round_state["state"]
            draft = round_state["draft"]
            query_ids = (
                draft[None] if state.pending is None else mx.concatenate((state.pending, draft[None]), axis=1)
            )
            truncate_cache(state.prefix.cache, state.cached_length)
            raw_verify = text_logits(
                model,
                query_ids,
                cache=state.prefix.cache,
                position_ids=round_state["positions"],
                mask=causal_mask(
                    round_state["query_length"],
                    state.cached_length,
                ),
            )
            state.forwards += 1
            shifted_verify = (
                mx.concatenate(
                    (state.prefix.seed_logits, raw_verify[:, :-1]),
                    axis=1,
                )
                if state.pending is None
                else raw_verify[:, :-1]
            )
            round_state["ar_prediction"] = mx.argmax(
                shifted_verify,
                axis=-1,
            )[0]
        mx.eval(*[round_state["ar_prediction"] for round_state in rounds])
        check_cancelled(should_cancel)

        mismatch_indices = [
            accept_kernel(
                round_state["draft"],
                round_state["ar_prediction"],
            )
            for round_state in rounds
        ]
        mx.eval(*mismatch_indices)
        check_cancelled(should_cancel)
        for round_state, mismatch_index in zip(rounds, mismatch_indices, strict=True):
            state = round_state["state"]
            draft = round_state["draft"]
            ar_prediction = round_state["ar_prediction"]
            accepted_prefix = int(mismatch_index.item())
            new_token_array = (
                mx.concatenate(
                    (
                        draft[:accepted_prefix],
                        ar_prediction[accepted_prefix : accepted_prefix + 1],
                    )
                )
                if accepted_prefix < config.block_size
                else draft
            )
            remaining = state.prefix.max_new_tokens - len(state.committed)
            new_tokens = new_token_array[:remaining].tolist()
            eos_at = next(
                (index for index, token in enumerate(new_tokens) if token in eos_token_ids),
                None,
            )
            if eos_at is not None:
                new_tokens = new_tokens[: eos_at + 1]
            count = len(new_tokens)
            if not count:
                state.finished = True
                continue

            truncate_cache(
                state.prefix.cache,
                state.cached_length + round_state["pending_length"] + count - 1,
            )
            state.cached_length += round_state["pending_length"] + count - 1
            state.pending = mx.array(
                [[new_tokens[-1]]],
                dtype=state.prefix.input_ids.dtype,
            )
            state.committed.extend(new_tokens)
            state.accepted_counts.append(count)
            state.finished = eos_at is not None or len(state.committed) >= state.prefix.max_new_tokens
            if (
                not state.finished
                and state.prefix.stop_on_repetition
                and (cut := repetition_cut(state.committed, state.prefix.repetition_markup_ids)) is not None
            ):
                del state.committed[cut:]
                state.finished = True
                state.repetition_stopped = True

    results = []
    for state in states:
        if state.committed and state.committed[-1] in eos_token_ids:
            state.committed.pop()
        results.append(
            MlxGenerationResult(
                token_ids=state.committed,
                text_forwards=state.forwards,
                rounds=len(state.accepted_counts),
                accepted_counts=state.accepted_counts,
                repetition_stopped=state.repetition_stopped,
            )
        )
    return results
