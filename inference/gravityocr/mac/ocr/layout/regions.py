from dataclasses import dataclass

from PIL import Image

# PP-DocLayoutV3 class ids: config.json `id2label` at the pinned revision
# (ocr.layout.assets.TORCH_LAYOUT_REVISION), which the Core ML export keeps.
PP_DOCLAYOUT_V3_LABELS = (
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer",
    "footnote",
    "formula_number",
    "header",
    "header",
    "image",
    "formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "text",
    "vision_footnote",
)

LABEL_TO_TASK = {
    "abstract": "text",
    "algorithm": "text",
    "aside_text": "text",
    "content": "text",
    "doc_title": "text",
    "figure_title": "text",
    "formula_number": "text",
    "paragraph_title": "text",
    "reference": "skip",
    "reference_content": "text",
    "seal": "text",
    "table": "table",
    "formula": "formula",
    "text": "text",
    "vision_footnote": "text",
    # Figures are not read; the app crops them from the source page.
    "image": "image",
    "chart": "image",
    "header": "skip",
    "footer": "skip",
    "footnote": "skip",
    "number": "skip",
}


@dataclass(frozen=True)
class LayoutRegion:
    label: str
    task: str
    score: float
    box: tuple[int, int, int, int]
    query_order: int


def intersection_over_union(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def containment(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    """Fraction of `inner`'s area that lies inside `outer`."""
    x0 = max(inner[0], outer[0])
    y0 = max(inner[1], outer[1])
    x1 = min(inner[2], outer[2])
    y1 = min(inner[3], outer[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
    return intersection / area if area else 0.0


def drop_formulas_inside_text(
    regions: list[LayoutRegion],
    threshold: float = 0.7,
) -> list[LayoutRegion]:
    """Remove formula regions that sit inside a text region.

    The HF config names both PP-DocLayoutV3 classes `display_formula` (5)
    and `inline_formula` (15) "formula". An inline formula is already read
    as part of its paragraph; recognizing it again appended every inline
    expression as a display block after the paragraph.
    """
    text_boxes = [region.box for region in regions if region.task == "text"]
    return [
        region
        for region in regions
        if region.task != "formula"
        or not any(containment(region.box, box) >= threshold for box in text_boxes)
    ]


def drop_nested_regions(
    regions: list[LayoutRegion],
    threshold: float = 0.85,
) -> list[LayoutRegion]:
    """Remove a readable region that lies inside a larger readable one.

    PP-DocLayoutV3 sometimes returns a paragraph and the lines inside it, or
    a table and a formula inside a cell. Their IoU is low, so NMS keeps
    both, and the page text then says the same sentence twice. The larger
    region already covers the text, so the nested one goes. Figures are not
    read, so text inside a figure is kept.
    """
    readable = [region for region in regions if region.task not in {"image", "skip"}]

    def area(box: tuple[int, int, int, int]) -> int:
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def nested(region: LayoutRegion) -> bool:
        return region.task not in {"image", "skip"} and any(
            other is not region
            and area(other.box) > area(region.box)
            and containment(region.box, other.box) >= threshold
            for other in readable
        )

    return [region for region in regions if not nested(region)]


def non_maximum_suppression(
    regions: list[LayoutRegion],
    same_label_threshold: float = 0.6,
    different_label_threshold: float = 0.98,
) -> list[LayoutRegion]:
    selected: list[LayoutRegion] = []
    for candidate in sorted(regions, key=lambda region: region.score, reverse=True):
        if any(
            intersection_over_union(candidate.box, kept.box)
            >= (same_label_threshold if candidate.label == kept.label else different_label_threshold)
            for kept in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda region: region.query_order)


def build_regions(
    result: dict,
    id2label: dict,
    image: Image.Image,
) -> list[LayoutRegion]:
    """Turn one post-processed detection result into clipped regions.

    The processor already returns queries sorted by the predicted reading order, so the
    enumeration index carries that order through NMS.

    Regions the page text leaves out (headers, footers, page numbers, footnotes)
    come back with task "skip" so the app can still show them on the source. They
    are suppressed among themselves only, so they never displace a readable region.
    """
    regions = []
    skipped = []
    for query_order, (score, label_id, box) in enumerate(
        zip(result["scores"], result["labels"], result["boxes"], strict=True)
    ):
        label = id2label[int(label_id.item())]
        task = LABEL_TO_TASK.get(label, "skip")
        x0, y0, x1, y1 = (round(value) for value in box.tolist())
        x0 = max(0, min(image.width, x0))
        x1 = max(0, min(image.width, x1))
        y0 = max(0, min(image.height, y0))
        y1 = max(0, min(image.height, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        (skipped if task == "skip" else regions).append(
            LayoutRegion(
                label=label,
                task=task,
                score=float(score.item()),
                box=(x0, y0, x1, y1),
                query_order=query_order,
            )
        )
    kept = drop_nested_regions(drop_formulas_inside_text(non_maximum_suppression(regions)))
    return sorted(kept + non_maximum_suppression(skipped), key=lambda region: region.query_order)


def expand_box(
    box: tuple[int, int, int, int],
    margin: int,
    page_size: tuple[int, int],
    neighbors: list[tuple[int, int, int, int]] = (),
) -> tuple[int, int, int, int]:
    """`box` grown by `margin` pixels on each side, for the recognizer's crop.

    PP-DocLayoutV3 boxes hug the ink, so a crop on the box can shave the tops
    of tall glyphs, descenders and the last character of a line. Each side
    grows by at most half the gap to a neighboring region facing it, so the
    crop never takes in the next region's text, and stays inside the page.
    """
    if margin <= 0:
        return box
    x0, y0, x1, y1 = box
    width, height = page_size
    left = top = right = bottom = margin
    for nx0, ny0, nx1, ny1 in neighbors:
        overlaps_rows = ny0 < y1 and ny1 > y0
        overlaps_columns = nx0 < x1 and nx1 > x0
        if overlaps_rows and nx1 <= x0:
            left = min(left, (x0 - nx1) // 2)
        if overlaps_rows and nx0 >= x1:
            right = min(right, (nx0 - x1) // 2)
        if overlaps_columns and ny1 <= y0:
            top = min(top, (y0 - ny1) // 2)
        if overlaps_columns and ny0 >= y1:
            bottom = min(bottom, (ny0 - y1) // 2)
    return (max(0, x0 - left), max(0, y0 - top), min(width, x1 + right), min(height, y1 + bottom))
