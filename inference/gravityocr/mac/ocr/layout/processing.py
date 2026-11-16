"""PP-DocLayoutV3 pre- and post-processing in numpy (no torch, no transformers).

Mirrors `transformers.PPDocLayoutV3ImageProcessor` (fast, torchvision backend)
for what the product uses: the 800x800 resize and scaling in, and scores,
labels, boxes in reading order out. Its mask-to-polygon step (OpenCV) is not
used by Gravity and is not reproduced. `tests/test_layout_processing.py` pins
parity against transformers whenever it is installed.
"""

import numpy as np

from ocr.resample import resize_bicubic_uint8

LAYOUT_INPUT_SIZE = 800


def preprocess_layout(image) -> np.ndarray:
    """PIL image -> float32 [1, 3, 800, 800] in [0, 1].

    transformers resizes the uint8 tensor with torch's classic bicubic
    (antialias=False), then fuses rescale (1/255) and the identity mean/std
    into a single `(x - 0) / 255` float32 normalize.
    """
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    resized = resize_bicubic_uint8(
        np.asarray(rgb, dtype=np.uint8),
        LAYOUT_INPUT_SIZE,
        LAYOUT_INPUT_SIZE,
        antialias=False,
    )
    pixels = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
    pixels -= np.float32(0.0)
    pixels /= np.float32(255.0)
    return pixels[None]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    # exp of a large positive argument overflows; the tails saturate anyway.
    exp = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + exp), exp / (1 + exp)).astype(np.float32)


def reading_order(order_logits: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Rank of every query in reading order, from pairwise order logits [Q, Q].

    Saturated order scores often give two regions exactly the same vote.
    torch's argsort leaves such ties in an unspecified order, so ties go top
    to bottom, then left to right, using the boxes (x0, y0, x1, y1).
    """
    scores = _sigmoid(order_logits)
    votes = np.triu(scores, 1).sum(axis=0) + np.tril(1.0 - scores.T, -1).sum(axis=0)
    pointers = np.lexsort((boxes[:, 0], boxes[:, 1], votes))
    ranks = np.empty_like(pointers)
    ranks[pointers] = np.arange(len(pointers))
    return ranks


def postprocess_layout(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    order_logits: np.ndarray,
    *,
    threshold: float,
    image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Detections above `threshold` in reading order, for a single image.

    Inputs are the batch-1 model outputs: logits [1, Q, C], pred_boxes
    [1, Q, 4] (normalized cx, cy, w, h), order_logits [1, Q, Q].
    `image_size` is (width, height). Returns `scores`, `labels` and `boxes`
    (absolute x0, y0, x1, y1).
    """
    width, height = image_size
    logits, pred_boxes, order_logits = logits[0], pred_boxes[0], order_logits[0]
    queries, classes = logits.shape

    centers, dims = pred_boxes[:, :2], pred_boxes[:, 2:]
    boxes = np.concatenate([centers - 0.5 * dims, centers + 0.5 * dims], axis=-1)
    order = reading_order(order_logits, boxes)
    boxes = boxes * np.asarray([width, height, width, height], dtype=np.float32)

    flat = _sigmoid(logits).reshape(-1)
    # topk over all (query, class) pairs, keeping as many as there are queries.
    top = np.argsort(-flat, kind="stable")[:queries]
    scores = flat[top]
    labels = top % classes
    query = top // classes
    keep = scores >= threshold
    scores, labels, query = scores[keep], labels[keep], query[keep]
    by_order = np.argsort(order[query], kind="stable")
    return {
        "scores": scores[by_order],
        "labels": labels[by_order],
        "boxes": boxes[query][by_order],
    }
