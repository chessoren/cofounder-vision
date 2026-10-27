"""Overlay drawing in the Cofounder brand blue (#3D82DE, BGR (222, 130, 61))."""

from __future__ import annotations

import cv2
import numpy as np

from .imgio import to_bgr

BRAND_BLUE = (222, 130, 61)  # BGR of #3D82DE


def annotate(image: np.ndarray, boxes=None, points=None, labels=None, *, color=BRAND_BLUE) -> np.ndarray:
    """Return a copy of ``image`` with boxes, points and labels drawn on it.

    Args:
        image: BGR uint8 image (not modified).
        boxes: iterable of ``[x, y, w, h]``.
        points: iterable of ``[x, y]`` (drawn as a ring + dot, like a click).
        labels: strings; assigned to boxes first, then to points, in order.
        color: BGR colour (default brand blue).
    """
    out = to_bgr(image).copy()
    H, W = out.shape[:2]
    t = max(1, int(round(max(H, W) / 700)))
    fs = max(0.4, max(H, W) / 2400)
    labels = list(labels or [])
    li = 0

    def put_label(text: str, x: int, y: int) -> None:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t))
        x = int(min(max(0, x), max(0, W - tw - 6)))
        y = int(max(th + 6, y))
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 6, y + base - 2), color, -1)
        cv2.putText(out, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, t), cv2.LINE_AA)

    for b in boxes or []:
        x, y, w, h = [int(round(v)) for v in b]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, t + 1, cv2.LINE_AA)
        if li < len(labels):
            put_label(str(labels[li]), x, y - 2)
            li += 1
    for p in points or []:
        x, y = int(round(p[0])), int(round(p[1]))
        r = max(6, 6 * t)
        cv2.circle(out, (x, y), r, color, t + 1, cv2.LINE_AA)
        cv2.circle(out, (x, y), max(2, t + 1), color, -1, cv2.LINE_AA)
        if li < len(labels):
            put_label(str(labels[li]), x + r + 2, y - r)
            li += 1
    return out
