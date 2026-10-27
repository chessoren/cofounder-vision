"""Synthetic macOS arrow cursor: rendering and masked template detection.

macOS screen recordings (QuickTime / ScreenCaptureKit) burn the pointer into the
frames. The default pointer is a black arrow with a white outline. We draw that
shape ourselves with ``cv2.fillPoly`` (supersampled, then ``INTER_AREA``
downscaled for anti-aliasing) and detect it with *masked* normalized
cross-correlation (``cv2.matchTemplate(..., TM_CCOEFF_NORMED, mask=...)``), so
the background around the arrow is ignored. Because the template contains both
the black body and the white rim, the zero-mean correlation is high on any
background colour.

The same renderer is used by the benchmark to synthesise recordings, which is
stated in RESULTS.md: detection is therefore tested on the *ideal* cursor shape
(plus video compression and noise), not on Apple's exact bitmap.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

# Arrow outline in "cursor units" (tip at the origin), close to the classic
# macOS arrow: left edge, notch, tail, right wing.
_ARROW = np.array(
    [[0.0, 0.0], [0.0, 15.0], [3.6, 11.7], [6.1, 17.4], [8.3, 16.5], [5.9, 10.9], [10.6, 10.9]],
    dtype=np.float64,
)
_ARROW_H = 17.4
_SS = 8  # supersampling factor

DEFAULT_SIZES = (16, 20, 24, 28, 32, 40, 48)


@lru_cache(maxsize=64)
def cursor_template(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Render the arrow cursor whose total height is about ``size`` pixels.

    Returns ``(bgr, gray, mask, tip)`` where ``mask`` is a uint8 0/255 matching
    mask (outline + body), ``bgr``/``gray`` contain the cursor on a mid-gray
    background (irrelevant under the mask) and ``tip`` is the ``(x, y)`` of the
    hotspot inside the template.
    """
    size = int(max(8, size))
    outline = max(1.0, size / 16.0)
    pad = int(np.ceil(outline)) + 1
    unit = (size - 2 * outline) / _ARROW_H
    w = int(np.ceil(10.6 * unit + 2 * pad)) + 1
    h = int(np.ceil(_ARROW_H * unit + 2 * pad)) + 1
    W, H = w * _SS, h * _SS
    pts = ((_ARROW * unit + pad) * _SS).round().astype(np.int32).reshape(-1, 1, 2)
    alpha = np.zeros((H, W), np.uint8)
    body = np.zeros((H, W), np.uint8)
    thick = max(1, int(round(2 * outline * _SS)))
    cv2.fillPoly(alpha, [pts], 255, lineType=cv2.LINE_8)
    cv2.polylines(alpha, [pts], True, 255, thickness=thick, lineType=cv2.LINE_8)
    cv2.fillPoly(body, [pts], 255, lineType=cv2.LINE_8)
    # colour image: white where alpha, black where body
    col = np.where(body > 0, 0, 255).astype(np.uint8)
    col = cv2.resize(col, (w, h), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
    gray = np.where(a > 0, col, 128).astype(np.uint8)
    mask = np.where(a >= 128, 255, 0).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return bgr, gray, mask, (pad, pad)


def _alpha(size: int) -> np.ndarray:
    """Soft alpha (0..1 float) of the cursor template, for compositing."""
    size = int(max(8, size))
    outline = max(1.0, size / 16.0)
    pad = int(np.ceil(outline)) + 1
    unit = (size - 2 * outline) / _ARROW_H
    w = int(np.ceil(10.6 * unit + 2 * pad)) + 1
    h = int(np.ceil(_ARROW_H * unit + 2 * pad)) + 1
    pts = ((_ARROW * unit + pad) * _SS).round().astype(np.int32).reshape(-1, 1, 2)
    a = np.zeros((h * _SS, w * _SS), np.uint8)
    cv2.fillPoly(a, [pts], 255)
    cv2.polylines(a, [pts], True, 255, thickness=max(1, int(round(2 * outline * _SS))))
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def draw_cursor(image: np.ndarray, x: float, y: float, size: int = 24) -> np.ndarray:
    """Alpha-composite the arrow cursor onto ``image`` (in place) with its tip at ``(x, y)``.

    Returns ``image`` for convenience. Parts falling outside the image are clipped.
    """
    bgr, _, _, (tx, ty) = cursor_template(size)
    a = _alpha(size)
    h, w = a.shape
    x0, y0 = int(round(x)) - tx, int(round(y)) - ty
    H, W = image.shape[:2]
    ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if ix1 <= ix0 or iy1 <= iy0:
        return image
    sub_a = a[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0, None]
    sub_c = bgr[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0].astype(np.float32)
    roi = image[iy0:iy1, ix0:ix1].astype(np.float32)
    image[iy0:iy1, ix0:ix1] = np.clip(roi * (1 - sub_a) + sub_c * sub_a, 0, 255).astype(np.uint8)
    return image


def detect_cursor(gray: np.ndarray, *, sizes=DEFAULT_SIZES, roi=None, min_score: float = 0.80) -> dict | None:
    """Find the arrow cursor in a grayscale frame with masked ``TM_CCOEFF_NORMED``.

    Args:
        gray: uint8 single-channel frame.
        sizes: candidate cursor heights in pixels (in ``gray``'s resolution).
        roi: optional ``[x, y, w, h]`` search window.
        min_score: minimum masked correlation to accept.

    Returns ``{x, y (tip), score, size, bbox:[x,y,w,h]}`` or ``None``.
    """
    H, W = gray.shape[:2]
    ox, oy = 0, 0
    img = gray
    if roi is not None:
        x, y, w, h = [int(v) for v in roi]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        img = gray[y0:y1, x0:x1]
        ox, oy = x0, y0
    best = None
    for s in sizes:
        _, tg, tm, (tx, ty) = cursor_template(int(s))
        th, tw = tg.shape
        if img.shape[0] < th or img.shape[1] < tw:
            continue
        r = cv2.matchTemplate(img, tg, cv2.TM_CCOEFF_NORMED, mask=tm)
        r[~np.isfinite(r)] = -1.0
        _, mx, _, loc = cv2.minMaxLoc(r)
        if best is None or mx > best["score"]:
            best = {
                "x": int(loc[0] + tx + ox),
                "y": int(loc[1] + ty + oy),
                "score": float(mx),
                "size": int(s),
                "bbox": [int(loc[0] + ox), int(loc[1] + oy), int(tw), int(th)],
            }
    if best is None or best["score"] < min_score:
        return None
    return best


class CursorTracker:
    """Frame-to-frame cursor tracker.

    Search order per frame (cheapest first):

    1. a window around the last position at the last known size;
    2. the bounding boxes of motion between the previous and the current frame
       (``cv2.absdiff`` + ``connectedComponentsWithStats``): a moving cursor
       always lies inside such a blob;
    3. a full-frame multi-size search (first frame, or after the cursor was
       lost and reappeared without detectable motion).
    """

    def __init__(self, sizes=DEFAULT_SIZES, min_score: float = 0.80, window: int = 160):
        self.sizes = tuple(int(s) for s in sizes)
        self.min_score = min_score
        self.window = window
        self.last: dict | None = None
        self.prev: np.ndarray | None = None

    def _near_sizes(self) -> tuple[int, ...]:
        if self.last is None:
            return self.sizes
        s = self.last["size"]
        return tuple(v for v in self.sizes if abs(v - s) <= max(4, s * 0.2)) or (s,)

    def update(self, gray: np.ndarray) -> dict | None:
        """Locate the cursor in ``gray``; returns the :func:`detect_cursor` dict or ``None``."""
        det = None
        sizes = self._near_sizes()
        pad = max(sizes) + 8
        if self.last is not None:
            win = self.window
            roi = [self.last["x"] - win, self.last["y"] - win, 2 * win, 2 * win]
            det = detect_cursor(gray, sizes=sizes, roi=roi, min_score=self.min_score)
        if det is None and self.prev is not None and self.prev.shape == gray.shape:
            d = cv2.absdiff(self.prev, gray)
            _, m = cv2.threshold(d, 20, 255, cv2.THRESH_BINARY)
            m = cv2.dilate(m, np.ones((5, 5), np.uint8))
            n, _, st, _ = cv2.connectedComponentsWithStats(m)
            comps = sorted(range(1, n), key=lambda i: -st[i, 4])[:12]
            for i in comps:
                x, y, w, h, _ = st[i]
                if w > gray.shape[1] * 0.6 and h > gray.shape[0] * 0.6:
                    continue  # whole-screen change: handled by the full search below
                c = detect_cursor(gray, sizes=sizes, roi=[x - pad, y - pad, w + 2 * pad, h + 2 * pad],
                                  min_score=self.min_score)
                if c is not None and (det is None or c["score"] > det["score"]):
                    det = c
        if det is None:
            det = detect_cursor(gray, sizes=self.sizes, min_score=self.min_score)
        self.prev = gray
        self.last = det
        return det
