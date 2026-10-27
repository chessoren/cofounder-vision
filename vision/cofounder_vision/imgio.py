"""Image I/O and small shared helpers.

All public functions of :mod:`cofounder_vision` take and return numpy ``uint8``
BGR images (``H x W x 3``), the native OpenCV layout. Grayscale or BGRA input is
accepted and normalised by :func:`to_bgr`.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (PNG, JPEG, WebP, ...) into a BGR uint8 array.

    Raises ``ValueError`` if the bytes cannot be decoded.
    """
    if not data:
        raise ValueError("decode_image: empty buffer")
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("decode_image: OpenCV could not decode the buffer")
    return to_bgr(img)


def encode_png(image: np.ndarray) -> bytes:
    """Encode a BGR (or gray) uint8 array as PNG bytes."""
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(image))
    if not ok:
        raise ValueError("encode_png: OpenCV failed to encode the image")
    return buf.tobytes()


def png_b64(image: np.ndarray) -> str:
    """PNG-encode an image and return it as an ASCII base64 string."""
    return base64.b64encode(encode_png(image)).decode("ascii")


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Normalise gray / BGRA / 16-bit input to 3-channel uint8 BGR."""
    if img is None:
        raise ValueError("image is None")
    img = np.asarray(img)
    if img.dtype != np.uint8:
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 version of ``img`` (no copy if already gray)."""
    img = np.asarray(img)
    if img.ndim == 2:
        return img if img.dtype == np.uint8 else to_bgr(img)[:, :, 0]
    return cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2GRAY)


def clip_rect(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Clip an ``[x, y, w, h]`` rectangle to an image of size ``width x height``.

    Returns a possibly empty rectangle (w or h may be 0).
    """
    x0 = int(max(0, min(width, x)))
    y0 = int(max(0, min(height, y)))
    x1 = int(max(0, min(width, x + w)))
    y1 = int(max(0, min(height, y + h)))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def merge_boxes(boxes: list[list[int]], gap: int = 8) -> list[list[int]]:
    """Merge axis-aligned boxes that overlap or lie within ``gap`` pixels.

    Iterates until a fixed point, so chains of nearby boxes collapse into one.
    Returns boxes sorted by area (largest first) as ``[x, y, w, h]`` int lists.
    """
    rects = [[int(b[0]), int(b[1]), int(b[0] + b[2]), int(b[1] + b[3])] for b in boxes if b[2] > 0 and b[3] > 0]
    changed = True
    while changed and len(rects) > 1:
        changed = False
        out: list[list[int]] = []
        while rects:
            a = rects.pop()
            i = 0
            while i < len(rects):
                b = rects[i]
                if (a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]):
                    a = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    rects.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(a)
        rects = out
    res = [[r[0], r[1], r[2] - r[0], r[3] - r[1]] for r in rects]
    res.sort(key=lambda r: -r[2] * r[3])
    return res
