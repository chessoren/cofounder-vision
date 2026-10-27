"""Compact screen signatures for "am I on the expected screen?" checks.

* ``dhash``: 64-bit difference hash. The grayscale frame is shrunk with
  ``cv2.resize(INTER_AREA)`` to 9x8 and each bit records whether a pixel is
  brighter than its right neighbour. Robust to resolution (Retina vs 1x),
  compression and small local edits; sensitive to layout changes.
* ``hist``: HSV colour histogram (8 hue x 4 saturation x 4 value bins,
  ``cv2.calcHist``), L1-normalised so it sums to 1.

:func:`compare_screens` combines Hamming similarity of the hashes with
histogram intersection.
"""

from __future__ import annotations

import cv2
import numpy as np

from .imgio import to_bgr, to_gray

HIST_BINS = (8, 4, 4)
SAME_DHASH_MAX_BITS = 10
SAME_HIST_MIN = 0.70
W_DHASH = 0.6


def screen_signature(image: np.ndarray) -> dict:
    """Return ``{"dhash": 16-hex-char str, "hist": list[float] (128 values, sum 1)}``."""
    bgr = to_bgr(image)
    g = to_gray(bgr)
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS), [0, 180, 0, 256, 0, 256]).ravel()
    s = float(hist.sum()) or 1.0
    return {"dhash": f"{value:016x}", "hist": [round(float(v) / s, 6) for v in hist]}


def compare_screens(sig_a: dict, sig_b: dict) -> dict:
    """Compare two signatures.

    Returns ``{similarity (0-1), same (bool), dhash_distance (bits),
    hist_similarity (0-1)}``. ``same`` requires the hashes to differ by at most
    ``SAME_DHASH_MAX_BITS`` bits *and* histogram intersection >= ``SAME_HIST_MIN``.
    """
    da = int(sig_a["dhash"], 16)
    db = int(sig_b["dhash"], 16)
    dist = bin(da ^ db).count("1")
    ha = np.asarray(sig_a["hist"], np.float32)
    hb = np.asarray(sig_b["hist"], np.float32)
    if ha.shape != hb.shape:
        raise ValueError("compare_screens: histogram sizes differ")
    hist_sim = float(np.minimum(ha, hb).sum())
    hist_sim = max(0.0, min(1.0, hist_sim))
    dsim = 1.0 - dist / 64.0
    sim = W_DHASH * dsim + (1 - W_DHASH) * hist_sim
    return {"similarity": float(sim), "same": bool(dist <= SAME_DHASH_MAX_BITS and hist_sim >= SAME_HIST_MIN),
            "dhash_distance": int(dist), "hist_similarity": hist_sim}
