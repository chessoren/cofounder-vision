"""Did an action change the screen? (:func:`verify_change`) and SSIM.

Pipeline:

1. Grayscale + 3x3 Gaussian blur on both frames, ``cv2.absdiff``, fixed
   threshold (``DIFF_THRESHOLD``), OR-ed with a chroma (YCrCb Cr/Cb) difference
   above ``CHROMA_THRESHOLD`` for equal-luminance colour changes -> raw change
   mask. The blur + thresholds absorb JPEG / video re-encode noise.
2. ``cv2.dilate`` groups nearby changed pixels (glyphs of a word, parts of a
   button) and ``cv2.connectedComponentsWithStats`` yields components.
3. Each component is filtered:
   * fewer than ``MIN_PIXELS`` raw changed pixels -> noise;
   * thin and elongated (<= ``THIN_PX`` px across, >= 3x longer) -> blinking
     text caret (a single typed "l" or "i" is also dropped: known limitation);
   * cursor-sized and containing the arrow cursor in the before or after frame
     (masked template match, see :mod:`cofounder_vision.cursor`) -> pointer
     moved, not the UI.
4. Remaining components are merged into ``regions`` (``[x, y, w, h]``).
5. SSIM (Wang et al. 2004, 11x11 Gaussian, sigma 1.5, implemented with
   ``cv2.GaussianBlur``) on the ROI (full res) and on the whole frame
   (downscaled to <= 512 px for speed).
6. "Global" change: >= 10 % of pixels changed, or change spread over >= 25 %
   of a 16x10 grid with global SSIM < 0.9.
7. Verdict: ``changed`` / ``no_change`` / ``unexpected_global`` (the screen
   changed globally while the ROI around the click did not, e.g. navigation or a
   popup).
"""

from __future__ import annotations

import cv2
import numpy as np

from .cursor import DEFAULT_SIZES, detect_cursor
from .imgio import clip_rect, merge_boxes, to_bgr, to_gray

DIFF_THRESHOLD = 30
CHROMA_THRESHOLD = 24
GROUP_DILATE = 5
MIN_PIXELS = 20
THIN_PX = 5
CURSOR_MAX_SIDE = 64
LOCAL_MIN_FRACTION = 0.002
GLOBAL_CHANGE_FRACTION = 0.10
GLOBAL_COVERAGE = 0.25
COVERAGE_GRID = (16, 10)
GLOBAL_SSIM = 0.90
SSIM_MAX_SIDE = 512

_C1 = (0.01 * 255) ** 2
_C2 = (0.03 * 255) ** 2


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity of two same-size grayscale images.

    Standard formula with an 11x11 Gaussian window (sigma 1.5), K1=0.01,
    K2=0.03, L=255, computed with ``cv2.GaussianBlur`` (no scikit-image).
    """
    if a.shape != b.shape:
        raise ValueError("ssim: shape mismatch")
    if a.size == 0:
        return 1.0
    x = a.astype(np.float32)
    y = b.astype(np.float32)
    k, s = (11, 11), 1.5
    mx = cv2.GaussianBlur(x, k, s)
    my = cv2.GaussianBlur(y, k, s)
    sxx = cv2.GaussianBlur(x * x, k, s) - mx * mx
    syy = cv2.GaussianBlur(y * y, k, s) - my * my
    sxy = cv2.GaussianBlur(x * y, k, s) - mx * my
    num = (2 * mx * my + _C1) * (2 * sxy + _C2)
    den = (mx * mx + my * my + _C1) * (sxx + syy + _C2)
    return float(np.mean(num / den))


def _downscale(g: np.ndarray, max_side: int) -> np.ndarray:
    h, w = g.shape[:2]
    f = max_side / max(h, w)
    if f >= 1:
        return g
    return cv2.resize(g, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)


def _is_cursor(gb: np.ndarray, ga: np.ndarray, box) -> bool:
    """True if the arrow cursor is visible inside ``box`` (padded) in either frame."""
    x, y, w, h = box
    pad = 8
    roi = [x - pad, y - pad, w + 2 * pad, h + 2 * pad]
    sizes = tuple(s for s in DEFAULT_SIZES if s <= max(w, h) + 10) or DEFAULT_SIZES[:1]
    for g in (gb, ga):
        if detect_cursor(g, sizes=sizes, roi=roi, min_score=0.75) is not None:
            return True
    return False


def chroma(bgr: np.ndarray) -> np.ndarray:
    """Cr, Cb channels (``cv2.COLOR_BGR2YCrCb``) of a BGR image, H x W x 2 uint8."""
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 1:])


def change_components(gb: np.ndarray, ga: np.ndarray, *, ignore_cursor: bool = True,
                      ignore_mask: np.ndarray | None = None, cb: np.ndarray | None = None,
                      ca: np.ndarray | None = None) -> tuple[np.ndarray, list[list[int]], dict]:
    """Core change detector shared by :func:`verify_change` and the video segmenter.

    Args:
        gb, ga: grayscale before / after frames (same shape).
        cb, ca: optional chroma (Cr, Cb) of before / after, same H x W. Colour
            changes at equal luminance (e.g. a green toggle turning red) are
            invisible in gray, so a pixel also counts as changed when its
            chroma moves by more than ``CHROMA_THRESHOLD``.
        ignore_cursor: drop cursor-only components.
        ignore_mask: optional uint8 mask (non-zero = ignore these pixels).

    Returns ``(kept_mask, boxes, info)``: a uint8 0/255 mask of *raw* changed
    pixels belonging to kept components, their bounding boxes, and counters of
    dropped components (``noise``, ``thin``, ``cursor``).
    """
    b = cv2.GaussianBlur(gb, (3, 3), 0)
    a = cv2.GaussianBlur(ga, (3, 3), 0)
    diff = cv2.absdiff(b, a)
    _, raw = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    if cb is not None and ca is not None:
        cd = cv2.absdiff(cv2.GaussianBlur(cb, (5, 5), 0), cv2.GaussianBlur(ca, (5, 5), 0)).max(axis=2)
        raw = cv2.bitwise_or(raw, np.where(cd > CHROMA_THRESHOLD, 255, 0).astype(np.uint8))
    if ignore_mask is not None:
        raw[ignore_mask > 0] = 0
    info = {"noise": 0, "thin": 0, "cursor": 0}
    if not raw.any():
        return raw, [], info
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (GROUP_DILATE, GROUP_DILATE))
    grouped = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
    grouped = cv2.dilate(grouped, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    raw_counts = np.bincount(labels[raw > 0].ravel(), minlength=n)
    keep = np.zeros(n, bool)
    boxes = []
    pad = GROUP_DILATE // 2
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        rx, ry, rw, rh = x + pad, y + pad, max(1, w - 2 * pad), max(1, h - 2 * pad)
        if raw_counts[i] < MIN_PIXELS:
            info["noise"] += 1
            continue
        if min(rw, rh) <= THIN_PX and max(rw, rh) >= 3 * min(rw, rh):
            info["thin"] += 1
            continue
        if ignore_cursor and max(rw, rh) <= CURSOR_MAX_SIDE and _is_cursor(gb, ga, (rx, ry, rw, rh)):
            info["cursor"] += 1
            continue
        keep[i] = True
        boxes.append([int(rx), int(ry), int(rw), int(rh)])
    kept = np.where(keep[labels] & (raw > 0), 255, 0).astype(np.uint8)
    return kept, boxes, info


def verify_change(before: np.ndarray, after: np.ndarray, *, roi=None, expect: str = "any",
                  ignore_cursor: bool = True) -> dict:
    """Compare two screenshots taken before / after an action.

    Args:
        before, after: BGR uint8 screenshots. ``after`` is resized to ``before``
            if the sizes differ.
        roi: optional ``[x, y, w, h]`` around the click / target.
        expect: ``"any"`` (any real change counts), ``"local"`` (only a change
            inside ``roi`` counts) or ``"global"`` (a whole-screen change is the
            expected outcome, e.g. a navigation link).
        ignore_cursor: ignore components that are just the moving pointer.

    Returns:
        ``{changed, local_change, global_change, ssim_local, ssim_global,
        regions, verdict}`` plus ``coverage`` (fraction of a 16x10 grid with
        change) and ``ignored`` (counts of dropped noise / caret /
        cursor components).
    """
    if expect not in ("any", "local", "global"):
        raise ValueError("expect must be 'any', 'local' or 'global'")
    bb = to_bgr(before)
    ab = to_bgr(after)
    if ab.shape != bb.shape:
        ab = cv2.resize(ab, (bb.shape[1], bb.shape[0]), interpolation=cv2.INTER_AREA)
    gb, ga = to_gray(bb), to_gray(ab)
    H, W = gb.shape
    kept, boxes, info = change_components(gb, ga, ignore_cursor=ignore_cursor, cb=chroma(bb), ca=chroma(ab))
    regions = merge_boxes(boxes, gap=10)

    total = float(H * W)
    global_change = float(np.count_nonzero(kept)) / total
    # fraction of a coarse 16x10 grid whose cells contain kept change (>= 0.5 % of the cell)
    cells = cv2.resize((kept > 0).astype(np.float32), COVERAGE_GRID, interpolation=cv2.INTER_AREA)
    coverage = float(np.mean(cells >= 0.005))
    ssim_global = ssim(_downscale(gb, SSIM_MAX_SIDE), _downscale(ga, SSIM_MAX_SIDE))

    if roi is not None:
        rx, ry, rw, rh = clip_rect(*[int(v) for v in roi], W, H)
    else:
        rx, ry, rw, rh = 0, 0, W, H
    if rw > 0 and rh > 0:
        local_px = float(np.count_nonzero(kept[ry:ry + rh, rx:rx + rw]))
        local_change = local_px / float(rw * rh)
        ssim_local = ssim(gb[ry:ry + rh, rx:rx + rw], ga[ry:ry + rh, rx:rx + rw])
    else:
        local_px, local_change, ssim_local = 0.0, 0.0, 1.0

    global_big = global_change >= GLOBAL_CHANGE_FRACTION or (
        coverage >= GLOBAL_COVERAGE and ssim_global < GLOBAL_SSIM)
    roi_changed = local_px >= max(10.0, LOCAL_MIN_FRACTION * rw * rh)

    if roi is not None:
        if global_big and not roi_changed:
            verdict = "changed" if expect == "global" else "unexpected_global"
        elif roi_changed:
            verdict = "changed"
        elif regions and expect == "any":
            verdict = "changed"
        else:
            verdict = "no_change"
    else:
        if expect == "global":
            verdict = "changed" if global_big else "no_change"
        else:
            verdict = "changed" if regions else "no_change"

    return {
        "changed": verdict == "changed",
        "local_change": float(local_change),
        "global_change": float(global_change),
        "ssim_local": float(ssim_local),
        "ssim_global": float(ssim_global),
        "regions": regions,
        "verdict": verdict,
        "coverage": coverage,
        "ignored": info,
    }
