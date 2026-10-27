"""Locate a UI element (template) on a screenshot.

Pipeline (see :func:`locate`):

1. **Multi-scale NCC, coarse-to-fine.** For each scale ``s`` (default 13
   geometric steps over 0.5-2.0, ratio ~1.12, so Retina 2x vs 1x captures are
   covered both ways) the template is resized and matched with
   ``cv2.matchTemplate(TM_CCOEFF_NORMED)`` on the *coarsest* level of a
   ``cv2.pyrDown`` image pyramid at which the resized template still has a
   minimum side of ``COARSE_MIN_SIDE`` pixels. Two channels are matched and
   combined: grayscale intensity and a blurred Canny edge map
   (``score = 0.7 * gray + 0.3 * edge``). Edges make the score sensitive to
   structure (text, borders) rather than to smooth gradients.
2. **Candidate NMS + refinement.** The best few peaks per scale are pooled,
   non-maximum suppressed by centre distance, and each surviving candidate is
   re-scored at full resolution in a small window with fine scale steps
   (+-3 %, +-6 %).
3. **Ambiguity.** ``runner_up`` is the best refined score of a candidate whose
   centre lies outside the best match's neighbourhood. If
   ``best - runner_up < AMBIGUITY_MARGIN`` the result is flagged ``ambiguous``
   so the agent can ask a human instead of clicking the wrong twin.
4. **Feature fallback.** If the template score is below ``min_score``:
   ORB keypoints + Hamming kNN + Lowe ratio test + RANSAC homography
   (``cv2.findHomography``). Accepted only with >= ``MIN_INLIERS`` inliers, a
   geometrically sane homography (convex, near-similarity, plausible scale) and
   a photometric check of the rectified patch. AKAZE is not available in the
   OpenCV 5 main modules (it moved to contrib), so ORB is used.
5. Nothing passes: ``found=False`` and ``method="none"``. The function never
   reports a guess as found.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .imgio import to_bgr, to_gray

DEFAULT_SCALES = tuple(float(s) for s in np.round(np.geomspace(0.5, 2.0, 13), 4))
W_GRAY = 0.7
W_EDGE = 0.3
COARSE_MIN_SIDE = 8
MAX_PYR_LEVEL = 3
PEAKS_PER_SCALE = 3
MAX_CANDIDATES = 8
AMBIGUITY_MARGIN = 0.03
MIN_INLIERS = 8
FEATURE_VERIFY_NCC = 0.5
MIN_TEMPLATE_STD = 3.0
MIN_SCALED_SIDE = 10
MIN_SCALED_AREA = 200


def _edges(gray: np.ndarray) -> np.ndarray:
    """Blurred Canny edge map (uint8); blur gives +-1-2 px tolerance to edge positions."""
    # light pre-smoothing makes the edge map stable under blur / resampling of the screenshot
    e = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.0), 30, 90)
    return cv2.GaussianBlur(e, (5, 5), 1.2)


def _ncc_map(img: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    """``TM_CCOEFF_NORMED`` map with NaN/inf (flat windows) mapped to 0 and clipped."""
    r = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    r[~np.isfinite(r)] = 0.0
    return np.clip(r, -1.0, 1.0, out=r)


def _combined(gray_img, edge_img, tpl_gray, tpl_edge) -> np.ndarray:
    """Weighted gray + edge NCC. Falls back to gray only when the template has no edges."""
    g = _ncc_map(gray_img, tpl_gray)
    if tpl_edge is None or float(tpl_edge.std()) < 1.0:
        return g
    e = _ncc_map(edge_img, tpl_edge)
    return W_GRAY * g + W_EDGE * e


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    interp = cv2.INTER_AREA if (w < img.shape[1] or h < img.shape[0]) else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, w), max(1, h)), interpolation=interp)


def _top_peaks(score: np.ndarray, k: int, sup_w: int, sup_h: int):
    """Return up to ``k`` (score, x, y) maxima, suppressing a ``sup_w x sup_h`` box around each."""
    s = score.copy()
    out = []
    for _ in range(k):
        _, mx, _, (x, y) = cv2.minMaxLoc(s)
        if mx <= -0.5:
            break
        out.append((float(mx), int(x), int(y)))
        x0, y0 = max(0, x - sup_w), max(0, y - sup_h)
        s[y0:y + sup_h + 1, x0:x + sup_w + 1] = -1.0
    return out


def _empty(t0: float, **kw) -> dict:
    res = {"found": False, "x": 0, "y": 0, "w": 0, "h": 0, "score": 0.0, "scale": 0.0,
           "method": "none", "ambiguous": False, "runner_up": 0.0,
           "elapsed_ms": 0.0}
    res.update(kw)
    res["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    return res


def _template_search(gs: np.ndarray, gt: np.ndarray, scales) -> list[dict]:
    """Coarse-to-fine multi-scale search. Returns refined candidates sorted by score."""
    H, W = gs.shape
    th0, tw0 = gt.shape
    pyr = [gs]
    for _ in range(MAX_PYR_LEVEL):
        if min(pyr[-1].shape) < 64:
            break
        pyr.append(cv2.pyrDown(pyr[-1]))
    pyr_edges: dict[int, np.ndarray] = {}

    cands = []
    for s in scales:
        tw, th = int(round(tw0 * s)), int(round(th0 * s))
        if min(tw, th) < MIN_SCALED_SIDE or tw * th < MIN_SCALED_AREA or tw > W or th > H:
            continue  # too few pixels left for a meaningful correlation
        lvl = 0
        while lvl + 1 < len(pyr) and min(tw, th) * 0.5 ** (lvl + 1) >= COARSE_MIN_SIDE:
            lvl += 1
        f = 0.5 ** lvl
        img = pyr[lvl]
        ctw, cth = max(3, int(round(tw * f))), max(3, int(round(th * f)))
        if ctw > img.shape[1] or cth > img.shape[0]:
            continue
        tpl = _resize(gt, ctw, cth)
        if float(tpl.std()) < 1.0:
            continue
        if lvl not in pyr_edges:
            pyr_edges[lvl] = _edges(img)
        score = _combined(img, pyr_edges[lvl], tpl, _edges(tpl))
        for sc, x, y in _top_peaks(score, PEAKS_PER_SCALE, max(1, ctw // 2), max(1, cth // 2)):
            cands.append({"coarse": sc, "scale": float(s), "f": f,
                          "x": x / f, "y": y / f, "w": tw, "h": th})
    if not cands:
        return []

    # NMS on centres across scales
    cands.sort(key=lambda c: -c["coarse"])
    kept = []
    for c in cands:
        cx, cy = c["x"] + c["w"] / 2, c["y"] + c["h"] / 2
        ok = True
        for k in kept:
            kx, ky = k["x"] + k["w"] / 2, k["y"] + k["h"] / 2
            r = 0.5 * min(min(k["w"], k["h"]), min(c["w"], c["h"]))
            if abs(cx - kx) <= max(r, 4) and abs(cy - ky) <= max(r, 4):
                ok = False
                break
        if ok:
            kept.append(c)
        if len(kept) >= MAX_CANDIDATES:
            break

    # full-resolution refinement around each candidate
    smin, smax = min(scales) * 0.94, max(scales) * 1.06

    def refine_at(c: dict, s: float, margin: int) -> dict | None:
        tw, th = int(round(tw0 * s)), int(round(th0 * s))
        if min(tw, th) < MIN_SCALED_SIDE or tw * th < MIN_SCALED_AREA or tw > W or th > H:
            return None
        x0 = int(max(0, np.floor(c["x"] - margin)))
        y0 = int(max(0, np.floor(c["y"] - margin)))
        x1 = int(min(W, np.ceil(c["x"] + c["w"] + margin)))
        y1 = int(min(H, np.ceil(c["y"] + c["h"] + margin)))
        if x1 - x0 < tw or y1 - y0 < th:
            return None
        # edges computed on a padded window so Canny has context at the borders
        px0, py0 = max(0, x0 - 4), max(0, y0 - 4)
        px1, py1 = min(W, x1 + 4), min(H, y1 + 4)
        win_e = _edges(gs[py0:py1, px0:px1])[y0 - py0:y1 - py0, x0 - px0:x1 - px0]
        tpl = _resize(gt, tw, th)
        sc = _combined(gs[y0:y1, x0:x1], win_e, tpl, _edges(tpl))
        _, mx, _, (lx, ly) = cv2.minMaxLoc(sc)
        return {"score": float(mx), "scale": float(s), "x": x0 + lx, "y": y0 + ly, "w": tw, "h": th}

    refined = []
    for c in kept:
        margin = int(np.ceil(2.0 / c["f"])) + 3 + int(0.06 * max(c["w"], c["h"]))
        best = None
        tried: set[float] = set()
        # pass 1: half-step grid spanning one coarse scale step; pass 2: +-3 % around the winner
        for mults in ((0.89, 0.94, 1.0, 1.06, 1.12), (0.97, 1.03)):
            base = c["scale"] if best is None else best["scale"]
            for m in mults:
                s = round(base * m, 4)
                if s < smin or s > smax or s in tried:
                    continue
                tried.add(s)
                r = refine_at(c, s, margin)
                if r is not None and (best is None or r["score"] > best["score"]):
                    best = r
        if best is not None:
            refined.append(best)
    refined.sort(key=lambda r: -r["score"])
    return refined


def _feature_search(gs: np.ndarray, gt: np.ndarray) -> dict | None:
    """ORB + ratio test + RANSAC homography with geometric and photometric sanity checks."""
    th0, tw0 = gt.shape
    up = 1.0
    if min(th0, tw0) < 64:
        up = min(2.0, 64.0 / min(th0, tw0))
    tq = cv2.resize(gt, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up != 1.0 else gt
    orb_t = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=15,
                           patchSize=15, fastThreshold=8)
    orb_s = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8, edgeThreshold=15,
                           patchSize=15, fastThreshold=8)
    kt, dt = orb_t.detectAndCompute(tq, None)
    if dt is None or len(kt) < MIN_INLIERS:
        return None
    ks, ds = orb_s.detectAndCompute(gs, None)
    if ds is None or len(ks) < MIN_INLIERS:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in bf.knnMatch(dt, ds, k=2):
        if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance:
            good.append(pair[0])
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([kt[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / up
    dst = np.float32([ks[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0, maxIters=3000, confidence=0.995)
    if Hm is None or inl is None:
        return None
    n_inl = int(inl.sum())
    if n_inl < MIN_INLIERS:
        return None
    corners = np.float32([[0, 0], [tw0, 0], [tw0, th0], [0, th0]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(corners, Hm).reshape(-1, 2)
    if not np.all(np.isfinite(q)) or not cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2)):
        return None
    area = cv2.contourArea(q.astype(np.float32))
    scale = float(np.sqrt(max(area, 1e-6) / (tw0 * th0)))
    if not (0.33 <= scale <= 3.0):
        return None
    top, bot = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
    left, right = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
    if min(top, bot) / max(top, bot, 1e-6) < 0.8 or min(left, right) / max(left, right, 1e-6) < 0.8:
        return None
    aspect = ((top + bot) / 2 / tw0) / max(1e-6, (left + right) / 2 / th0)
    if not (0.7 <= aspect <= 1.43):
        return None
    # photometric verification: rectify the found quad back to template frame
    Hinv = np.linalg.inv(Hm)
    rect = cv2.warpPerspective(gs, Hinv, (tw0, th0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ncc = float(np.nan_to_num(cv2.matchTemplate(rect, gt, cv2.TM_CCOEFF_NORMED)[0, 0]))
    if ncc < FEATURE_VERIFY_NCC:
        return None
    x0, y0 = q.min(axis=0)
    x1, y1 = q.max(axis=0)
    return {"score": ncc, "scale": scale, "cx": float(q[:, 0].mean()), "cy": float(q[:, 1].mean()),
            "w": int(round(x1 - x0)), "h": int(round(y1 - y0)), "inliers": n_inl}


def locate(screenshot: np.ndarray, template: np.ndarray, *, min_score: float = 0.80, scales=None,
           search_roi=None, use_features: bool = True) -> dict:
    """Find ``template`` (a UI element crop) on ``screenshot``.

    Args:
        screenshot: BGR uint8 screen image.
        template: BGR uint8 crop of the element to find.
        min_score: acceptance threshold on the combined NCC score.
        scales: iterable of template->screenshot scale factors to try
            (default: 13 geometric steps from 0.5 to 2.0).
        search_roi: optional ``[x, y, w, h]`` restricting the search; returned
            coordinates are still in full-screenshot pixels.
        use_features: enable the ORB/RANSAC fallback (default ``True``).

    Returns:
        dict with ``found, x, y`` (match centre), ``w, h, score, scale, method``
        (``"template" | "features" | "none"``), ``ambiguous, runner_up,
        elapsed_ms``; plus ``template_ms`` (time spent in the template stage
        alone) and ``inliers`` for feature matches. When ``found`` is False the coordinates are the best
        (rejected) template candidate, for debugging only, and must not be used.
    """
    t0 = time.perf_counter()
    shot = to_bgr(screenshot)
    gs = to_gray(shot)
    gt = to_gray(to_bgr(template))
    ox = oy = 0
    if search_roi is not None:
        x, y, w, h = [int(v) for v in search_roi]
        H, W = gs.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return _empty(t0)
        gs = gs[y0:y1, x0:x1]
        ox, oy = x0, y0
    if gt.shape[0] < 4 or gt.shape[1] < 4 or float(gt.std()) < MIN_TEMPLATE_STD:
        return _empty(t0)  # uniform patch: any position would be a guess
    scales = tuple(float(s) for s in (scales if scales is not None else DEFAULT_SCALES))

    refined = _template_search(gs, gt, scales)
    template_ms = (time.perf_counter() - t0) * 1000.0
    best = refined[0] if refined else None
    runner = 0.0
    if best is not None:
        bcx, bcy = best["x"] + best["w"] / 2, best["y"] + best["h"] / 2
        for r in refined[1:]:
            rcx, rcy = r["x"] + r["w"] / 2, r["y"] + r["h"] / 2
            if abs(rcx - bcx) > 0.5 * best["w"] or abs(rcy - bcy) > 0.5 * best["h"]:
                runner = max(runner, r["score"])
                break
    if best is not None and best["score"] >= min_score:
        res = {"found": True,
               "x": int(round(best["x"] + best["w"] / 2 + ox)), "y": int(round(best["y"] + best["h"] / 2 + oy)),
               "w": int(best["w"]), "h": int(best["h"]), "score": float(best["score"]),
               "scale": float(best["scale"]), "method": "template",
               "ambiguous": bool(best["score"] - runner < AMBIGUITY_MARGIN), "runner_up": float(runner),
               "template_ms": template_ms}
        res["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
        return res

    if use_features:
        fr = _feature_search(gs, gt)
        if fr is not None:
            res = {"found": True, "x": int(round(fr["cx"] + ox)), "y": int(round(fr["cy"] + oy)),
                   "w": fr["w"], "h": fr["h"], "score": float(fr["score"]), "scale": float(fr["scale"]),
                   "method": "features", "ambiguous": False, "runner_up": float(runner),
                   "inliers": fr["inliers"], "template_ms": template_ms}
            res["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
            return res

    if best is None:
        return _empty(t0, template_ms=template_ms)
    return _empty(t0, template_ms=template_ms, x=int(round(best["x"] + best["w"] / 2 + ox)), y=int(round(best["y"] + best["h"] / 2 + oy)),
                  w=int(best["w"]), h=int(best["h"]), score=float(best["score"]), scale=float(best["scale"]),
                  runner_up=float(runner))
