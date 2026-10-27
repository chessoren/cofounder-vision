"""Turn a screen recording of a person doing a workflow into replayable steps.

Two passes over the video with ``cv2.VideoCapture``:

**Pass 1 (analysis, at ``sample_fps``).** Every sampled frame is converted to
grayscale and downscaled to at most ``analysis_width`` pixels wide.

* *Cursor tracking* - :class:`~cofounder_vision.cursor.CursorTracker`: masked
  ``TM_CCOEFF_NORMED`` against a synthetic arrow (``cv2.fillPoly``), sizes
  16-48 px (video pixels), searched in a window around the previous position
  and over the full frame when lost.
* *Frame-to-frame change* - :func:`~cofounder_vision.verify.change_components`
  between consecutive samples with the cursor boxes masked out (so pointer
  motion is not "change"), plus a 64-bin gray histogram correlation. A
  hysteresis state machine on the changed fraction (enter above ``HI``, leave
  after ``MIN_STABLE`` samples below ``LO``) yields stable segments /
  transitions.
* *Event inference*, for each sample ``j`` with a real change:
    - **click**: the cursor was present and still (<= ``STILL_TOL`` px) over
      samples ``j-2 -> j-1`` or ``j-1 -> j`` (>= 1 / sample_fps >= 150 ms at
      <= 6.6 fps), the
      cursor moved since the previous click, and the change is a global
      transition or has a region within ``NEAR_FRAC`` x width of the cursor
      tip. Changes within ``REFRACTORY_S`` after a click while the cursor stays
      put belong to that click (e.g. pressed state, then navigation). The click
      time is moved back to the earliest *weak* (sub-threshold) change near the
      resting cursor, when there is one (press feedback before a slow update).
    - **hover suppression**: a change right under the cursor in the sample
      where it *arrives* is recorded as a hover box and ignored; a change
      overlapping that box when the cursor *leaves* is ignored too.
    - **type**: small, line-shaped local changes while the cursor is not
      moving (or hidden), that keep growing inside one region (successive
      changes within ``TYPE_GAP_S`` and adjacent to the running union box).
      Needs >= 2 changed samples.
    - **transition**: a global screen change that could not be attributed to a
      click (e.g. keyboard shortcut, or a missed cursor).

**Pass 2 (full resolution).** Only the frames needed for keyframes and target
crops are decoded again. The target crop (about 160x64 px at 1440 px width,
scaled with the video width) is cut from the frame *before* the click, centred
on the cursor tip, and the cursor itself is removed by copying the pixels
from an earlier frame in which that area was identical and the cursor was
elsewhere (``cv2.inpaint`` as a fallback), so the crop can later be matched by
:func:`~cofounder_vision.locate.locate` on cursor-free screenshots.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .cursor import CursorTracker
from .imgio import clip_rect, merge_boxes, png_b64
from .signature import screen_signature
from .verify import change_components, chroma

HI = 0.02
LO = 0.002
MIN_STABLE = 2
GLOBAL_FRAC = 0.08
GLOBAL_SPREAD = 0.25
HIST_CORR_GLOBAL = 0.90
STILL_TOL = 2.5
NEAR_FRAC = 0.07
REFRACTORY_S = 0.6
TYPE_GAP_S = 1.6
WEAK_THRESHOLD = 12
WEAK_MIN_PIXELS = 12
WEAK_LOOKBACK = 5
TYPE_MAX_H_FRAC = 0.08
TYPE_MAX_AREA_FRAC = 0.03
CURSOR_SIZES_VIDEO = (16, 20, 24, 28, 32, 40, 48)
TARGET_W, TARGET_H, TARGET_REF_W = 160, 64, 1440


def _hist(g: np.ndarray) -> np.ndarray:
    h = cv2.calcHist([g], [0], None, [64], [0, 256])
    cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1)
    return h


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_box_dist(p, box) -> float:
    x, y, w, h = box
    dx = max(x - p[0], 0, p[0] - (x + w))
    dy = max(y - p[1], 0, p[1] - (y + h))
    return math.hypot(dx, dy)


def _union(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def _cursor_mask(shape, dets, pad=3) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for d in dets:
        if d is None:
            continue
        x, y, w, h = d["bbox"]
        cv2.rectangle(m, (x - pad, y - pad), (x + w + pad, y + h + pad), 255, -1)
    return m


def _read_samples(path: str, step: int, analysis_width: int):
    """Pass 1: yield (frame_index, gray_small, chroma_half, (W, H)) for every ``step``-th frame.

    ``chroma_half`` is the Cr/Cb pair at half the analysis resolution (keeps
    memory at ~1.5 bytes per analysed pixel while still seeing colour-only changes).
    """
    cap = cv2.VideoCapture(path)
    idx = 0
    try:
        while True:
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break
                H, W = frame.shape[:2]
                f = min(1.0, analysis_width / W)
                small = frame
                if f < 1.0:
                    small = cv2.resize(frame, (int(round(W * f)), int(round(H * f))), interpolation=cv2.INTER_AREA)
                g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                half = cv2.resize(small, (max(1, small.shape[1] // 2), max(1, small.shape[0] // 2)),
                                  interpolation=cv2.INTER_AREA)
                yield idx, g, chroma(half), (W, H)
            idx += 1
    finally:
        cap.release()


def _read_frames(path: str, wanted: set[int]) -> dict[int, np.ndarray]:
    """Pass 2: decode only the frame indices in ``wanted`` (full resolution BGR)."""
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    last = max(wanted)
    cap = cv2.VideoCapture(path)
    idx = 0
    try:
        while idx <= last:
            if not cap.grab():
                break
            if idx in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    out[idx] = frame
            idx += 1
    finally:
        cap.release()
    return out


def segment_recording(video_path, *, sample_fps: float = 5, analysis_width: int = 1280,
                      keyframe_max_width: int | None = 960) -> dict:
    """Segment a screen recording into click / type steps.

    Args:
        video_path: path to an mp4 / mov readable by ``cv2.VideoCapture``.
        sample_fps: analysis rate (frames per second actually examined).
        analysis_width: frames wider than this are downscaled for analysis.
        keyframe_max_width: keyframes are downscaled to this width before PNG
            encoding to keep payloads small (``None`` keeps full resolution).
            Target crops are always full resolution.

    Returns:
        ``{fps, duration_s, frames_analyzed, width, height, steps, stats}``; each
        step has ``index, action ("click"|"type"|"transition"), t_start, t_end,
        keyframe_png_b64, click {x, y, t} | None, target_png_b64 | None,
        target_bbox | None, change_regions, signature``. All coordinates are in
        video pixels.
    """
    path = str(Path(video_path))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if not fps or fps != fps or fps > 1000:
        fps = 30.0
    step = max(1, int(round(fps / float(sample_fps))))
    dt = step / fps

    idxs: list[int] = []
    grays: list[np.ndarray] = []
    chromas: list[np.ndarray] = []
    curs: list[dict | None] = []
    tracker = None
    vid_wh = None
    f = 1.0
    for idx, g, chh, wh in _read_samples(path, step, analysis_width):
        if tracker is None:
            vid_wh = wh
            f = g.shape[1] / wh[0]
            sizes = sorted({max(8, int(round(s * f))) for s in CURSOR_SIZES_VIDEO})
            tracker = CursorTracker(sizes=sizes, min_score=0.80, window=max(60, int(160 * f)))
        idxs.append(idx)
        grays.append(g)
        chromas.append(chh)
        curs.append(tracker.update(g))
    n = len(grays)
    if n == 0:
        raise ValueError(f"no frames decoded from {path}")
    W, H = vid_wh
    duration = idxs[-1] / fps + 1.0 / fps
    Ha, Wa = grays[0].shape
    area_a = float(Ha * Wa)

    def chf(j: int) -> np.ndarray:
        return cv2.resize(chromas[j], (Wa, Ha), interpolation=cv2.INTER_LINEAR)

    near_r = NEAR_FRAC * Wa

    # ---- per-sample change measurements --------------------------------
    frac = [0.0] * n
    regions: list[list[list[int]]] = [[] for _ in range(n)]
    is_global = [False] * n
    weak = [False] * n
    for j in range(1, n):
        mask = _cursor_mask(grays[j].shape, [curs[j - 1], curs[j]])
        kept, boxes, _ = change_components(grays[j - 1], grays[j], ignore_cursor=True, ignore_mask=mask,
                                           cb=chf(j - 1), ca=chf(j))
        frac[j] = float(np.count_nonzero(kept)) / area_a
        regions[j] = merge_boxes(boxes, gap=8)
        cells = cv2.resize((kept > 0).astype(np.float32), (16, 10), interpolation=cv2.INTER_AREA)
        spread = float(np.mean(cells >= 0.005))  # grid coverage, as in verify_change
        hc = float(cv2.compareHist(_hist(grays[j - 1]), _hist(grays[j]), cv2.HISTCMP_CORREL))
        is_global[j] = frac[j] >= GLOBAL_FRAC or spread >= GLOBAL_SPREAD or hc < HIST_CORR_GLOBAL
        # weak evidence near the cursor (sub-threshold press feedback), used only for click timing
        c = curs[j - 1]
        if c is not None:
            r = int(near_r / 2)
            x0, y0 = max(0, c["x"] - r), max(0, c["y"] - r)
            x1, y1 = min(Wa, c["x"] + r), min(Ha, c["y"] + r)
            d = cv2.absdiff(cv2.GaussianBlur(grays[j - 1][y0:y1, x0:x1], (3, 3), 0),
                            cv2.GaussianBlur(grays[j][y0:y1, x0:x1], (3, 3), 0))
            d[mask[y0:y1, x0:x1] > 0] = 0
            weak[j] = int(np.count_nonzero(d > WEAK_THRESHOLD)) >= WEAK_MIN_PIXELS

    # ---- hysteresis: stable segments / transitions -----------------------
    transitions = 0
    in_trans = False
    calm = 0
    stable_after = [False] * n  # sample j is part of a stable run
    for j in range(n):
        if not in_trans and frac[j] > HI:
            in_trans = True
            transitions += 1
            calm = 0
        elif in_trans:
            calm = calm + 1 if frac[j] < LO else 0
            if calm >= MIN_STABLE:
                in_trans = False
        stable_after[j] = not in_trans and frac[j] < LO

    # ---- event inference -------------------------------------------------
    def moved(a, b) -> bool | None:
        if a is None or b is None:
            return None
        return _dist((a["x"], a["y"]), (b["x"], b["y"])) > STILL_TOL

    events: list[dict] = []
    last_click: dict | None = None
    moved_since_click = True
    typing: dict | None = None
    hover_box: list[int] | None = None

    def overlaps(a, b) -> bool:
        return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]

    def close_typing():
        nonlocal typing
        if typing is not None and typing["count"] >= 2:
            events.append(typing)
        typing = None

    for j in range(1, n):
        if last_click is not None and (moved(curs[j - 1], curs[j])
                                       or moved(last_click["cursor"], curs[j - 1])):
            moved_since_click = True  # the pointer has left the last click position
        active = bool(regions[j]) or is_global[j]
        if not active:
            continue
        t_j = idxs[j] / fps
        cprev = curs[j - 1]
        mvj = moved(cprev, curs[j])
        if mvj and not is_global[j]:
            cj = curs[j]
            # hover-on: the cursor just arrived and the change sits right under it
            on = [r for r in regions[j] if _point_box_dist((cj["x"], cj["y"]), r) <= 4]
            if on:
                hover_box = on[0]
                continue
            # hover-off: the cursor is leaving and the change is where the hover appeared
            if hover_box is not None and any(overlaps(r, hover_box) for r in regions[j]):
                hover_box = None
                continue
        # absorb follow-up changes of the last click (pressed state -> page load)
        if (last_click is not None and not moved_since_click
                and t_j - last_click["t_last"] <= REFRACTORY_S):
            last_click["t_last"] = t_j
            last_click["j_last"] = j
            continue
        # still for >= one sample interval (>= 150 ms at <= 6.6 fps) before or across the change
        still = cprev is not None and (
            (j >= 2 and curs[j - 2] is not None and not moved(curs[j - 2], cprev)) or mvj is False)
        near = cprev is not None and any(_point_box_dist((cprev["x"], cprev["y"]), r) <= near_r for r in regions[j])
        if still and moved_since_click and (is_global[j] or near):
            close_typing()
            # timing: earliest weak near-cursor change while the cursor rested here
            # (press feedback often precedes a slow page update)
            k0 = j
            k = j - 1
            while k >= 1 and j - k <= WEAK_LOOKBACK and curs[k] is not None and curs[k - 1] is not None \
                    and not moved(curs[k - 1], curs[k]):
                if weak[k] and not regions[k]:
                    k0 = k
                k -= 1
            ev = {"action": "click", "j": j, "j_last": j, "t": (idxs[k0 - 1] + idxs[k0]) / 2 / fps,
                  "t_last": t_j, "cursor": cprev}
            events.append(ev)
            last_click = ev
            moved_since_click = False
            continue
        mv = moved(cprev, curs[j])
        if not is_global[j] and regions[j] and not mv:
            u = _union(regions[j])
            small = u[3] <= TYPE_MAX_H_FRAC * Ha and u[2] * u[3] <= TYPE_MAX_AREA_FRAC * area_a
            if small:
                if typing is not None and t_j - typing["t_last"] <= TYPE_GAP_S:
                    tb = typing["bbox"]
                    ex = [tb[0] - 3 * tb[3] - 8, tb[1] - tb[3], tb[2] + 6 * tb[3] + 16, 3 * tb[3]]
                    if _point_box_dist((u[0] + u[2] / 2, u[1] + u[3] / 2), ex) == 0:
                        typing["bbox"] = _union([tb, u])
                        typing["count"] += 1
                        typing["t_last"] = t_j
                        typing["j_last"] = j
                        continue
                close_typing()
                typing = {"action": "type", "j": j, "j_last": j, "t": (idxs[j - 1] + idxs[j]) / 2 / fps,
                          "t_last": t_j, "bbox": u, "count": 1, "cursor": None}
                continue
        if is_global[j]:
            close_typing()
            events.append({"action": "transition", "j": j, "j_last": j, "t": (idxs[j - 1] + idxs[j]) / 2 / fps,
                           "t_last": t_j, "cursor": None})
    close_typing()
    events.sort(key=lambda e: e["t"])

    # ---- settle frame per event, cursor-free source frames --------------
    plans = []
    for k, ev in enumerate(events):
        j0 = ev["j"] - 1  # pre-action sample
        nxt = events[k + 1]["j"] - 1 if k + 1 < len(events) else n - 1
        j1 = ev["j_last"]
        while j1 < nxt and not stable_after[j1]:
            j1 += 1  # advance to the first settled sample (before the next event)
        mask = _cursor_mask(grays[j0].shape, [curs[j0], curs[j1]])
        _, boxes, _ = change_components(grays[j0], grays[j1], ignore_cursor=True, ignore_mask=mask)
        plans.append({"ev": ev, "j0": j0, "j1": j1, "boxes": merge_boxes(boxes, gap=10)})

    scale_t = min(3.0, max(0.5, W / TARGET_REF_W))
    tw_v, th_v = int(round(TARGET_W * scale_t)), int(round(TARGET_H * scale_t))
    wanted: set[int] = set()
    for p in plans:
        ev = p["ev"]
        wanted.add(idxs[p["j0"]])
        if ev["action"] == "click":
            c = ev["cursor"]
            cx, cy = c["x"] / f, c["y"] / f
            p["crop"] = clip_rect(int(round(cx - tw_v / 2)), int(round(cy - th_v / 2)), tw_v, th_v, W, H)
        elif ev["action"] == "type":
            bx, by, bw, bh = [v / f for v in ev["bbox"]]
            padx, pady = max(16.0, 0.5 * bh), max(8.0, 0.5 * bh)
            p["crop"] = clip_rect(int(bx - padx), int(by - pady), int(bw + 2 * padx), int(bh + 2 * pady), W, H)
        else:
            p["crop"] = None
        # find an earlier sample where the crop is unchanged and the cursor is outside it
        p["clean_j"] = None
        if p["crop"] is not None:
            cx0, cy0, cw, ch = [int(round(v * f)) for v in p["crop"]]
            if cw > 0 and ch > 0:
                ref = grays[p["j0"]][cy0:cy0 + ch, cx0:cx0 + cw].astype(np.int16)
                cm = _cursor_mask(grays[p["j0"]].shape, [curs[p["j0"]]], pad=4)[cy0:cy0 + ch, cx0:cx0 + cw] == 0
                for k in range(p["j0"], max(-1, p["j0"] - 40), -1):
                    ck = curs[k]
                    if ck is not None:
                        bx, by, bw, bh = ck["bbox"]
                        if bx < cx0 + cw and bx + bw > cx0 and by < cy0 + ch and by + bh > cy0:
                            continue
                    cand = grays[k][cy0:cy0 + ch, cx0:cx0 + cw].astype(np.int16)
                    if cm.any() and float(np.abs(cand - ref)[cm].mean()) < 2.0:
                        p["clean_j"] = k
                        wanted.add(idxs[k])
                        break
    frames = _read_frames(path, wanted)

    # ---- build steps -------------------------------------------------------
    steps = []
    for k, p in enumerate(plans):
        ev = p["ev"]
        pre = frames.get(idxs[p["j0"]])
        if pre is None:
            continue
        key = pre
        if keyframe_max_width and pre.shape[1] > keyframe_max_width:
            s = keyframe_max_width / pre.shape[1]
            key = cv2.resize(pre, (keyframe_max_width, int(round(pre.shape[0] * s))), interpolation=cv2.INTER_AREA)
        target_b64 = None
        bbox = None
        if p["crop"] is not None and p["crop"][2] > 0 and p["crop"][3] > 0:
            x, y, w, h = p["crop"]
            crop = pre[y:y + h, x:x + w].copy()
            c = curs[p["j0"]]
            if c is not None:
                bx, by, bw, bh = [int(round(v / f)) for v in c["bbox"]]
                m = np.zeros(pre.shape[:2], np.uint8)
                cv2.rectangle(m, (bx - 3, by - 3), (bx + bw + 3, by + bh + 3), 255, -1)
                mc = m[y:y + h, x:x + w]
                if mc.any():
                    clean = frames.get(idxs[p["clean_j"]]) if p["clean_j"] is not None else None
                    if clean is not None:
                        crop[mc > 0] = clean[y:y + h, x:x + w][mc > 0]
                    else:
                        crop = cv2.inpaint(crop, mc, 5, cv2.INPAINT_TELEA)
            target_b64 = png_b64(crop)
            bbox = [int(x), int(y), int(w), int(h)]
        click = None
        if ev["action"] == "click":
            c = ev["cursor"]
            click = {"x": int(round(c["x"] / f)), "y": int(round(c["y"] / f)), "t": round(float(ev["t"]), 3)}
        t_end = plans[k + 1]["ev"]["t"] if k + 1 < len(plans) else duration
        steps.append({
            "index": len(steps),
            "action": ev["action"],
            "t_start": round(float(ev["t"]), 3),
            "t_end": round(float(t_end), 3),
            "keyframe_png_b64": png_b64(key),
            "click": click,
            "target_png_b64": target_b64,
            "target_bbox": bbox,
            "change_regions": [[int(round(v / f)) for v in r] for r in p["boxes"]],
            "signature": screen_signature(pre),
        })

    found = sum(1 for c in curs if c is not None)
    return {
        "fps": fps,
        "duration_s": round(duration, 3),
        "frames_analyzed": n,
        "width": int(W),
        "height": int(H),
        "steps": steps,
        "stats": {
            "transitions": transitions,
            "clicks_detected": sum(1 for s in steps if s["action"] == "click"),
            "typing_detected": sum(1 for s in steps if s["action"] == "type"),
            "unattributed_transitions": sum(1 for s in steps if s["action"] == "transition"),
            "cursor_found_ratio": round(found / n, 3),
            "sample_interval_s": round(dt, 4),
        },
    }
