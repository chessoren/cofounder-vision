"""Reproducible benchmark for cofounder_vision.

Usage (from vision/):  uv run python eval/run_eval.py [--quick]

Writes eval/results.json, eval/RESULTS.md and eval/failures/*.png.
Every number in RESULTS.md is computed by this script on this run; nothing is
hard-coded. Seeds are fixed (SEED) so reruns produce the same data.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import shutil
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import cofounder_vision as cvv  # noqa: E402
from synth import SYNTH_SCREENS, Recorder, screen_dashboard, screen_table  # noqa: E402

SEED = 20261026
DATA = os.path.join(HERE, "data")
FAIL = os.path.join(HERE, "failures")
VIDEOS = os.path.join(HERE, "videos")
REAL_VIDEO = os.path.join(os.path.dirname(ROOT), "demo-app", "recordings", "approve-invoice.mp4")
GT_COLOR = (60, 170, 40)  # green = ground truth in failure images; brand blue = prediction

REAL_SCREENS = {
    "cofounder_home_1440x900": "cofounder_home_1440x900.png",
    "cofounder_home_retina_2048x1400": "cofounder_home_retina_2048x1400.png",
    "cofounder_home_900x1100": "cofounder_home_900x1100.png",
    "cofounder_home_1280x800": "cofounder_home_1280x800.png",
}
# templates are sampled from these (1280x800 real screen is kept for verify/video use)
TEMPLATE_SOURCES = ["cofounder_home_1440x900", "cofounder_home_retina_2048x1400", "cofounder_home_900x1100",
                    "synth_form", "synth_table", "synth_dashboard"]
CONDITIONS = ["translate", "scale_0.5", "scale_0.8", "scale_1.25", "scale_2.0", "brightness_contrast",
              "blur_s1.5", "jpeg_q40", "occlusion_20", "negative"]


def family(name: str) -> str:
    return "cofounder" if name.startswith("cofounder") else "synth"


def load_screens() -> dict[str, np.ndarray]:
    screens = {k: cv2.imread(os.path.join(DATA, v), cv2.IMREAD_COLOR) for k, v in REAL_SCREENS.items()}
    for k, fn in SYNTH_SCREENS.items():
        img, _ = fn()
        screens[k] = img
        cv2.imwrite(os.path.join(DATA, f"{k}.png"), img)
    for k, v in screens.items():
        if v is None:
            raise FileNotFoundError(k)
    return screens


# --------------------------------------------------------------------------
# Element proposals (contours on Canny) + sanity filters
# --------------------------------------------------------------------------
def propose_elements(img: np.ndarray, rng, k: int, margin: int = 4) -> list[list[int]]:
    """Button/field-like boxes: Canny -> dilate -> external contours -> filters.

    Filters: size 24-320 x 16-120 px (scaled for Retina), aspect 0.3-10, gray
    std >= 15, edge density 2-45 %, not on the border, and *unique* in their own
    image (second NCC peak < 0.90) so the ground truth location is well defined.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Hh, Ww = g.shape
    f = 2.0 if Ww >= 2000 else 1.0
    edges = cv2.Canny(g, 40, 120)
    d = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=2)
    cnts, _ = cv2.findContours(d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = x - margin, y - margin, w + 2 * margin, h + 2 * margin
        if not (24 * f <= w <= 320 * f and 16 * f <= h <= 120 * f and 0.3 <= w / h <= 10):
            continue
        if x < 2 or y < 2 or x + w > Ww - 2 or y + h > Hh - 2:
            continue
        patch = g[y:y + h, x:x + w]
        if patch.std() < 15:
            continue
        dens = float(np.count_nonzero(edges[y:y + h, x:x + w])) / (w * h)
        if not (0.02 <= dens <= 0.45):
            continue
        boxes.append([x, y, w, h])
    # dedupe by IoU
    boxes.sort(key=lambda b: -b[2] * b[3])
    kept = []
    for b in boxes:
        ok = True
        for k2 in kept:
            ix = max(0, min(b[0] + b[2], k2[0] + k2[2]) - max(b[0], k2[0]))
            iy = max(0, min(b[1] + b[3], k2[1] + k2[3]) - max(b[1], k2[1]))
            if ix * iy > 0.3 * min(b[2] * b[3], k2[2] * k2[3]):
                ok = False
                break
        if ok:
            kept.append(b)
    rng.shuffle(kept)
    out = []
    for b in kept:
        x, y, w, h = b
        r = cv2.matchTemplate(g, g[y:y + h, x:x + w], cv2.TM_CCOEFF_NORMED)
        r[~np.isfinite(r)] = 0
        r[max(0, y - h // 2):y + h // 2 + 1, max(0, x - w // 2):x + w // 2 + 1] = -1
        if float(r.max()) >= 0.90:
            continue
        out.append(b)
        if len(out) >= k:
            break
    return out


# --------------------------------------------------------------------------
# Locate benchmark
# --------------------------------------------------------------------------
def perturb(cond: str, shot: np.ndarray, box, rng, screens, src_name):
    """Return (image, gt_center or None, gt_size or None)."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    Hh, Ww = shot.shape[:2]
    if cond == "translate":
        dx, dy = rng.integers(20, 61) * rng.choice([-1, 1]), rng.integers(20, 61) * rng.choice([-1, 1])
        if not (w <= cx + dx <= Ww - w):
            dx = -dx
        if not (h <= cy + dy <= Hh - h):
            dy = -dy
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(shot, M, (Ww, Hh), borderMode=cv2.BORDER_REPLICATE), (cx + dx, cy + dy), (w, h)
    if cond.startswith("scale_"):
        s = float(cond.split("_")[1])
        img = cv2.resize(shot, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        gcx, gcy = cx * s, cy * s
        CW, CH = 1600, 1000  # bound the search area to a realistic screen size
        H2, W2 = img.shape[:2]
        if W2 > CW or H2 > CH:
            cw, ch = min(CW, W2), min(CH, H2)
            lo_x = int(max(0, min(W2 - cw, gcx + w * s - cw)))
            hi_x = int(max(lo_x, min(W2 - cw, gcx - w * s)))
            lo_y = int(max(0, min(H2 - ch, gcy + h * s - ch)))
            hi_y = int(max(lo_y, min(H2 - ch, gcy - h * s)))
            ox, oy = int(rng.integers(lo_x, hi_x + 1)), int(rng.integers(lo_y, hi_y + 1))
            img = img[oy:oy + ch, ox:ox + cw]
            gcx, gcy = gcx - ox, gcy - oy
        return np.ascontiguousarray(img), (gcx, gcy), (w * s, h * s)
    if cond == "brightness_contrast":
        a, b = float(rng.uniform(0.7, 1.3)), float(rng.uniform(-40, 40))
        return cv2.convertScaleAbs(shot, alpha=a, beta=b), (cx, cy), (w, h)
    if cond == "blur_s1.5":
        return cv2.GaussianBlur(shot, (0, 0), 1.5), (cx, cy), (w, h)
    if cond == "jpeg_q40":
        _, buf = cv2.imencode(".jpg", shot, [cv2.IMWRITE_JPEG_QUALITY, 40])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR), (cx, cy), (w, h)
    if cond == "occlusion_20":
        img = shot.copy()
        ow, oh = int(round(w * 0.447)), int(round(h * 0.447))  # 0.447^2 = 20 % of the area
        corner = int(rng.integers(0, 4))
        ox = x if corner in (0, 3) else x + w - ow
        oy = y if corner in (0, 1) else y + h - oh
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        cv2.rectangle(img, (ox, oy), (ox + ow - 1, oy + oh - 1), col, -1)
        return img, (cx, cy), (w, h)
    if cond == "negative":
        others = [k for k in screens if family(k) != family(src_name)]
        return screens[others[int(rng.integers(0, len(others)))]], None, None
    raise ValueError(cond)


_SCREENS = None


def _init_worker():
    global _SCREENS
    cv2.setNumThreads(1)  # single-threaded latency, closer to a 1-2 vCPU Lambda
    _SCREENS = load_screens_cached()


def load_screens_cached():
    screens = {k: cv2.imread(os.path.join(DATA, v), cv2.IMREAD_COLOR) for k, v in REAL_SCREENS.items()}
    for k in SYNTH_SCREENS:
        screens[k] = cv2.imread(os.path.join(DATA, f"{k}.png"), cv2.IMREAD_COLOR)
    return screens


def _locate_job(job):
    tid, src, box, cond, seed = job
    rng = np.random.default_rng(seed)
    shot = _SCREENS[src]
    x, y, w, h = box
    tmpl = shot[y:y + h, x:x + w].copy()
    img, gt, gsz = perturb(cond, shot, box, rng, _SCREENS, src)
    r = cvv.locate(img, tmpl)
    rec = {"tid": tid, "src": src, "box": box, "cond": cond, "found": r["found"], "method": r["method"],
           "x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"], "score": r["score"], "ambiguous": r["ambiguous"],
           "runner_up": r["runner_up"], "ms": r["elapsed_ms"], "template_ms": r.get("template_ms", r["elapsed_ms"]),
           "gt": None if gt is None else [float(gt[0]), float(gt[1])],
           "gt_size": None if gsz is None else [float(gsz[0]), float(gsz[1])]}
    if gt is not None:
        tol = max(8.0, 0.1 * max(gsz))
        rec["tol"] = tol
        rec["err"] = float(np.hypot(r["x"] - gt[0], r["y"] - gt[1]))
        rec["correct_loc"] = rec["err"] <= tol
    bad = (gt is None and r["found"]) or (gt is not None and not (r["found"] and rec["correct_loc"]))
    if bad:
        rec["fail_png"] = _failure_image(img, tmpl, r, gt, gsz)
    return rec


def _failure_image(img, tmpl, r, gt, gsz) -> bytes:
    """Annotated crop: GT (green) vs prediction (brand blue), template inset top-left."""
    boxes, labels, gboxes = [], [], []
    if r["w"] > 0:
        boxes.append([r["x"] - r["w"] / 2, r["y"] - r["h"] / 2, r["w"], r["h"]])
        labels.append(f"{r['method']} {r['score']:.2f}{' found' if r['found'] else ' rejected'}")
    out = cvv.annotate(img, boxes=boxes, labels=labels)
    if gt is not None:
        out = cvv.annotate(out, boxes=[[gt[0] - gsz[0] / 2, gt[1] - gsz[1] / 2, gsz[0], gsz[1]]], labels=["GT"],
                           color=GT_COLOR)
    pts = [(r["x"], r["y"])] if r["w"] > 0 else []
    if gt is not None:
        pts.append(gt)
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        H, W = out.shape[:2]
        x0, x1 = int(max(0, min(xs) - 300)), int(min(W, max(xs) + 300))
        y0, y1 = int(max(0, min(ys) - 200)), int(min(H, max(ys) + 200))
        out = out[y0:y1, x0:x1].copy()
    th, tw = tmpl.shape[:2]
    if th + 4 < out.shape[0] and tw + 4 < out.shape[1]:
        out[2:2 + th, 2:2 + tw] = tmpl
        cv2.rectangle(out, (1, 1), (tw + 2, th + 2), (0, 0, 0), 1)
    return cvv.encode_png(out)


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def bench_locate(screens, rng, n_templates, workers):
    templates = []
    per = max(1, int(np.ceil(n_templates / len(TEMPLATE_SOURCES))))
    for src in TEMPLATE_SOURCES:
        for b in propose_elements(screens[src], rng, per):
            templates.append((src, b))
    templates = templates[:n_templates] if len(templates) > n_templates else templates
    jobs = []
    for tid, (src, b) in enumerate(templates):
        for ci, cond in enumerate(CONDITIONS):
            jobs.append((tid, src, [int(v) for v in b], cond, SEED + 1000 * tid + ci))
    t0 = time.time()
    with Pool(workers, initializer=_init_worker) as pool:
        recs = pool.map(_locate_job, jobs, chunksize=4)
    wall = time.time() - t0

    rows = {}
    for path in ("template", "template+features"):
        for cond in CONDITIONS:
            rs = [r for r in recs if r["cond"] == cond]
            if path == "template":
                found = [r["found"] and r["method"] == "template" for r in rs]
                lat = [r["template_ms"] for r in rs]
            else:
                found = [r["found"] for r in rs]
                lat = [r["ms"] for r in rs]
            n = len(rs)
            if cond == "negative":
                row = {"n": n, "false_positive_rate": float(np.mean(found))}
            else:
                hit = [f and r["correct_loc"] for f, r in zip(found, rs)]
                wrong = [f and not r["correct_loc"] for f, r in zip(found, rs)]
                row = {"n": n, "hit_rate": float(np.mean(hit)), "miss_rate": float(np.mean([not f for f in found])),
                       "wrong_location_rate": float(np.mean(wrong))}
            amb = [r["ambiguous"] for f, r in zip(found, rs) if f and r["method"] == "template"]
            row["ambiguous_rate"] = float(np.mean(amb)) if amb else 0.0
            row["features_used"] = int(sum(1 for f, r in zip(found, rs) if f and r["method"] == "features"))
            row["latency_ms_p50"] = pct(lat, 50)
            row["latency_ms_p95"] = pct(lat, 95)
            rows[f"{path}|{cond}"] = row
    # overall positive summary
    summ = {}
    for path in ("template", "template+features"):
        pos = [r for r in recs if r["cond"] != "negative"]
        f = [(r["found"] and (r["method"] == "template" or path != "template")) for r in pos]
        summ[path] = {
            "hit_rate_all_positive_conditions": float(np.mean([a and r["correct_loc"] for a, r in zip(f, pos)])),
            "wrong_location_rate": float(np.mean([a and not r["correct_loc"] for a, r in zip(f, pos)])),
            "negative_false_positive_rate": rows[f"{path}|negative"]["false_positive_rate"],
        }
    # failures
    nfail = 0
    for r in recs:
        png = r.pop("fail_png", None)
        if png is not None and nfail < 60:
            fn = f"locate_{r['cond']}_t{r['tid']:02d}.png"
            with open(os.path.join(FAIL, fn), "wb") as fh:
                fh.write(png)
            r["failure_image"] = f"failures/{fn}"
            nfail += 1
    by_src = {}
    for src in TEMPLATE_SOURCES:
        pos = [r for r in recs if r["src"] == src and r["cond"] != "negative"]
        if pos:
            by_src[src] = {"templates": len({r["tid"] for r in pos}),
                           "hit_rate": float(np.mean([r["found"] and r["correct_loc"] for r in pos]))}
    sizes = [templates[i][1][2:] for i in range(len(templates))]
    return {"n_templates": len(templates), "n_runs": len(recs), "wall_s": wall, "rows": rows, "summary": summ,
            "by_source": by_src, "template_sizes_wh_min": [int(min(s[0] for s in sizes)), int(min(s[1] for s in sizes))],
            "template_sizes_wh_max": [int(max(s[0] for s in sizes)), int(max(s[1] for s in sizes))],
            "failures_saved": nfail, "records": recs, "templates": [[s, b] for s, b in templates]}


# --------------------------------------------------------------------------
# verify_change benchmark
# --------------------------------------------------------------------------
VERIFY_CATS = {"button_state": True, "text_insert": True, "navigation": True,
               "cursor_only": False, "caret_blink": False, "jpeg_noise": False}


def make_verify_pair(cat, name, screens, boxes, rng):
    a = screens[name].copy()
    H, W = a.shape[:2]
    f = 2 if W >= 2000 else 1
    b = a.copy()
    box = boxes[int(rng.integers(0, len(boxes)))]
    x, y, w, h = box
    roi = [x - 10 * f, y - 10 * f, w + 20 * f, h + 20 * f]
    if cat == "button_state":
        tint = np.array(rng.choice([[222, 130, 61], [40, 40, 40], [60, 180, 60]]), np.float32)
        al = float(rng.uniform(0.3, 0.5))
        b[y:y + h, x:x + w] = np.clip(b[y:y + h, x:x + w] * (1 - al) + tint * al, 0, 255).astype(np.uint8)
    elif cat == "text_insert":
        txt = "".join(chr(int(c)) for c in rng.integers(97, 123, int(rng.integers(4, 10))))
        tx, ty = int(rng.integers(40 * f, W - 260 * f)), int(rng.integers(60 * f, H - 40 * f))
        cv2.putText(b, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * f, (25, 25, 25), f, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * f, f)
        roi = [tx - 20 * f, ty - th - 20 * f, tw + 40 * f, th + 40 * f]
    elif cat == "navigation":
        others = [k for k in screens if k != name and not (k.startswith("cofounder") and name.startswith("cofounder"))]
        o = screens[others[int(rng.integers(0, len(others)))]]
        b = cv2.resize(o, (W, H), interpolation=cv2.INTER_AREA)
    elif cat == "cursor_only":
        size = int(rng.choice([20, 24, 32])) * f
        p1 = (int(rng.integers(60, W - 60)), int(rng.integers(60, H - 60)))
        ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(30, 200) * f
        p2 = (int(np.clip(p1[0] + dist * np.cos(ang), 40, W - 60)), int(np.clip(p1[1] + dist * np.sin(ang), 40, H - 60)))
        cvv.draw_cursor(a, p1[0], p1[1], size)
        cvv.draw_cursor(b, p2[0], p2[1], size)
        roi = [p2[0] - 80 * f, p2[1] - 32 * f, 160 * f, 64 * f]
    elif cat == "caret_blink":
        cx, cy = int(rng.integers(60, W - 60)), int(rng.integers(60, H - 60))
        cv2.line(a, (cx, cy), (cx, cy + 18 * f), (25, 25, 25), int(rng.integers(1, 3)) * f)
        roi = [cx - 80 * f, cy - 23 * f, 160 * f, 64 * f]
    elif cat == "jpeg_noise":
        _, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 75))])
        b = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return a, b, roi


def bench_verify(screens, rng, per_cat):
    names = list(screens)
    boxes = {n: propose_elements(screens[n], np.random.default_rng(SEED + i), 30) for i, n in enumerate(names)}
    recs = []
    cm = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    lat = []
    for cat, should in VERIFY_CATS.items():
        for name in names:
            for k in range(per_cat):
                a, b, roi = make_verify_pair(cat, name, screens, boxes[name], rng)
                t0 = time.perf_counter()
                r = cvv.verify_change(a, b, roi=roi)
                lat.append((time.perf_counter() - t0) * 1000)
                pred = r["changed"] or r["verdict"] == "unexpected_global"  # "something changed" either way
                key = ("tp" if pred else "fn") if should else ("fp" if pred else "tn")
                cm[key] += 1
                recs.append({"cat": cat, "screen": name, "expected_changed": should, "pred_changed": pred,
                             "verdict": r["verdict"], "local": r["local_change"], "global": r["global_change"],
                             "ssim_local": r["ssim_local"], "ssim_global": r["ssim_global"]})
                if pred != should:
                    fn = f"verify_{cat}_{name}_{k}.png"
                    side = np.hstack([cvv.annotate(a, boxes=[roi], labels=["roi before"]),
                                      cvv.annotate(b, boxes=[roi] + r["regions"][:8],
                                                   labels=[f"roi after: {r['verdict']}"])])
                    cv2.imwrite(os.path.join(FAIL, fn), side)
    # unexpected-global set: lower/right part of the screen replaced, ROI in the untouched top-left
    ug = {"n": 0, "unexpected_global": 0, "other": {}}
    for name in names:
        for k in range(per_cat):
            a = screens[name]
            H, W = a.shape[:2]
            others = [o for o in names if family(o) != family(name)]
            o = cv2.resize(screens[others[int(rng.integers(0, len(others)))]], (W, H), interpolation=cv2.INTER_AREA)
            sx, sy = int(W * rng.uniform(0.3, 0.4)), int(H * rng.uniform(0.15, 0.3))
            b = a.copy()
            b[sy:, sx:] = o[sy:, sx:]
            cand = [bx for bx in boxes[name] if bx[0] + bx[2] + 12 < sx or bx[1] + bx[3] + 12 < sy]
            if not cand:
                cand = [[10, 10, max(20, sx - 30), max(20, sy - 30)]]
            bx = cand[int(rng.integers(0, len(cand)))]
            roi = [bx[0] - 6, bx[1] - 6, bx[2] + 12, bx[3] + 12]
            roi = [max(0, roi[0]), max(0, roi[1]), min(roi[2], sx - max(0, roi[0])) if roi[1] + roi[3] > sy else roi[2],
                   roi[3] if roi[0] + roi[2] < sx else min(roi[3], sy - max(0, roi[1]))]
            r = cvv.verify_change(a, b, roi=roi)
            ug["n"] += 1
            if r["verdict"] == "unexpected_global":
                ug["unexpected_global"] += 1
            else:
                ug["other"][r["verdict"]] = ug["other"].get(r["verdict"], 0) + 1
                fn = f"verify_unexpected_global_{name}_{k}.png"
                cv2.imwrite(os.path.join(FAIL, fn), cvv.annotate(b, boxes=[roi], labels=[r["verdict"]]))
    per = {}
    for cat in VERIFY_CATS:
        rs = [r for r in recs if r["cat"] == cat]
        per[cat] = {"n": len(rs), "accuracy": float(np.mean([r["pred_changed"] == r["expected_changed"] for r in rs])),
                    "verdicts": {v: sum(1 for r in rs if r["verdict"] == v) for v in sorted({r["verdict"] for r in rs})}}
    n = sum(cm.values())
    return {"confusion": cm, "accuracy": (cm["tp"] + cm["tn"]) / n, "n": n, "per_category": per,
            "unexpected_global": ug, "latency_ms_p50": pct(lat, 50), "latency_ms_p95": pct(lat, 95), "records": recs}


# --------------------------------------------------------------------------
# screen_signature benchmark
# --------------------------------------------------------------------------
def bench_signature(screens, rng):
    names = list(screens)
    same, diff = [], []
    for n in names:
        a = screens[n]
        s0 = cvv.screen_signature(a)
        variants = {
            "jpeg_q50": cv2.imdecode(cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 50])[1], cv2.IMREAD_COLOR),
            "rescale_0.5x": cv2.resize(a, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
            "cursor+text": a.copy(),
        }
        cvv.draw_cursor(variants["cursor+text"], a.shape[1] // 2, a.shape[0] // 2, 24)
        cv2.putText(variants["cursor+text"], "typed text", (60, a.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)
        for k, v in variants.items():
            same.append((n, k, cvv.compare_screens(s0, cvv.screen_signature(v))))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = cvv.compare_screens(cvv.screen_signature(screens[a]), cvv.screen_signature(screens[b]))
            diff.append((a, b, r))
    # cofounder_* real screens are the same app page at different window sizes: reported separately
    diff_layout = [d for d in diff if family(d[0]) == family(d[1]) == "cofounder"]
    diff_true = [d for d in diff if not (family(d[0]) == family(d[1]) == "cofounder")]
    return {"same_pairs": len(same), "same_detected": sum(r["same"] for *_, r in same),
            "different_pairs": len(diff_true), "different_called_same": sum(r["same"] for *_, r in diff_true),
            "same_page_resized_pairs": len(diff_layout), "same_page_resized_called_same": sum(r["same"] for *_, r in diff_layout),
            "same_similarity_min": min(r["similarity"] for *_, r in same),
            "different_similarity_max": max(r["similarity"] for *_, r in diff_true),
            "misses": [[a, k] for a, k, r in same if not r["same"]]}


# --------------------------------------------------------------------------
# segment_recording benchmark
# --------------------------------------------------------------------------
def _focus(box, prev=None):
    def fx(scr):
        if prev is not None:
            x, y, w, h = prev
            cv2.rectangle(scr, (x - 2, y - 2), (x + w + 2, y + h + 2), (248, 248, 248), 3)
            cv2.rectangle(scr, (x, y), (x + w, y + h), (195, 195, 195), 1)
        x, y, w, h = box
        cv2.rectangle(scr, (x - 2, y - 2), (x + w + 2, y + h + 2), (222, 130, 61), 3, cv2.LINE_AA)
    return fx


def _replace(img):
    def fx(scr):
        scr[:] = cv2.resize(img, (scr.shape[1], scr.shape[0]), interpolation=cv2.INTER_AREA)
    return fx


def make_videos(screens):
    os.makedirs(VIDEOS, exist_ok=True)
    vids = []
    # 1) form: click field, type, click field, type, submit -> navigation
    form, el = SYNTH_SCREENS["synth_form"]()
    p = os.path.join(VIDEOS, "v1_form_typing.mp4")
    rec = Recorder(p, form, seed=1)
    rec.idle(1.0)
    f0, f1 = el["field_0"], el["field_1"]
    rec.click(f0[0] + 120, f0[1] + 17, _focus(f0), label="click company field")
    rec.idle(0.8)
    rec.type_text(f0, "Acme Robotics", label="type company")
    rec.idle(0.8)
    rec.click(f1[0] + 160, f1[1] + 17, _focus(f1, f0), move_s=0.5, label="click email field")
    rec.caret = None
    rec.idle(0.8)
    rec.type_text(f1, "ap@acme.io", label="type email")
    rec.idle(0.9)
    rec.caret = None
    sb = el["submit"]
    rec.click(sb[0] + sb[2] // 2, sb[1] + sb[3] // 2, _replace(screen_table()[0]), label="submit -> table")
    rec.idle(1.5)
    rec.close()
    vids.append({"name": "v1_form_typing", "path": p, "truth": rec.truth, "kind": "synthetic"})

    # 2) real Cofounder home screenshot: input, typing, popup menu, menu item, send -> navigation
    home = screens["cofounder_home_1280x800"]
    p = os.path.join(VIDEOS, "v2_cofounder_home.mp4")
    rec = Recorder(p, home, seed=2)
    rec.idle(1.0)
    saved = {}

    def clear_input(scr):
        cv2.rectangle(scr, (305, 522), (955, 552), (255, 255, 255), -1)

    rec.click(560, 537, clear_input, move_s=0.7, label="click prompt input")
    rec.idle(0.8)
    rec.type_text([300, 520, 650, 34], "Book a table for two at 8pm", char_s=0.1, label="type prompt")
    rec.idle(0.8)
    rec.caret = None

    def open_menu(scr):
        saved["menu"] = scr[400:585, 300:560].copy()
        cv2.rectangle(scr, (300, 400), (560, 585), (255, 255, 255), -1)
        cv2.rectangle(scr, (300, 400), (560, 585), (210, 210, 210), 1)
        for i, m in enumerate(["Claude Opus 5.5", "Claude Sonnet 4.6", "Claude Haiku 4.5", "Fable 1"]):
            _y = 432 + 42 * i
            cv2.putText(scr, m, (318, _y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

    rec.click(386, 601, open_menu, move_s=0.6, label="open model menu")
    rec.idle(1.0)

    def pick(scr):
        scr[400:585, 300:560] = saved["menu"]
        cv2.rectangle(scr, (318, 590), (455, 612), (250, 238, 228), -1)
        cv2.putText(scr, "Claude Opus 5.5", (336, 606), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 110, 50), 1, cv2.LINE_AA)

    rec.click(400, 428, pick, move_s=0.5, label="pick model")
    rec.idle(0.9)
    rec.click(949, 597, _replace(screen_dashboard()[0]), move_s=0.8, label="send -> dashboard")
    rec.idle(1.5)
    rec.close()
    vids.append({"name": "v2_cofounder_home", "path": p, "truth": rec.truth, "kind": "synthetic (real screenshot)"})

    # 3) hard: hover highlight without click, fast double action, click on busy chart
    dash, el = SYNTH_SCREENS["synth_dashboard"]()
    p = os.path.join(VIDEOS, "v3_hard_dashboard.mp4")
    rec = Recorder(p, dash, seed=3)
    rec.idle(0.8)
    sh = el["share"]

    def hover_fx(fr):
        x, y, w, h = sh
        fr[y:y + h, x:x + w] = np.clip(fr[y:y + h, x:x + w].astype(np.int16) - 35, 0, 255).astype(np.uint8)

    rec.hover = (sh, hover_fx)
    rec.move(sh[0] + 50, sh[1] + 18, 0.7)
    rec.idle(1.2)  # hover, no click (must NOT become a step)
    t0, t1 = el["toggle_0"], el["toggle_1"]

    def toggle(box):
        def fx(scr):
            x, y, w, h = box
            cv2.rectangle(scr, (x, y), (x + w, y + h), (255, 255, 255), -1)
            cv2.rectangle(scr, (x, y), (x + w, y + h), (60, 60, 230), -1, cv2.LINE_AA)
            cv2.circle(scr, (x + w - 12, y + h // 2), 8, (255, 255, 255), -1, cv2.LINE_AA)
        return fx

    rec.click(t0[0] + 22, t0[1] + 10, toggle(t0), move_s=0.8, label="toggle 0")
    # fast double action: 0.15 s move + 0.2 s dwell to the next toggle (~0.4 s apart)
    rec.click(t1[0] + 22, t1[1] + 10, toggle(t1), move_s=0.15, dwell=0.2, label="toggle 1 (fast)")
    rec.idle(0.8)
    ch = el["chart"]

    def tooltip(scr):
        cv2.rectangle(scr, (520, 300), (680, 350), (40, 40, 40), -1)
        cv2.putText(scr, "Jun: 1,284", (535, 332), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    rec.click(560, 420, tooltip, move_s=0.7, label="click busy chart -> tooltip")
    rec.idle(0.9)
    rf = el["refresh"]
    rec.click(rf[0] + 60, rf[1] + 18, _replace(screen_dashboard(seed=7)[0]), move_s=0.6, label="refresh -> new data")
    rec.idle(1.5)
    rec.close()
    vids.append({"name": "v3_hard_dashboard", "path": p, "truth": rec.truth, "kind": "synthetic, hard"})

    # 4) held-out: written after the segmenter rules were frozen, run once, never tuned on.
    #    Table screen, bigger cursor (32 px), search + typing, hover without click,
    #    page buttons, row "Open" -> navigation to the form.
    table, el = SYNTH_SCREENS["synth_table"]()
    p = os.path.join(VIDEOS, "v4_heldout_table.mp4")
    rec = Recorder(p, table, cursor_size=32, noise=3.0, seed=4)
    rec.idle(0.7)
    se = el["search"]
    rec.click(se[0] + 200, se[1] + 17, _focus(se), move_s=0.9, label="click search")
    rec.idle(1.0)
    rec.type_text(se, "acme", char_s=0.15, label="type search")
    rec.caret = None
    rec.idle(1.0)
    ex = el["export"]

    def ex_hover(fr):
        x, y, w, h = ex
        cv2.rectangle(fr, (x, y), (x + w, y + h), (200, 200, 200), 2)

    rec.hover = (ex, ex_hover)
    rec.move(ex[0] + 50, ex[1] + 17, 0.6)
    rec.idle(0.9)  # hover only
    p2 = el["page_2"]

    def page(scr):
        x, y, w, h = p2
        cv2.rectangle(scr, (x, y), (x + w, y + h), (222, 130, 61), -1)
        cv2.putText(scr, "3", (x + 13, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    rec.click(p2[0] + 18, p2[1] + 15, page, move_s=0.7, label="page 3")
    rec.idle(0.7)
    r4 = el["row_4"]
    rec.click(r4[0] + 40, r4[1] + 13, _replace(SYNTH_SCREENS["synth_form"]()[0]), move_s=0.7, label="open row -> form")
    rec.idle(1.2)
    rec.close()
    vids.append({"name": "v4_heldout_table", "path": p, "truth": rec.truth, "kind": "synthetic, held-out"})

    # 5) real recording of a demo app (recorder-drawn cursor), ground truth from its .truth.json
    if os.path.exists(REAL_VIDEO):
        with open(REAL_VIDEO.replace(".mp4", ".truth.json")) as fh:
            tj = json.load(fh)
        truth = [{"action": "click", "t": c["t"], "x": c["x"], "y": c["y"], "label": c.get("label", "")}
                 for c in tj["ground_truth_clicks"]]
        vids.append({"name": "real_approve_invoice", "path": REAL_VIDEO, "truth": truth, "kind": "real app recording"})
    return vids


def match_steps(pred, truth, tol=0.6):
    """Greedy one-to-one matching by time (|dt| <= tol s), closest first."""
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(truth):
            dt = abs(p["t_start"] - g["t"])
            if dt <= tol:
                pairs.append((dt, i, j))
    pairs.sort()
    used_p, used_g, out = set(), set(), []
    for dt, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j))
    return out


def bench_segment(screens):
    vids = make_videos(screens)
    res = []
    for v in vids:
        t0 = time.time()
        seg = cvv.segment_recording(v["path"], sample_fps=5)
        el = time.time() - t0
        steps = seg["steps"]
        m = match_steps(steps, v["truth"])
        loc_err, t_err, act_ok = [], [], 0
        details = []
        for i, j in m:
            p, g = steps[i], v["truth"][j]
            t_err.append(abs(p["t_start"] - g["t"]))
            act_ok += int(p["action"] == g["action"])
            if g["action"] == "click" and p["click"] is not None:
                loc_err.append(float(np.hypot(p["click"]["x"] - g["x"], p["click"]["y"] - g["y"])))
        unmatched_gt = [v["truth"][j] for j in range(len(v["truth"])) if j not in {b for _, b in m}]
        unmatched_pred = [{"action": steps[i]["action"], "t": steps[i]["t_start"], "click": steps[i]["click"]}
                          for i in range(len(steps)) if i not in {a for a, _ in m}]
        for s in steps:
            details.append({"action": s["action"], "t": s["t_start"], "click": s["click"],
                            "target_bbox": s["target_bbox"], "n_regions": len(s["change_regions"])})
        # visual summary of the predicted steps (failures folder if anything is off)
        n_tp = len(m)
        prec = n_tp / len(steps) if steps else 0.0
        rec_ = n_tp / len(v["truth"]) if v["truth"] else 0.0
        if unmatched_gt or unmatched_pred:
            _save_segment_failure(v, steps, unmatched_gt, unmatched_pred)
        res.append({"video": v["name"], "kind": v["kind"], "gt_steps": len(v["truth"]), "pred_steps": len(steps),
                    "matched": n_tp, "precision": prec, "recall": rec_,
                    "action_accuracy_matched": act_ok / n_tp if n_tp else 0.0,
                    "click_loc_err_px_mean": float(np.mean(loc_err)) if loc_err else None,
                    "click_loc_err_px_max": float(np.max(loc_err)) if loc_err else None,
                    "timing_err_s_mean": float(np.mean(t_err)) if t_err else None,
                    "timing_err_s_max": float(np.max(t_err)) if t_err else None,
                    "runtime_s": el, "duration_s": seg["duration_s"], "frames_analyzed": seg["frames_analyzed"],
                    "stats": seg["stats"], "unmatched_gt": unmatched_gt, "unmatched_pred": unmatched_pred,
                    "truth": v["truth"], "pred": details})
    tot_gt = sum(r["gt_steps"] for r in res)
    tot_pred = sum(r["pred_steps"] for r in res)
    tot_m = sum(r["matched"] for r in res)
    errs = []
    for r in res:
        pass
    return {"videos": res, "overall": {"gt_steps": tot_gt, "pred_steps": tot_pred, "matched": tot_m,
                                       "precision": tot_m / tot_pred if tot_pred else 0.0,
                                       "recall": tot_m / tot_gt if tot_gt else 0.0}}


def _save_segment_failure(v, steps, unmatched_gt, unmatched_pred):
    cap = cv2.VideoCapture(v["path"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    pts, labels = [], []
    for g in unmatched_gt:
        pts.append((g["x"], g["y"]))
        labels.append(f"MISSED {g['action']} t={g['t']:.2f}")
    img = cvv.annotate(frame, points=pts, labels=labels, color=GT_COLOR)
    pts, labels = [], []
    for p in unmatched_pred:
        if p["click"]:
            pts.append((p["click"]["x"], p["click"]["y"]))
            labels.append(f"EXTRA {p['action']} t={p['t']:.2f}")
    img = cvv.annotate(img, points=pts, labels=labels)
    y = 30
    for p in unmatched_pred:
        if not p["click"]:
            cv2.putText(img, f"EXTRA {p['action']} t={p['t']:.2f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cvv.BRAND_BLUE, 2)
            y += 24
    cv2.imwrite(os.path.join(FAIL, f"segment_{v['name']}.png"), img)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def f3(v):
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.3f}"


def f1(v):
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.1f}"


def pc(v):
    return f"{100 * v:.1f}%"


def write_report(R):
    L = []
    env = R["env"]
    L.append("# cofounder_vision benchmark results\n")
    L.append(f"Generated by `uv run python eval/run_eval.py` on {env['date']} "
             f"(OpenCV {env['opencv']}, numpy {env['numpy']}, Python {env['python']}, {env['machine']}). "
             f"Total runtime {env['runtime_s']:.0f} s. Seed {SEED}. All numbers below are computed by this run.\n")
    L.append("Images: 4 real screenshots of the Cofounder web UI (one page rendered at 1440x900, 1280x800, "
             "900x1100 and 2048x1400 Retina 2x, captured with headless Chrome) + 3 synthetic app screens "
             "drawn with OpenCV (form, table, dashboard). Latencies are measured with `cv2.setNumThreads(1)` "
             "in worker processes (Apple Silicon); a 1-vCPU Lambda will differ.\n")
    lb = R["locate"]
    L.append("## 1. locate()\n")
    L.append(f"{lb['n_templates']} element templates (sizes {lb['template_sizes_wh_min']} to "
             f"{lb['template_sizes_wh_max']} px w,h) picked by Canny-contour proposals with filters "
             "(size, aspect, contrast, edge density, unique in their own image), x 10 conditions = "
             f"{lb['n_runs']} runs. Hit = found and centre error <= max(8 px, 10% of the template's larger side). "
             "Wrong = found but elsewhere (the dangerous failure). Negatives use a template from the other "
             "image family (real vs synthetic), so the element is absent. Template-only numbers are the same runs "
             "stopped after the template stage (its own timer).\n")
    for path in ("template", "template+features"):
        L.append(f"\n**Path: {path}**\n")
        L.append("| condition | n | hit | miss | wrong | ambiguous | via features | p50 ms | p95 ms |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for cond in CONDITIONS:
            r = lb["rows"][f"{path}|{cond}"]
            if cond == "negative":
                L.append(f"| negative (absent) | {r['n']} | - | - | FP {pc(r['false_positive_rate'])} | "
                         f"{pc(r['ambiguous_rate'])} | {r['features_used']} | {f1(r['latency_ms_p50'])} | {f1(r['latency_ms_p95'])} |")
            else:
                L.append(f"| {cond} | {r['n']} | {pc(r['hit_rate'])} | {pc(r['miss_rate'])} | "
                         f"{pc(r['wrong_location_rate'])} | {pc(r['ambiguous_rate'])} | {r['features_used']} | "
                         f"{f1(r['latency_ms_p50'])} | {f1(r['latency_ms_p95'])} |")
        s = lb["summary"][path]
        L.append(f"\nAll positive conditions: hit {pc(s['hit_rate_all_positive_conditions'])}, wrong location "
                 f"{pc(s['wrong_location_rate'])}; negatives FP {pc(s['negative_false_positive_rate'])}.")
    L.append("\nHit rate by template source (template+features, positive conditions):\n")
    L.append("| source | templates | hit |")
    L.append("|---|---|---|")
    for k, v in lb["by_source"].items():
        L.append(f"| {k} | {v['templates']} | {pc(v['hit_rate'])} |")
    L.append(f"\nAnnotated failure cases: {lb['failures_saved']} images `eval/failures/locate_*.png` "
             "(green = ground truth, blue = prediction, template inset top-left).\n")

    vb = R["verify"]
    cm = vb["confusion"]
    L.append("## 2. verify_change()\n")
    L.append(f"{vb['n']} before/after pairs over 7 screens. Positive = the screen really changed "
             "(button state, text inserted, navigation); negative = cursor moved only, caret blink only, JPEG "
             "re-encode (q40-75). Prediction 'changed' = verdict `changed` or `unexpected_global`.\n")
    L.append("| | predicted changed | predicted no change |")
    L.append("|---|---|---|")
    L.append(f"| **actually changed** | {cm['tp']} (TP) | {cm['fn']} (FN) |")
    L.append(f"| **actually unchanged** | {cm['fp']} (FP) | {cm['tn']} (TN) |")
    L.append(f"\nAccuracy **{pc(vb['accuracy'])}**. Latency p50 {f1(vb['latency_ms_p50'])} ms, p95 {f1(vb['latency_ms_p95'])} ms.\n")
    L.append("| category | n | accuracy | verdicts |")
    L.append("|---|---|---|---|")
    for k, v in vb["per_category"].items():
        L.append(f"| {k} | {v['n']} | {pc(v['accuracy'])} | {', '.join(f'{a}: {b}' for a, b in v['verdicts'].items())} |")
    ug = vb["unexpected_global"]
    L.append(f"\n`unexpected_global` set (main area replaced by another screen, ROI in the untouched corner): "
             f"{ug['unexpected_global']}/{ug['n']} labelled `unexpected_global`; others: {ug['other'] or 'none'}.\n")

    sb = R["signature"]
    L.append("## 3. screen_signature() / compare_screens()\n")
    L.append(f"- Same screen after JPEG q50, 0.5x rescale, or cursor+typed text: {sb['same_detected']}/{sb['same_pairs']} "
             f"called `same` (min similarity {sb['same_similarity_min']:.3f}).")
    L.append(f"- Different screens: {sb['different_called_same']}/{sb['different_pairs']} wrongly called `same` "
             f"(max similarity {sb['different_similarity_max']:.3f}).")
    L.append(f"- Same Cofounder page rendered at different window sizes/aspect ratios (layout reflows): "
             f"{sb['same_page_resized_called_same']}/{sb['same_page_resized_pairs']} called `same` "
             "(reported separately: there is no single right answer).")
    if sb["misses"]:
        L.append(f"- Missed same-screen pairs: {sb['misses']}")
    L.append("")

    sg = R["segment"]
    L.append("## 4. segment_recording()\n")
    L.append("Four synthetic recordings (1280x800, 30 fps, mp4v, drawn arrow cursor on smooth eased paths, "
             "per-frame Gaussian noise sigma 2, blinking caret, idle periods) + one real recording of a demo app "
             "(1180x760, 10 fps, cursor drawn by the recorder, ground truth from its `.truth.json`). Analysis at "
             "5 fps. A predicted step matches a ground-truth step if |dt| <= 0.6 s (one-to-one, greedy).\n")
    L.append("| video | kind | GT steps | predicted | precision | recall | action acc. | click err px (mean/max) | time err s (mean/max) | runtime s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sg["videos"]:
        L.append(f"| {r['video']} | {r['kind']} | {r['gt_steps']} | {r['pred_steps']} | {pc(r['precision'])} | "
                 f"{pc(r['recall'])} | {pc(r['action_accuracy_matched'])} | {f1(r['click_loc_err_px_mean'])} / "
                 f"{f1(r['click_loc_err_px_max'])} | {f3(r['timing_err_s_mean'])} / {f3(r['timing_err_s_max'])} | "
                 f"{r['runtime_s']:.1f} |")
    o = sg["overall"]
    L.append(f"\nOverall: {o['matched']} matched of {o['gt_steps']} GT and {o['pred_steps']} predicted steps -> "
             f"precision {pc(o['precision'])}, recall {pc(o['recall'])}.\n")
    for r in sg["videos"]:
        if r["unmatched_gt"] or r["unmatched_pred"]:
            L.append(f"- **{r['video']}**: missed {[(g['action'], g['t'], g.get('label', '')) for g in r['unmatched_gt']]}; "
                     f"extra {[(p['action'], p['t']) for p in r['unmatched_pred']]} "
                     f"(see `failures/segment_{r['video']}.png`).")
    L.append("\n**Development vs held-out.** v1-v3 and the real demo-app recording were used while developing the "
             "segmenter (hover suppression, the relaxed stillness rule, chroma-aware change, click-time "
             "refinement and the moved-since-click fix were added after seeing failures on them), so their scores "
             "are optimistic. v4 was written after those rules were frozen and run once, without tuning.\n")
    L.append("\nThe hard video (v3) contains: a hover highlight with a resting cursor and no click (must not "
             "become a step), two toggle clicks ~0.4 s apart (fast double action), and a click on a busy bar chart.\n")
    L.append("## Notes on honesty / limits of this benchmark\n")
    L.append("- The cursor in the synthetic videos is drawn with the same renderer the detector's template is "
             "built from, so cursor detection is tested on the ideal shape (+ codec + noise). The real demo-app "
             "recording uses its own recorder-drawn cursor.")
    L.append("- Perturbations are synthetic (resize, blur, JPEG, occluding rectangles); the Retina screenshot is "
             "a real 2x render, but templates are always cut from the same image they are searched in "
             "(before perturbation).")
    L.append("- Thresholds were set while writing the unit tests, before this benchmark was run; they were not "
             "tuned per condition.")
    with open(os.path.join(HERE, "RESULTS.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer templates / pairs (smoke run)")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args()
    t_all = time.time()
    shutil.rmtree(FAIL, ignore_errors=True)
    os.makedirs(FAIL, exist_ok=True)
    rng = np.random.default_rng(SEED)
    screens = load_screens()
    print(f"OpenCV {cv2.__version__}; screens: {', '.join(f'{k} {v.shape[1]}x{v.shape[0]}' for k, v in screens.items())}")
    assert cv2.__version__.startswith("5."), "OpenCV 5 required"

    t = time.time()
    lb = bench_locate(screens, rng, 12 if args.quick else 48, args.workers)
    print(f"locate: {lb['n_runs']} runs in {time.time() - t:.1f}s; summary {lb['summary']}")
    t = time.time()
    vb = bench_verify(screens, np.random.default_rng(SEED + 1), 1 if args.quick else 4)
    print(f"verify: acc {vb['accuracy']:.3f} confusion {vb['confusion']} ({time.time() - t:.1f}s)")
    sb = bench_signature(screens, np.random.default_rng(SEED + 2))
    print(f"signature: same {sb['same_detected']}/{sb['same_pairs']}, diff->same {sb['different_called_same']}/{sb['different_pairs']}")
    t = time.time()
    sg = bench_segment(screens)
    print(f"segment: {sg['overall']} ({time.time() - t:.1f}s)")
    for r in sg["videos"]:
        print(f"  {r['video']}: P {r['precision']:.2f} R {r['recall']:.2f} loc {r['click_loc_err_px_mean']} "
              f"t {r['timing_err_s_mean']} missed {r['unmatched_gt']} extra {r['unmatched_pred']}")
    R = {"env": {"opencv": cv2.__version__, "numpy": np.__version__, "python": platform.python_version(),
                 "machine": f"{platform.system()} {platform.machine()}", "date": time.strftime("%Y-%m-%d %H:%M"),
                 "runtime_s": time.time() - t_all, "seed": SEED, "quick": args.quick},
         "locate": lb, "verify": vb, "signature": sb, "segment": sg}
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_report(R)
    print(f"done in {time.time() - t_all:.0f}s -> eval/RESULTS.md, eval/results.json, eval/failures/")


if __name__ == "__main__":
    main()
