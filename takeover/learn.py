"""Learn a workflow from one screen recording (OpenCV 5 only, no model).

segment_recording() turns the video into steps: where the cursor clicked, a crop
of what it clicked, the change it caused, and a signature of the screen. On top
of that we add *ordinal grounding*: if the clicked control is one of several
identical ones (a "Review" button on every row), we record which one the person
picked (e.g. "the first, top to bottom"), so the replay picks the same one on a
list that has changed since.

    uv run python learn.py ../demo-app/recordings/approve-invoice.mp4 --out workflows/approve-invoice.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from vision_client import ROOT, Vision, locate_all, unb64

sys.path.insert(0, str(ROOT / "vision"))
import cofounder_vision as cvl  # noqa: E402


def segment(video: Path, vision: Vision) -> tuple[dict, str]:
    """Run segmentation on AWS when available (upload + /api/segment), else locally."""
    url = vision.url
    if url:
        try:
            import urllib.request

            req = urllib.request.Request(f"{url}/api/upload-url", data=json.dumps({"content_type": "video/mp4"}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                up = json.loads(r.read())
            put = urllib.request.Request(up["url"], data=video.read_bytes(), method="PUT",
                                         headers={"Content-Type": up.get("content_type", "video/mp4")})
            urllib.request.urlopen(put, timeout=120).read()
            req = urllib.request.Request(f"{url}/api/segment", data=json.dumps({"key": up["key"], "full_targets": True}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read()), "aws-lambda-arm64"
        except Exception as err:
            sys.stderr.write(f"[learn] cloud segmentation failed ({err}); running locally\n")
    return cvl.segment_recording(str(video)), "local"


def frame_at(video: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame


def learn(video: Path, out: Path) -> dict:
    vision = Vision()
    t0 = time.time()
    seg, where = segment(video, vision)
    steps = []
    for s in seg["steps"]:
        if s.get("action") != "click" or not s.get("click") or not s.get("target_png_b64"):
            continue
        click = s["click"]
        before = frame_at(video, click["t"] - 0.25)  # the screen just before the click
        target = unb64(s["target_png_b64"])
        # Ordinal grounding: which of the identical controls did the person pick?
        cands = locate_all(before, target, min_score=0.85)
        rank = None
        if len(cands) > 1:
            nearest = min(cands, key=lambda c: (c["x"] - click["x"]) ** 2 + (c["y"] - click["y"]) ** 2)
            rank = nearest["rank"]
        # What did this click do to the screen in the demonstration? The replay compares
        # its own effect with this, so "the screen changed" is not enough: it must change
        # by about as much as when the person did it.
        after = frame_at(video, click["t"] + 1.2)
        effect = cvl.verify_change(before, after, roi=[max(0, int(click["x"]) - 120), max(0, int(click["y"]) - 60), 240, 120])
        steps.append({
            "index": len(steps),
            "demo_effect": {"global_change": round(float(effect["global_change"]), 4),
                            "local_change": round(float(effect["local_change"]), 4), "verdict": effect["verdict"]},
            "t": round(click["t"], 2),
            "click": {"x": int(click["x"]), "y": int(click["y"])},
            "target_png_b64": s["target_png_b64"],
            "screen_before_png_b64": base64.b64encode(cv2.imencode(".png", before)[1].tobytes()).decode(),
            "repeated_control": {"count": len(cands), "rank": rank} if rank is not None else None,
        })
    workflow = {
        "name": "Approve a supplier invoice",
        "source_video": str(video),
        "segmented_on": where,
        "learn_seconds": round(time.time() - t0, 2),
        "stats": seg.get("stats", {}),
        "steps": steps,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(workflow))
    return workflow


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="workflows/approve-invoice.json")
    a = ap.parse_args()
    wf = learn(Path(a.video), Path(a.out))
    print(json.dumps({k: v for k, v in wf.items() if k != "steps"}, indent=1))
    for s in wf["steps"]:
        print(f"step {s['index']}: click {s['click']} at t={s['t']}s  repeated={s['repeated_control']}")
