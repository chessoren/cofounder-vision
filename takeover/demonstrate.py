"""Record a demonstration: a person approving one supplier invoice in Ledgerly.

This plays the human in "show it once". A scripted demonstrator moves the cursor
along smooth, slightly irregular paths, hesitates like a person, and clicks. The
window is captured at ~10 fps with the cursor drawn in, as a screen recorder
would, and written to an mp4. Cofounder then learns the workflow from that video
with OpenCV alone: it has no access to this script, its targets or the DOM.

    uv run python demonstrate.py --out ../demo-app/recordings/approve-invoice.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from display import VirtualDisplay, draw_cursor

# Where a person would click, in window pixels. Used ONLY to drive the
# demonstration and as ground truth for evaluation, never by the agent.
STEPS = [
    ("Review the first invoice", (1045, 220)),
    ("Match it with its purchase order", (309, 322)),
    ("Approve", (413, 383)),
]


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


class Recorder(threading.Thread):
    def __init__(self, display: VirtualDisplay, out: Path, fps: int = 10):
        super().__init__(daemon=True)
        self.display, self.out, self.fps = display, out, fps
        self.cursor = (600, 600)
        self.pressed_until = 0.0
        self.running = True
        self.frames = 0

    def run(self) -> None:
        w, h = self.display.size()
        writer = cv2.VideoWriter(str(self.out), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        period = 1.0 / self.fps
        next_t = time.time()
        while self.running:
            frame = self.display.screenshot()
            if frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h))
            x, y = self.cursor
            frame = draw_cursor(frame, x, y, pressed=time.time() < self.pressed_until)
            writer.write(frame)
            self.frames += 1
            next_t += period
            time.sleep(max(0.0, next_t - time.time()))
        writer.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../demo-app/recordings/approve-invoice.mp4")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d = VirtualDisplay()
    d.load("reset=1")
    time.sleep(1.0)

    rec = Recorder(d, out)
    rec.start()
    t0 = time.time()
    truth = []
    time.sleep(1.5)
    cur = rec.cursor
    for label, (tx, ty) in STEPS:
        # A human-ish path: eased motion with a slight curve and jitter.
        steps = random.randint(14, 20)
        bend = random.uniform(-60, 60)
        sx, sy = cur
        for i in range(1, steps + 1):
            t = ease(i / steps)
            x = sx + (tx - sx) * t + math.sin(t * math.pi) * bend
            y = sy + (ty - sy) * t + random.uniform(-1.5, 1.5)
            rec.cursor = (int(x), int(y))
            d.move(x, y)
            time.sleep(0.045)
        time.sleep(random.uniform(0.45, 0.8))  # people look before they click
        rec.pressed_until = time.time() + 0.25
        d.click(tx, ty)
        truth.append({"label": label, "t": round(time.time() - t0, 2), "x": tx, "y": ty})
        cur = (tx, ty)
        time.sleep(random.uniform(1.4, 1.9))
    time.sleep(1.5)
    rec.running = False
    rec.join()
    meta = {"video": str(out), "frames": rec.frames, "fps": rec.fps, "ground_truth_clicks": truth,
            "note": "Scripted demonstrator; ground truth is for evaluation only and is never given to the agent."}
    out.with_suffix(".truth.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
