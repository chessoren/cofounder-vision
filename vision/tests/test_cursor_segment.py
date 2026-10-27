"""Tests for the synthetic cursor detector and segment_recording()."""

import base64
import os
import tempfile

import cv2
import numpy as np
from _helpers import app_screen, run_all

import cofounder_vision as cv


def test_cursor_detect_multiple_sizes():
    img = app_screen(0)
    for size, (x, y) in [(16, (700, 200)), (24, (300, 100)), (40, (820, 500))]:
        f = img.copy()
        cv.draw_cursor(f, x, y, size)
        d = cv.detect_cursor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        assert d is not None and abs(d["x"] - x) <= 2 and abs(d["y"] - y) <= 2, (size, d)


def test_cursor_absent():
    d = cv.detect_cursor(cv2.cvtColor(app_screen(0), cv2.COLOR_BGR2GRAY))
    assert d is None, d


def _make_video(path):
    """640x400 @ 20 fps: move -> click button (state change) -> move -> click field -> type."""
    base = cv2.resize(app_screen(0), (640, 400), interpolation=cv2.INTER_AREA)
    fps = 20
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 400))
    assert w.isOpened()
    screen = base.copy()
    pos = np.array([500.0, 300.0])
    truth = []

    def emit(n, p):
        for _ in range(n):
            f = screen.copy()
            cv.draw_cursor(f, p[0], p[1], 20)
            w.write(f)

    def move(to, n):
        nonlocal pos
        to = np.array(to, float)
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)
            emit(1, pos + (to - pos) * a)
        pos = to

    t = 0
    emit(10, pos); t += 10
    move((190, 252), 16); t += 16
    emit(8, pos); t += 8
    truth.append(("click", t / fps, (190, 252)))
    cv2.rectangle(screen, (150, 237), (225, 260), (40, 160, 40), -1)  # button turns green
    emit(14, pos); t += 14
    move((250, 72), 16); t += 16
    emit(8, pos); t += 8
    truth.append(("click", t / fps, (250, 72)))
    cv2.rectangle(screen, (150, 56), (375, 76), (222, 130, 61), 2)  # focus ring
    emit(20, pos); t += 20
    truth.append(("type", t / fps, None))
    for i, ch in enumerate("typing!"):
        cv2.putText(screen, ch, (156 + 9 * i, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
        emit(4, pos)
    emit(20, pos)
    w.release()
    return truth


def test_segment_recording_synthetic():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mp4")
        truth = _make_video(p)
        r = cv.segment_recording(p, sample_fps=10)
    for k in ("fps", "duration_s", "frames_analyzed", "steps", "stats"):
        assert k in r
    acts = [s["action"] for s in r["steps"]]
    clicks = [s for s in r["steps"] if s["action"] == "click"]
    assert len(clicks) == 2, acts
    for s, (_, t, (x, y)) in zip(clicks, [g for g in truth if g[0] == "click"]):
        assert abs(s["click"]["x"] - x) <= 4 and abs(s["click"]["y"] - y) <= 4, s["click"]
        assert abs(s["click"]["t"] - t) <= 0.35, (s["click"], t)
        assert s["target_png_b64"] and s["target_bbox"] and s["keyframe_png_b64"]
        crop = cv.decode_image(base64.b64decode(s["target_png_b64"]))
        assert crop.shape[0] > 10 and crop.shape[1] > 10
        assert "dhash" in s["signature"]
    assert "type" in acts, acts
    for s in r["steps"]:
        assert {"index", "t_start", "t_end", "click", "target_bbox", "change_regions", "action"} <= set(s)


if __name__ == "__main__":
    run_all(globals())
