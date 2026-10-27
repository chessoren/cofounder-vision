"""Tests for verify_change() and the OpenCV SSIM."""

import cv2
import numpy as np
from _helpers import app_screen, run_all

import cofounder_vision as cv

KEYS = {"changed", "local_change", "global_change", "ssim_local", "ssim_global", "regions", "verdict"}


def test_ssim_identity_and_drop():
    a = cv2.cvtColor(app_screen(0), cv2.COLOR_BGR2GRAY)
    assert abs(cv.ssim(a, a) - 1.0) < 1e-6
    b = a.copy()
    b[100:300, 300:700] = 0
    assert cv.ssim(a, b) < 0.9


def test_button_state_change_local():
    a = app_screen(0)
    b = a.copy()
    cv2.rectangle(b, (240, 380), (360, 416), (150, 80, 30), -1)
    r = cv.verify_change(a, b, roi=[230, 370, 140, 56])
    assert KEYS <= set(r)
    assert r["changed"] and r["verdict"] == "changed", r
    assert r["local_change"] > 0.2 and r["ssim_local"] < 0.95
    x, y, w, h = r["regions"][0]
    assert 230 <= x <= 245 and 375 <= y <= 385 and 110 <= w <= 130, r["regions"]


def test_text_inserted():
    a = app_screen(0)
    b = a.copy()
    cv2.putText(b, "hello world", (248, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    r = cv.verify_change(a, b, roi=[240, 90, 360, 32])
    assert r["changed"], r


def test_cursor_only_is_no_change():
    a = app_screen(0)
    b = a.copy()
    cv.draw_cursor(a, 500, 300, 24)
    cv.draw_cursor(b, 560, 330, 24)
    r = cv.verify_change(a, b, roi=[520, 300, 100, 80])
    assert not r["changed"] and r["verdict"] == "no_change", r
    assert r["ignored"]["cursor"] >= 1


def test_caret_blink_is_no_change():
    a = app_screen(0)
    b = a.copy()
    cv2.line(a, (250, 96), (250, 116), (20, 20, 20), 2)
    r = cv.verify_change(a, b, roi=[240, 90, 360, 32])
    assert not r["changed"], r


def test_jpeg_noise_is_no_change():
    a = app_screen(0)
    ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 50])
    b = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    r = cv.verify_change(a, b, roi=[240, 380, 120, 36])
    assert not r["changed"], r


def test_unexpected_global():
    a = app_screen(0)
    b = a.copy()
    b[36:, 200:] = app_screen(9)[36:, 200:][:, ::-1]  # main pane replaced, sidebar untouched
    r = cv.verify_change(a, b, roi=[10, 60, 150, 40])
    assert r["verdict"] == "unexpected_global" and not r["changed"], r
    r2 = cv.verify_change(a, b, roi=[10, 60, 150, 40], expect="global")
    assert r2["verdict"] == "changed"


def test_expect_local_ignores_far_change():
    a = app_screen(0)
    b = a.copy()
    cv2.rectangle(b, (700, 400), (800, 440), (0, 0, 255), -1)
    assert cv.verify_change(a, b, roi=[240, 380, 120, 36])["verdict"] == "changed"
    assert cv.verify_change(a, b, roi=[240, 380, 120, 36], expect="local")["verdict"] == "no_change"


if __name__ == "__main__":
    run_all(globals())
