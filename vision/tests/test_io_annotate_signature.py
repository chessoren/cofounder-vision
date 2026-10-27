"""Tests for decode/encode helpers, annotate, screen_signature and compare_screens."""

import cv2
import numpy as np
from _helpers import app_screen, run_all

import cofounder_vision as cv


def test_cv2_is_opencv5():
    assert cv2.__version__.startswith("5."), cv2.__version__


def test_png_roundtrip():
    img = app_screen(1)
    out = cv.decode_image(cv.encode_png(img))
    assert out.shape == img.shape and np.array_equal(out, img)


def test_decode_rejects_garbage():
    try:
        cv.decode_image(b"not an image")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_annotate_draws_brand_blue_and_copies():
    img = np.full((200, 300, 3), 255, np.uint8)
    out = cv.annotate(img, boxes=[[20, 20, 100, 50]], points=[[200, 150]], labels=["btn", "click"])
    assert np.all(img == 255), "input must not be modified"
    blue = np.all(out == np.array(cv.BRAND_BLUE, np.uint8), axis=2)
    assert blue.sum() > 100
    assert cv.BRAND_BLUE == (222, 130, 61)


def test_signature_format_and_same_screen():
    a = app_screen(2)
    s = cv.screen_signature(a)
    assert len(s["dhash"]) == 16 and int(s["dhash"], 16) >= 0
    assert abs(sum(s["hist"]) - 1.0) < 1e-3
    # same screen at another resolution and after JPEG -> same
    b = cv2.resize(a, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", b, [cv2.IMWRITE_JPEG_QUALITY, 50])
    b = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    r = cv.compare_screens(s, cv.screen_signature(b))
    assert r["same"] and r["similarity"] > 0.9, r


def test_signature_different_screen():
    a = app_screen(3)
    b = np.full_like(a, 30)
    cv2.rectangle(b, (100, 100), (800, 500), (200, 120, 40), -1)
    r = cv.compare_screens(cv.screen_signature(a), cv.screen_signature(b))
    assert not r["same"] and r["similarity"] < 0.7, r


if __name__ == "__main__":
    run_all(globals())
