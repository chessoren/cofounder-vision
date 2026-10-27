"""Tests for locate(): exact, multi-scale, ROI, ambiguity, fallback, negatives."""

import cv2
import numpy as np
from _helpers import app_screen, run_all

import cofounder_vision as cv

KEYS = {"found", "x", "y", "w", "h", "score", "scale", "method", "ambiguous", "runner_up", "elapsed_ms"}


def _submit(img):
    return img[376:420, 236:364].copy()  # "Submit" button with a small margin; centre (300, 398)


def test_exact_match_and_schema():
    img = app_screen(0)
    r = cv.locate(img, _submit(img))
    assert KEYS <= set(r), set(r)
    assert r["found"] and r["method"] == "template"
    assert abs(r["x"] - 300) <= 2 and abs(r["y"] - 398) <= 2, r
    assert r["score"] > 0.95 and abs(r["scale"] - 1.0) < 0.05
    assert not r["ambiguous"]


def test_retina_and_half_scale():
    img = app_screen(0)
    t = _submit(img)
    for s in (0.5, 2.0):
        big = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        r = cv.locate(big, t)
        assert r["found"], (s, r)
        assert abs(r["x"] - 300 * s) <= max(8, 0.1 * 128 * s) and abs(r["y"] - 398 * s) <= max(8, 0.1 * 44 * s), (s, r)
        assert abs(r["scale"] - s) / s < 0.1, (s, r)


def test_search_roi_offsets_coordinates():
    img = app_screen(0)
    r = cv.locate(img, _submit(img), search_roi=[200, 300, 400, 200])
    assert r["found"] and abs(r["x"] - 300) <= 2 and abs(r["y"] - 398) <= 2, r


def test_ambiguous_duplicate():
    img = app_screen(0)
    t = _submit(img)
    img2 = img.copy()
    img2[476:520, 636:764] = t  # an identical twin button elsewhere
    r = cv.locate(img2, t)
    assert r["found"] and r["ambiguous"] and r["runner_up"] > 0.95, r


def test_absent_template_not_found():
    img = app_screen(0)
    other = np.full((44, 128, 3), 250, np.uint8)
    cv2.circle(other, (30, 22), 14, (0, 0, 200), -1)
    cv2.putText(other, "Zq#", (55, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0), 2)
    r = cv.locate(img, other)
    assert not r["found"] and r["method"] == "none", r


def test_uniform_template_rejected():
    img = app_screen(0)
    r = cv.locate(img, np.full((30, 60, 3), 246, np.uint8))
    assert not r["found"]


def test_occlusion_never_wrong():
    """A dark block over 20 % of a flat button drops NCC to ~0.64: it may be
    rejected at the default threshold, but must never be reported elsewhere,
    and the best candidate is still the right one (found with a lower threshold)."""
    img = app_screen(0)
    t = _submit(img)
    occ = img.copy()
    cv2.rectangle(occ, (236, 376), (236 + 50, 376 + 22), (20, 20, 20), -1)  # ~20 % of the template
    r = cv.locate(occ, t)
    if r["found"]:
        assert abs(r["x"] - 300) <= 8 and abs(r["y"] - 398) <= 8, r
    r2 = cv.locate(occ, t, min_score=0.55, use_features=False)
    assert r2["found"] and abs(r2["x"] - 300) <= 8 and abs(r2["y"] - 398) <= 8, r2


def test_feature_fallback_path_runs():
    img = app_screen(5)
    t = img[60:300, 220:620].copy()  # large textured region -> plenty of ORB keypoints
    occ = img.copy()
    cv2.rectangle(occ, (220, 60), (420, 200), (30, 30, 30), -1)  # heavy occlusion pushes NCC down
    r = cv.locate(occ, t, min_score=0.95)
    assert r["method"] in ("features", "none")
    if r["found"]:
        assert abs(r["x"] - 420) <= 20 and abs(r["y"] - 180) <= 20, r


if __name__ == "__main__":
    run_all(globals())
