"""Shared synthetic fixtures for the tests (pure OpenCV drawing, deterministic)."""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def app_screen(seed: int = 0, w: int = 1000, h: int = 640) -> np.ndarray:
    """A deterministic app-like screen: title bar, sidebar, form fields, buttons, text."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 246, np.uint8)
    cv2.rectangle(img, (0, 0), (w, 36), (232, 232, 232), -1)
    for i, c in enumerate([(95, 95, 255), (60, 190, 255), (80, 200, 60)]):
        cv2.circle(img, (20 + 22 * i, 18), 7, c, -1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 36), (200, h), (238, 236, 234), -1)
    for i in range(6):
        cv2.putText(img, f"Section {i + 1}", (24, 80 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    for i in range(4):
        y = 80 + 70 * i
        cv2.putText(img, f"Field {chr(65 + i)} {rng.integers(100, 999)}", (240, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.rectangle(img, (240, y + 10), (600, y + 42), (255, 255, 255), -1)
        cv2.rectangle(img, (240, y + 10), (600, y + 42), (190, 190, 190), 1)
    cv2.rectangle(img, (240, 380), (360, 416), (222, 130, 61), -1)
    cv2.putText(img, "Submit", (262, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (380, 380), (500, 416), (225, 225, 225), -1)
    cv2.putText(img, "Cancel", (404, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)
    for i in range(8):
        txt = "".join(chr(int(c)) for c in rng.integers(97, 123, 18))
        cv2.putText(img, txt, (650, 90 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def run_all(namespace: dict) -> None:
    """Run every ``test_*`` function in ``namespace`` (plain-python runner)."""
    fails = 0
    tests = [(k, v) for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
