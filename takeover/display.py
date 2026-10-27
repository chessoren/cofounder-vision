"""Display adapters: where Cofounder looks and acts.

Both adapters expose the same four calls: screenshot() -> BGR ndarray, move(x, y),
click(x, y) and size(). Coordinates are screenshot pixels.

- VirtualDisplay drives the demo app's own window (demo-app/window.cjs) over
  localhost: reproducible, and it can never capture another app or notification.
- MacDisplay drives the real macOS screen (screencapture + pyautogui), with
  Retina scaling handled. It needs Accessibility and Screen Recording permissions.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import urllib.request

import cv2
import numpy as np


class VirtualDisplay:
    def __init__(self, base: str = "http://127.0.0.1:7788"):
        self.base = base
        self.cursor = (0, 0)

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read() or b"{}")

    def screenshot(self) -> np.ndarray:
        # A capture can fail while the page is navigating: retry briefly.
        last = None
        for _ in range(10):
            try:
                with urllib.request.urlopen(self.base + "/screenshot", timeout=10) as r:
                    data = np.frombuffer(r.read(), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except OSError as err:
                last = err
            time.sleep(0.1)
        raise RuntimeError(f"screenshot failed: {last}")

    def size(self) -> tuple[int, int]:
        with urllib.request.urlopen(self.base + "/info", timeout=5) as r:
            info = json.loads(r.read())
        return info["width"], info["height"]

    def move(self, x: float, y: float) -> None:
        self.cursor = (int(x), int(y))
        self._post("/move", {"x": x, "y": y})

    def click(self, x: float, y: float) -> None:
        self.cursor = (int(x), int(y))
        self._post("/click", {"x": x, "y": y})

    def load(self, query: str) -> None:
        self._post("/load", {"query": query})


class MacDisplay:
    """The real screen. Screenshots are in physical pixels; pyautogui works in points."""

    def __init__(self):
        import pyautogui  # imported lazily: needs a GUI session

        self.pg = pyautogui
        self.pg.FAILSAFE = True  # slam the mouse into a corner to stop the agent
        w_pts, _ = self.pg.size()
        self.scale = self.screenshot().shape[1] / w_pts
        self.cursor = (0, 0)

    def screenshot(self) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            subprocess.run(["screencapture", "-x", "-C", f.name], check=True)
            return cv2.imread(f.name, cv2.IMREAD_COLOR)

    def size(self) -> tuple[int, int]:
        img = self.screenshot()
        return img.shape[1], img.shape[0]

    def move(self, x: float, y: float) -> None:
        self.cursor = (int(x), int(y))
        self.pg.moveTo(x / self.scale, y / self.scale, duration=0.25)

    def click(self, x: float, y: float) -> None:
        self.cursor = (int(x), int(y))
        self.pg.click(x / self.scale, y / self.scale)


def draw_cursor(frame: np.ndarray, x: int, y: int, pressed: bool = False) -> np.ndarray:
    """Draw a macOS-style arrow cursor (black body, white outline) at (x, y), like a screen recorder does."""
    pts = np.array([[0, 0], [0, 22], [5, 17], [9, 26], [13, 24], [9, 16], [16, 16]], np.int32) + [x, y]
    out = frame.copy()
    if pressed:
        cv2.circle(out, (x, y), 18, (222, 130, 61), 2, cv2.LINE_AA)
    cv2.fillPoly(out, [pts], (0, 0, 0), cv2.LINE_AA)
    cv2.polylines(out, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
    return out
