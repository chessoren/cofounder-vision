"""Perception for the takeover agent: OpenCV 5, in the cloud first.

Every call goes to the Cofounder Vision service on AWS Lambda (Graviton) when an
endpoint is configured (COFOUNDER_VISION_URL or cloud/endpoint.json), and falls
back to the same library in-process (vision/cofounder_vision) if the network or
the service fails. Each result records where it ran and how long it took, so the
trace shows exactly which perception drove each decision.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))

try:  # the shared OpenCV 5 library (same code that runs on Lambda)
    import cofounder_vision as cv_local
except Exception:  # pragma: no cover - library still being built
    cv_local = None


def endpoint() -> str:
    url = os.environ.get("COFOUNDER_VISION_URL", "")
    if url:
        return url.rstrip("/")
    try:
        return json.loads((ROOT / "cloud" / "endpoint.json").read_text())["url"].rstrip("/")
    except Exception:
        return ""


def b64png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def unb64(data: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)


class Vision:
    def __init__(self, prefer_cloud: bool = True):
        self.url = endpoint() if prefer_cloud else ""
        self.cloud_failures = 0

    def _cloud(self, route: str, payload: dict) -> dict | None:
        if not self.url or self.cloud_failures >= 3:
            return None
        req = urllib.request.Request(f"{self.url}{route}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read())
            out.pop("annotated_png_b64", None)
            out["where"] = "aws-lambda-arm64"
            out["roundtrip_ms"] = round((time.time() - t) * 1000, 1)
            return out
        except Exception as err:  # degrade to local, loudly
            self.cloud_failures += 1
            sys.stderr.write(f"[vision] cloud {route} failed ({err}); using local OpenCV\n")
            return None

    def locate(self, screen: np.ndarray, template: np.ndarray, min_score: float = 0.8) -> dict:
        out = self._cloud("/api/locate", {"screenshot_b64": b64png(screen), "template_b64": b64png(template), "min_score": min_score})
        if out is None:
            t = time.time()
            out = dict(cv_local.locate(screen, template, min_score=min_score))
            out["where"] = "local"
            out["roundtrip_ms"] = round((time.time() - t) * 1000, 1)
        return out

    def verify(self, before: np.ndarray, after: np.ndarray, roi=None) -> dict:
        payload = {"before_b64": b64png(before), "after_b64": b64png(after)}
        if roi is not None:
            payload["roi"] = [int(v) for v in roi]
        out = self._cloud("/api/verify", payload)
        if out is None:
            t = time.time()
            out = dict(cv_local.verify_change(before, after, roi=roi))
            out["where"] = "local"
            out["roundtrip_ms"] = round((time.time() - t) * 1000, 1)
        return out

    def same_screen(self, a: np.ndarray, b: np.ndarray) -> dict:
        # Cheap and local: a perceptual check of "am I where I expect to be?".
        return dict(cv_local.compare_screens(cv_local.screen_signature(a), cv_local.screen_signature(b)))


def locate_all(screen: np.ndarray, template: np.ndarray, min_score: float = 0.85, max_hits: int = 12) -> list[dict]:
    """All placements of a template (OpenCV 5 matchTemplate + non-maximum suppression).

    Repeated controls (a "Review" button on every row) are the normal case in
    business apps. locate() answers "where is the best one, and is it ambiguous";
    this returns every candidate, ordered top-to-bottom, so the agent can ground
    "the first row" the same way the person did in the demonstration.
    """
    g = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    best = (-1.0, 1.0, None)
    for s in (1.0, 0.9, 1.1, 0.8, 1.25, 0.67, 1.5, 0.5, 2.0):
        tw, th = int(t.shape[1] * s), int(t.shape[0] * s)
        if tw < 8 or th < 8 or tw > g.shape[1] or th > g.shape[0]:
            continue
        res = cv2.matchTemplate(g, cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
        if res.max() > best[0]:
            best = (float(res.max()), s, res)
    score, scale, res = best
    if res is None or score < min_score:
        return []
    tw, th = int(t.shape[1] * scale), int(t.shape[0] * scale)
    hits = []
    res = res.copy()
    for _ in range(max_hits):
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx < min_score:
            break
        x, y = loc
        hits.append({"x": x + tw // 2, "y": y + th // 2, "w": tw, "h": th, "score": round(float(mx), 3), "scale": scale})
        cv2.rectangle(res, (max(0, x - tw // 2), max(0, y - th // 2)), (x + tw // 2, y + th // 2), -1.0, -1)
    hits.sort(key=lambda h: (h["y"], h["x"]))
    for i, h in enumerate(hits):
        h["rank"] = i
    return hits
