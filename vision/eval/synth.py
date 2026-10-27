"""Deterministic synthetic app screens and screen-recording synthesis for the benchmark.

Everything is drawn with OpenCV primitives (rectangles, anti-aliased text,
circles), seeded with ``numpy.random.default_rng`` so the benchmark is
reproducible. Screens return ``(image, elements)`` where ``elements`` maps a
name to an ``[x, y, w, h]`` box used to script clicks / typing in the videos.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cofounder_vision import draw_cursor  # noqa: E402

W, H = 1280, 800
FONT = cv2.FONT_HERSHEY_SIMPLEX
BLUE = (222, 130, 61)
WORDS = ("invoice account status total amount review vendor payment report export filter "
         "settings profile billing team project budget region update approve pending").split()


def _text(img, s, x, y, scale=0.5, color=(40, 40, 40), th=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, th, cv2.LINE_AA)


def _button(img, x, y, w, h, label, primary=False, rng=None):
    fill = BLUE if primary else (236, 236, 236)
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200) if not primary else (190, 110, 50), 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    _text(img, label, x + (w - tw) // 2, y + (h + th) // 2, 0.5, (255, 255, 255) if primary else (50, 50, 50))
    return [x, y, w, h]


def _chrome(img, title, rng):
    cv2.rectangle(img, (0, 0), (W, 38), (234, 234, 234), -1)
    for i, c in enumerate([(95, 95, 255), (60, 190, 255), (80, 200, 60)]):
        cv2.circle(img, (20 + 22 * i, 19), 7, c, -1, cv2.LINE_AA)
    _text(img, title, W // 2 - 60, 25, 0.55, (70, 70, 70))
    cv2.rectangle(img, (0, 38), (220, H), (240, 238, 236), -1)
    nav = {}
    for i, name in enumerate(["Inbox", "Invoices", "Vendors", "Reports", "Settings", "Team"]):
        y = 70 + 38 * i
        cv2.circle(img, (34, y - 5), 7, (150 + 10 * i, 120, 90), -1, cv2.LINE_AA)
        _text(img, name, 52, y, 0.52, (60, 60, 60))
        nav[f"nav_{name.lower()}"] = [24, y - 18, 150, 26]
    return nav


def screen_form(seed: int = 1):
    """Form screen: labelled text fields, checkboxes, Submit/Cancel."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 248, np.uint8)
    el = _chrome(img, "New vendor", rng)
    _text(img, "Vendor details", 260, 80, 0.8, (30, 30, 30), 2)
    labels = ["Company name", "Contact email", "Tax ID", "Amount (USD)", "Notes"]
    for i, lab in enumerate(labels):
        y = 120 + 78 * i
        _text(img, lab, 260, y, 0.5)
        cv2.rectangle(img, (260, y + 10), (700, y + 44), (255, 255, 255), -1)
        cv2.rectangle(img, (260, y + 10), (700, y + 44), (195, 195, 195), 1)
        el[f"field_{i}"] = [260, y + 10, 440, 34]
    for i in range(3):
        y = 540 + 30 * i
        cv2.rectangle(img, (260, y), (276, y + 16), (120, 120, 120), 1)
        if i == 1:
            cv2.line(img, (263, y + 8), (268, y + 13), BLUE, 2, cv2.LINE_AA)
            cv2.line(img, (268, y + 13), (274, y + 3), BLUE, 2, cv2.LINE_AA)
        _text(img, f"{rng.choice(WORDS).title()} {rng.choice(WORDS)} option", 286, y + 13, 0.48)
    el["submit"] = _button(img, 260, 650, 130, 38, "Submit", True)
    el["cancel"] = _button(img, 405, 650, 110, 38, "Cancel")
    # right info panel
    cv2.rectangle(img, (760, 70), (1240, 400), (255, 255, 255), -1)
    cv2.rectangle(img, (760, 70), (1240, 400), (215, 215, 215), 1)
    _text(img, "Recent activity", 780, 100, 0.6, (30, 30, 30), 1)
    for i in range(8):
        _text(img, " ".join(rng.choice(WORDS, 4)), 780, 135 + 32 * i, 0.47, (90, 90, 90))
        _text(img, f"{rng.integers(1, 28)} Oct", 1160, 135 + 32 * i, 0.45, (140, 140, 140))
    el["help"] = _button(img, 1100, 720, 130, 36, "Help center")
    return img, el


def screen_table(seed: int = 2):
    """Table screen: toolbar with search and buttons, striped table, pagination."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 250, np.uint8)
    el = _chrome(img, "Invoices", rng)
    cv2.rectangle(img, (250, 60), (600, 94), (255, 255, 255), -1)
    cv2.rectangle(img, (250, 60), (600, 94), (190, 190, 190), 1)
    cv2.circle(img, (270, 77), 7, (130, 130, 130), 1, cv2.LINE_AA)
    _text(img, "Search invoices", 285, 82, 0.48, (150, 150, 150))
    el["search"] = [250, 60, 350, 34]
    el["new"] = _button(img, 1080, 60, 160, 34, "+ New invoice", True)
    el["export"] = _button(img, 960, 60, 105, 34, "Export")
    cols = [250, 400, 640, 820, 980, 1110]
    heads = ["Number", "Vendor", "Date", "Amount", "Status", "Action"]
    cv2.rectangle(img, (240, 115), (1250, 147), (236, 236, 240), -1)
    for c, h in zip(cols, heads):
        _text(img, h, c, 137, 0.5, (60, 60, 60), 1)
    for r in range(13):
        y = 150 + 40 * r
        if r % 2:
            cv2.rectangle(img, (240, y), (1250, y + 40), (244, 244, 247), -1)
        _text(img, f"INV-{rng.integers(1000, 9999)}", cols[0], y + 26, 0.48)
        _text(img, f"{rng.choice(WORDS).title()} {rng.choice(['Ltd', 'Inc', 'GmbH', 'SA'])}", cols[1], y + 26, 0.48)
        _text(img, f"2026-{rng.integers(1, 12):02d}-{rng.integers(1, 28):02d}", cols[2], y + 26, 0.48)
        _text(img, f"${rng.integers(100, 99999):,}.{rng.integers(0, 99):02d}", cols[3], y + 26, 0.48)
        st = rng.choice(["Paid", "Due", "Draft"])
        col = {"Paid": (90, 170, 60), "Due": (60, 120, 230), "Draft": (150, 150, 150)}[st]
        cv2.rectangle(img, (cols[4], y + 10), (cols[4] + 60, y + 32), col, 1, cv2.LINE_AA)
        _text(img, st, cols[4] + 8, y + 26, 0.45, col)
        el[f"row_{r}"] = _button(img, cols[5], y + 7, 80, 26, "Open")
    for i in range(5):
        el[f"page_{i}"] = _button(img, 900 + 44 * i, 690, 36, 30, str(i + 1), i == 0)
    return img, el


def screen_dashboard(seed: int = 3):
    """Dashboard: KPI cards, bar chart (busy content), list with toggles."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 246, np.uint8)
    el = _chrome(img, "Dashboard", rng)
    for i in range(4):
        x = 250 + 250 * i
        cv2.rectangle(img, (x, 60), (x + 230, 160), (255, 255, 255), -1)
        cv2.rectangle(img, (x, 60), (x + 230, 160), (220, 220, 220), 1)
        _text(img, rng.choice(WORDS).title(), x + 16, 90, 0.5, (110, 110, 110))
        _text(img, f"{rng.integers(10, 999)}.{rng.integers(0, 9)}k", x + 16, 135, 1.0, (30, 30, 30), 2)
        el[f"card_{i}"] = [x, 60, 230, 100]
    cv2.rectangle(img, (250, 180), (850, 560), (255, 255, 255), -1)
    _text(img, "Monthly volume", 270, 210, 0.6, (30, 30, 30))
    for i in range(24):
        hgt = int(rng.integers(40, 300))
        x = 275 + 23 * i
        cv2.rectangle(img, (x, 540 - hgt), (x + 15, 540), (BLUE[0] - 4 * i, BLUE[1] + 2 * i, BLUE[2] + 3 * i), -1)
        for k in range(540 - hgt, 540, 12):
            cv2.line(img, (x, k), (x + 15, k), (255, 255, 255), 1)
    el["chart"] = [275, 240, 560, 300]
    cv2.rectangle(img, (870, 180), (1250, 560), (255, 255, 255), -1)
    for i in range(8):
        y = 215 + 42 * i
        _text(img, f"{rng.choice(WORDS).title()} alerts", 890, y + 5, 0.5)
        on = bool(rng.integers(0, 2))
        cv2.rectangle(img, (1180, y - 10), (1224, y + 10), (90, 180, 70) if on else (200, 200, 200), -1, cv2.LINE_AA)
        cv2.circle(img, (1212 if on else 1192, y), 8, (255, 255, 255), -1, cv2.LINE_AA)
        el[f"toggle_{i}"] = [1180, y - 10, 44, 20]
    el["refresh"] = _button(img, 250, 590, 120, 36, "Refresh", True)
    el["share"] = _button(img, 385, 590, 100, 36, "Share")
    el["settings"] = _button(img, 500, 590, 110, 36, "Settings")
    for i in range(4):
        _text(img, " ".join(rng.choice(WORDS, 6)), 250, 670 + 28 * i, 0.47, (100, 100, 100))
    return img, el


SYNTH_SCREENS = {"synth_form": screen_form, "synth_table": screen_table, "synth_dashboard": screen_dashboard}


# --------------------------------------------------------------------------
# Screen-recording synthesis
# --------------------------------------------------------------------------
class Recorder:
    """Writes frames with a drawn cursor, smooth moves, clicks, typing and idle time.

    Ground truth is collected in ``self.truth`` as
    ``{"action": "click"|"type", "t": seconds, "x", "y"}`` (typing: field centre).
    """

    def __init__(self, path, screen, fps=30, cursor_size=24, noise=2.0, seed=0):
        self.fps = fps
        self.screen = screen.copy()
        self.h, self.w = screen.shape[:2]
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.w, self.h))
        if not self.writer.isOpened():
            raise RuntimeError("VideoWriter(mp4v) failed")
        self.pos = np.array([self.w * 0.6, self.h * 0.6])
        self.size = cursor_size
        self.n = 0
        self.truth = []
        rng = np.random.default_rng(seed)
        self.noise = [rng.normal(0, noise, (self.h, self.w, 1)).astype(np.int16) for _ in range(6)] if noise else None
        self.hover = None  # (box, overlay_fn)
        self.caret = None  # (x, y0, y1) blinking caret

    @property
    def t(self):
        return self.n / self.fps

    def _frame(self):
        f = self.screen.copy()
        if self.caret is not None and (self.n // int(self.fps * 0.5)) % 2 == 0:
            x, y0, y1 = self.caret
            cv2.line(f, (x, y0), (x, y1), (30, 30, 30), 1)
        if self.hover is not None:
            box, fn = self.hover
            x, y, w, h = box
            if x <= self.pos[0] <= x + w and y <= self.pos[1] <= y + h:
                fn(f)
        draw_cursor(f, self.pos[0], self.pos[1], self.size)
        if self.noise is not None:
            f = np.clip(f.astype(np.int16) + self.noise[self.n % len(self.noise)], 0, 255).astype(np.uint8)
        self.writer.write(f)
        self.n += 1

    def idle(self, seconds):
        for _ in range(int(round(seconds * self.fps))):
            self._frame()

    def move(self, x, y, seconds=0.6):
        start = self.pos.copy()
        end = np.array([x, y], float)
        n = max(1, int(round(seconds * self.fps)))
        # smooth ease-in-out along a slightly curved path
        normal = np.array([-(end - start)[1], (end - start)[0]]) * 0.08
        for i in range(1, n + 1):
            a = i / n
            e = a * a * (3 - 2 * a)
            self.pos = start + (end - start) * e + normal * np.sin(np.pi * e)
            self._frame()
        self.pos = end

    def click(self, x, y, effect, move_s=0.6, dwell=0.35, label=""):
        """Move to (x, y), dwell, then apply ``effect(screen)`` (the UI reaction)."""
        self.move(x, y, move_s)
        self.idle(dwell)
        self.truth.append({"action": "click", "t": round(self.t, 3), "x": int(x), "y": int(y), "label": label})
        effect(self.screen)

    def type_text(self, box, text, char_s=0.12, label=""):
        """Type ``text`` into a field ``box`` (text grows left to right)."""
        x, y, w, h = box
        self.truth.append({"action": "type", "t": round(self.t, 3), "x": int(x + w / 2), "y": int(y + h / 2),
                           "label": label})
        for i in range(1, len(text) + 1):
            cv2.rectangle(self.screen, (x + 2, y + 2), (x + w - 2, y + h - 2), (255, 255, 255), -1)
            _text(self.screen, text[:i], x + 10, y + h // 2 + 6, 0.55, (20, 20, 20))
            (tw, _), _ = cv2.getTextSize(text[:i], FONT, 0.55, 1)
            self.caret = (x + 12 + tw, y + 7, y + h - 7)
            self.idle(char_s)

    def close(self):
        self.writer.release()
