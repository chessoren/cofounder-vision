"""Generate the bundled demo inputs (synthetic app screenshots + a short screen recording).

Everything is drawn with OpenCV so the samples are reproducible and contain no real user data.
Run via build.py, or directly:
  uv run --with opencv-python-headless==5.0.0.93 --with numpy python cloud/samples_gen.py cloud/samples
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 800
BG = (247, 241, 238)  # BGR of #EEF1F7
INK = (93, 49, 17)  # #11315D
ACCENT = (222, 130, 61)  # #3D82DE
MUTED = (150, 130, 120)
WHITE = (255, 255, 255)
GREEN = (120, 180, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.55, color=INK, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _rrect(img, x, y, w, h, color, r=10, border=None):
    if border is not None:
        _rrect(img, x - 1, y - 1, w + 2, h + 2, border, r=r + 1)
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def _button(img, x, y, w, h, label, primary=True, pressed=False):
    col = ACCENT if primary else WHITE
    if pressed:
        col = (170, 95, 40) if primary else (225, 220, 215)
    _rrect(img, x, y, w, h, col, r=8, border=None if primary else (220, 200, 190))
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
    _text(img, label, (x + (w - tw) // 2, y + (h + th) // 2), 0.55, WHITE if primary else INK, 1)


def app_frame(page="invoices", rows=6, form=None, toast=None, pressed=None, typed=""):
    """Draw a fake 'Ledger' business app."""
    img = np.full((H, W, 3), BG, np.uint8)
    # title bar
    cv2.rectangle(img, (0, 0), (W, 36), (235, 228, 224), -1)
    for i, c in enumerate(((95, 95, 255), (60, 190, 255), (90, 200, 40))):
        cv2.circle(img, (20 + i * 22, 18), 6, c, -1, cv2.LINE_AA)
    _text(img, "Ledger - Acme Studio", (W // 2 - 90, 24), 0.5, MUTED)
    # sidebar
    cv2.rectangle(img, (0, 36), (220, H), (252, 248, 246), -1)
    _text(img, "LEDGER", (24, 80), 0.7, INK, 2)
    for i, item in enumerate(["Dashboard", "Invoices", "Customers", "Reports", "Settings"]):
        y = 130 + i * 44
        if item.lower() == page:
            _rrect(img, 12, y - 24, 196, 36, (250, 225, 205), r=8)
        _text(img, item, (32, y), 0.58, ACCENT if item.lower() == page else INK)
    # header
    _text(img, page.capitalize(), (252, 92), 1.0, INK, 2)
    _button(img, 1060, 62, 180, 42, "+ New invoice", pressed=pressed == "new")
    _button(img, 940, 62, 104, 42, "Export", primary=False, pressed=pressed == "export")
    # table card
    _rrect(img, 252, 128, 988, 70 + rows * 52, WHITE, r=16, border=(235, 220, 210))
    for j, hdr in enumerate(["Invoice", "Customer", "Due", "Amount", "Status"]):
        _text(img, hdr, (280 + j * 190, 168), 0.5, MUTED)
    customers = ["Northwind", "Globex", "Initech", "Umbrella", "Hooli", "Stark Ind.", "Wayne Ent.", "Acme Co."]
    for r in range(rows):
        y = 214 + r * 52
        cv2.line(img, (272, y - 30), (1220, y - 30), (240, 232, 226), 1)
        vals = [f"INV-{1040 + r}", customers[r % len(customers)], f"Nov {3 + r * 2}", f"${(r + 3) * 415:,}.00"]
        for j, v in enumerate(vals):
            _text(img, v, (280 + j * 190, y), 0.55, INK)
        paid = r % 3 == 0
        _rrect(img, 1040, y - 20, 84, 28, (220, 240, 215) if paid else (215, 235, 250), r=12)
        _text(img, "Paid" if paid else "Open", (1060, y), 0.5, GREEN if paid else (40, 140, 220))
    if form:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (60, 40, 30), -1)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        _rrect(img, 390, 170, 500, 400, WHITE, r=20, border=(230, 210, 200))
        _text(img, "New invoice", (420, 220), 0.85, INK, 2)
        _text(img, "Customer", (420, 268), 0.5, MUTED)
        _rrect(img, 420, 280, 440, 44, (250, 246, 244), r=8, border=(220, 200, 190))
        _text(img, typed or "", (436, 309), 0.6, INK)
        _text(img, "Amount", (420, 358), 0.5, MUTED)
        _rrect(img, 420, 370, 440, 44, (250, 246, 244), r=8, border=(220, 200, 190))
        _text(img, "$1,250.00" if form == "filled" else "", (436, 399), 0.6, INK)
        _button(img, 700, 500, 160, 44, "Save invoice", pressed=pressed == "save")
        _button(img, 560, 500, 120, 44, "Cancel", primary=False)
    if toast:
        _rrect(img, 940, 720, 300, 52, (60, 45, 30), r=14)
        cv2.circle(img, (968, 746), 10, GREEN, -1, cv2.LINE_AA)
        _text(img, toast, (990, 752), 0.58, WHITE)
    return img


def draw_cursor(img, x, y, down=False):
    pts = np.array([[0, 0], [0, 22], [6, 17], [10, 27], [14, 25], [10, 16], [17, 16]], np.int32) + [x, y]
    if down:
        cv2.circle(img, (x, y), 16, (240, 180, 120), 2, cv2.LINE_AA)
    cv2.fillPoly(img, [pts], (20, 20, 20), cv2.LINE_AA)
    cv2.polylines(img, [pts], True, WHITE, 1, cv2.LINE_AA)
    return img


def make_recording(path: Path, fps=10):
    frames = []

    def hold(base_fn, pos, n, down=False):
        for _ in range(n):
            frames.append(draw_cursor(base_fn().copy(), *pos, down=down))

    def move(base_fn, a, b, n):
        base = base_fn()
        for i in range(1, n + 1):
            t = i / n
            t = t * t * (3 - 2 * t)
            frames.append(draw_cursor(base.copy(), int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t)))

    home = lambda: app_frame()
    # 1. idle on invoices list, then move to "+ New invoice" and click
    hold(home, (640, 600), 8)
    move(home, (640, 600), (1150, 83), 12)
    hold(lambda: app_frame(pressed="new"), (1150, 83), 3, down=True)
    # 2. dialog opens; click customer field and type
    blank = lambda: app_frame(form="empty")
    hold(blank, (1150, 83), 6)
    move(blank, (1150, 83), (640, 302), 10)
    hold(blank, (640, 302), 2, down=True)
    name = "Northwind Traders"
    for k in range(1, len(name) + 1):
        frames.append(draw_cursor(app_frame(form="empty", typed=name[:k]).copy(), 640, 302))
    filled = lambda: app_frame(form="filled", typed=name)
    hold(filled, (640, 302), 6)
    # 3. click "Save invoice"
    move(filled, (640, 302), (780, 522), 10)
    hold(lambda: app_frame(form="filled", typed=name, pressed="save"), (780, 522), 3, down=True)
    # 4. back to the list with one more row + toast
    done = lambda: app_frame(rows=7, toast="Invoice INV-1047 saved")
    hold(done, (780, 522), 14)
    move(done, (780, 522), (990, 83), 10)
    hold(lambda: app_frame(rows=7, pressed="export", toast="Invoice INV-1047 saved"), (990, 83), 3, down=True)
    hold(lambda: app_frame(rows=7, toast="Exported 7 invoices to CSV"), (990, 83), 14)

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not w.isOpened():
        raise RuntimeError("VideoWriter(mp4v) not available")
    for f in frames:
        w.write(f)
    w.release()
    return len(frames) / fps


PROJECT = Path(__file__).resolve().parent.parent
REAL_RECORDING = PROJECT / "demo-app" / "recordings" / "approve-invoice.mp4"
REAL_SCREENS = PROJECT / "vision" / "eval" / "data"


def generate(out: Path) -> dict:
    import shutil

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"locate": [], "verify": [], "segment": []}

    # Real Cofounder screens (from vision/eval/data): find the model picker captured at 1x
    # on a 2x Retina screenshot of the same app.
    if (REAL_SCREENS / "cofounder_home_1280x800.png").exists() and (REAL_SCREENS / "cofounder_home_retina_2048x1400.png").exists():
        src = cv2.imread(str(REAL_SCREENS / "cofounder_home_1280x800.png"))
        cv2.imwrite(str(out / "cofounder_template.png"), src[586:618, 309:464])
        shutil.copyfile(REAL_SCREENS / "cofounder_home_retina_2048x1400.png", out / "cofounder_retina_screenshot.png")
        manifest["locate"].append({
            "id": "cofounder-retina", "title": "Cofounder: model picker on a 2x Retina screen",
            "screenshot": "cofounder_retina_screenshot.png", "template": "cofounder_template.png",
        })

    # Locate pair 1: find the "+ New invoice" button on the invoices screen
    shot1 = app_frame()
    cv2.imwrite(str(out / "locate1_screenshot.png"), shot1)
    cv2.imwrite(str(out / "locate1_template.png"), shot1[58:108, 1056:1244])
    manifest["locate"].append({
        "id": "locate1", "title": "Find the '+ New invoice' button",
        "screenshot": "locate1_screenshot.png", "template": "locate1_template.png",
    })
    # Locate pair 2: template captured at Retina 2x, screenshot at 1x (multi-scale search) on a different layout
    shot2 = app_frame(rows=7, toast="Invoice INV-1047 saved")
    big = cv2.resize(shot2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(out / "locate2_screenshot.png"), shot2)
    cv2.imwrite(str(out / "locate2_template.png"), big[2 * 58:2 * 108, 2 * 936:2 * 1048])
    manifest["locate"].append({
        "id": "locate2", "title": "Find 'Export' (template captured at 2x Retina)",
        "screenshot": "locate2_screenshot.png", "template": "locate2_template.png",
    })
    # Verify pair: clicking "Save invoice" closes the dialog and shows a toast
    before = app_frame(form="filled", typed="Northwind Traders")
    after = app_frame(rows=7, toast="Invoice INV-1047 saved")
    cv2.imwrite(str(out / "verify1_before.png"), before)
    cv2.imwrite(str(out / "verify1_after.png"), after)
    manifest["verify"].append({
        "id": "verify1", "title": "Did 'Save invoice' work?",
        "before": "verify1_before.png", "after": "verify1_after.png",
    })
    # Verify pair 2: nothing happened (same screen, cursor moved only)
    still = app_frame()
    still2 = draw_cursor(app_frame().copy(), 700, 640)
    cv2.imwrite(str(out / "verify2_before.png"), draw_cursor(still.copy(), 640, 600))
    cv2.imwrite(str(out / "verify2_after.png"), still2)
    manifest["verify"].append({
        "id": "verify2", "title": "A click that did nothing",
        "before": "verify2_before.png", "after": "verify2_after.png",
    })
    # Real recording of the demo app (scripted demonstrator, no personal data)
    if REAL_RECORDING.exists():
        shutil.copyfile(REAL_RECORDING, out / "approve-invoice.mp4")
        manifest["segment"].append({
            "id": "approve-invoice", "title": "Approve an invoice (demo app recording, 12 s)",
            "video": "approve-invoice.mp4", "bytes": (out / "approve-invoice.mp4").stat().st_size,
        })
    # Synthetic recording
    dur = make_recording(out / "recording1.mp4")
    manifest["segment"].append({
        "id": "recording1", "title": "Create an invoice and export (synthetic, %.0f s)" % dur,
        "video": "recording1.mp4", "bytes": (out / "recording1.mp4").stat().st_size,
    })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    m = generate(Path(sys.argv[1] if len(sys.argv) > 1 else "cloud/samples"))
    print(json.dumps(m, indent=2))
