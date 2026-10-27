"""cofounder_vision - OpenCV 5 perception for the Cofounder macOS agent.

Public API (all images are numpy BGR uint8 arrays):

* :func:`locate` - find a UI element template on a screenshot
  (multi-scale gray+edge NCC, ambiguity check, ORB+RANSAC fallback).
* :func:`verify_change` - did an action change the screen (local / global /
  none), with SSIM and changed regions.
* :func:`screen_signature` / :func:`compare_screens` - dHash + HSV histogram
  screen identity check.
* :func:`segment_recording` - turn a screen recording into click / type steps.
* :func:`annotate` - brand-blue overlays.
* :func:`decode_image` / :func:`encode_png` - bytes <-> ndarray helpers.
"""

from .annotate import BRAND_BLUE, annotate
from .cursor import detect_cursor, draw_cursor
from .imgio import decode_image, encode_png
from .locate import locate
from .segment import segment_recording
from .signature import compare_screens, screen_signature
from .verify import ssim, verify_change

__version__ = "0.1.0"

__all__ = [
    "locate", "verify_change", "screen_signature", "compare_screens", "segment_recording",
    "annotate", "decode_image", "encode_png", "draw_cursor", "detect_cursor", "ssim", "BRAND_BLUE",
]
