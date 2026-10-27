"""Build the Lambda deployment zip for Cofounder Vision (no Docker needed).

  uv run --no-project --with opencv-python-headless==5.0.0.93 --with numpy python cloud/build.py

Steps
  1. read exact pins (opencv-python-headless, numpy) from vision/pyproject.toml
  2. `uv pip install --python-platform aarch64-manylinux_2_28 --python-version 3.12 --only-binary=:all:`
     into cloud/build/python (Linux arm64 wheels, cross-installed from any host)
  3. verify every installed native wheel is tagged manylinux *_aarch64
  4. copy cofounder_vision, handler.py (with web/index.html inlined), samples/, build_info.json
  5. strip tests, __pycache__, type stubs; zip to cloud/build/cofounder-vision.zip
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

CLOUD = Path(__file__).resolve().parent
ROOT = CLOUD.parent
VISION = ROOT / "vision"
BUILD = CLOUD / "build"
PKG = BUILD / "python"
ZIP = BUILD / "cofounder-vision.zip"
PLATFORM = "aarch64-manylinux_2_28"
PYVER = "3.12"


def pins() -> list[str]:
    text = (VISION / "pyproject.toml").read_text()
    deps = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.S).group(1)
    reqs = re.findall(r'"([^"]+)"', deps)
    for r in reqs:
        if "==" not in r:
            sys.exit(f"build: dependency {r!r} in vision/pyproject.toml is not pinned with ==")
    return reqs


def install(reqs: list[str]) -> None:
    if PKG.exists():
        shutil.rmtree(PKG)
    cmd = [
        "uv", "pip", "install", "--quiet",
        "--python-platform", PLATFORM, "--python-version", PYVER,
        "--only-binary=:all:", "--no-cache", "--target", str(PKG), *reqs,
    ]
    print("build: $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def verify_wheels() -> list[str]:
    tags = []
    for wheel in PKG.glob("*.dist-info/WHEEL"):
        wtags = [l.split(":", 1)[1].strip() for l in wheel.read_text().splitlines() if l.startswith("Tag:")]
        native = [t for t in wtags if not t.endswith("-any")]
        if native and not any(t.endswith("aarch64") and "manylinux" in t for t in native):
            sys.exit(f"build: {wheel.parent.name} is not a manylinux aarch64 wheel: {wtags}")
        tags.append(f"{wheel.parent.name}: {', '.join(wtags)}")
    so = next(PKG.glob("cv2/cv2*.so"))
    head = so.read_bytes()[:20]
    # ELF magic + e_machine == 0xB7 (EM_AARCH64)
    if head[:4] != b"\x7fELF" or int.from_bytes(head[18:20], "little") != 0xB7:
        sys.exit(f"build: {so.name} is not an aarch64 ELF binary")
    return tags


def strip() -> None:
    for pattern in ("**/__pycache__", "**/tests", "**/testing/tests", "*.dist-info/RECORD"):
        for p in PKG.glob(pattern):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    for p in list(PKG.rglob("*.pyi")) + list(PKG.rglob("*.pyc")):
        p.unlink()
    shutil.rmtree(PKG / "bin", ignore_errors=True)


def copy_app() -> dict:
    src_pkg = VISION / "cofounder_vision"
    if (src_pkg / "__init__.py").exists():
        shutil.copytree(src_pkg, PKG / "cofounder_vision", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"))
    # samples (regenerate if missing)
    samples = CLOUD / "samples"
    if not (samples / "manifest.json").exists():
        sys.path.insert(0, str(CLOUD))
        import samples_gen

        samples_gen.generate(samples)
    shutil.copytree(samples, PKG / "samples")
    # handler with the web page inlined
    html = (CLOUD / "web" / "index.html").read_text() if (CLOUD / "web" / "index.html").exists() else "<h1>Cofounder Vision</h1>"
    handler = (CLOUD / "handler.py").read_text()
    assert 'INDEX_HTML = "__INDEX_HTML__"' in handler
    handler = handler.replace('INDEX_HTML = "__INDEX_HTML__"', "INDEX_HTML = " + json.dumps(html))
    (PKG / "handler.py").write_text(handler)
    # build id: content hash of our sources + UTC date (no git needed)
    h = hashlib.sha256()
    for f in sorted([CLOUD / "handler.py", CLOUD / "web" / "index.html", *src_pkg.rglob("*.py"), *samples.glob("*")]):
        if f.is_file():
            h.update(f.name.encode() + f.read_bytes())
    now = dt.datetime.now(dt.timezone.utc)
    info = {
        "build_id": f"{now:%Y%m%d.%H%M}-{h.hexdigest()[:8]}",
        "built_at": now.isoformat(timespec="seconds"),
        "platform": PLATFORM,
        "python": PYVER,
        "requirements": pins(),
    }
    (PKG / "build_info.json").write_text(json.dumps(info))
    return info


def make_zip() -> int:
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(PKG.rglob("*")):
            if p.is_file():
                zi = zipfile.ZipInfo.from_file(p, p.relative_to(PKG).as_posix())
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = (0o755 if p.suffix == ".so" or ".so." in p.name else 0o644) << 16
                with open(p, "rb") as fh:
                    z.writestr(zi, fh.read(), compresslevel=9)
    return ZIP.stat().st_size


def main() -> dict:
    reqs = pins()
    install(reqs)
    tags = verify_wheels()
    strip()
    info = copy_app()
    size = make_zip()
    unzipped = sum(p.stat().st_size for p in PKG.rglob("*") if p.is_file())
    info.update({"zip": str(ZIP), "zip_mb": round(size / 2**20, 1), "unzipped_mb": round(unzipped / 2**20, 1), "wheels": tags})
    if unzipped > 250 * 2**20:
        sys.exit("build: unzipped size exceeds the 250 MB Lambda limit")
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    main()
