"""End-to-end smoke test of the deployed Cofounder Vision endpoint (stdlib only).

  python3 cloud/smoke_test.py                 # URL from cloud/endpoint.json
  python3 cloud/smoke_test.py https://xyz.lambda-url.us-west-2.on.aws/

Exercises every route with the bundled samples, including the full upload flow
(presigned S3 PUT -> /api/segment -> object deleted), plus input-validation errors.
Exits non-zero on any failure.
"""
from __future__ import annotations

import base64
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

CLOUD = Path(__file__).resolve().parent
results: list[tuple[str, bool, float, str]] = []


def base_url() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1].rstrip("/")
    return json.loads((CLOUD / "endpoint.json").read_text())["url"].rstrip("/")


URL = base_url()


def req(method: str, path: str, body=None, raw: bytes | None = None, headers=None, full_url=None, timeout=90):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"content-type": "application/json"} if data is not None and raw is None else {}
    h.update(headers or {})
    h.setdefault("user-agent", "cofounder-vision-smoke-test/1.0")
    r = urllib.request.Request(full_url or URL + path, data=data, method=method, headers=h)
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            payload, status, hdrs = resp.read(), resp.status, dict(resp.headers)
    except urllib.error.HTTPError as e:
        payload, status, hdrs = e.read(), e.code, dict(e.headers)
    ms = (time.perf_counter() - t) * 1000
    return status, payload, {k.lower(): v for k, v in hdrs.items()}, ms


def check(name: str, cond: bool, ms: float, detail: str = "") -> None:
    results.append((name, bool(cond), ms, detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name:<44} {ms:8.0f} ms  {detail}", flush=True)


def jbody(payload: bytes):
    try:
        return json.loads(payload)
    except Exception:
        return {}


def sample_b64(name: str) -> str:
    _, payload, _, _ = req("GET", f"/samples/{name}")
    return base64.b64encode(payload).decode()


def tiny_png(w: int, h: int) -> bytes:
    raw = b"".join(b"\x00" + b"\x80" * w for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def main() -> int:
    print(f"Smoke test: {URL}\n")
    # --- health (first call after a deploy is a cold start)
    s, p, h, ms = req("GET", "/health")
    hb = jbody(p)
    check("GET /health", s == 200 and hb.get("ok"), ms,
          f"opencv={hb.get('opencv')} arch={hb.get('arch')} cold={hb.get('cold_start')} build={(hb.get('version') or {}).get('build_id')}")
    check("  opencv 5.x on aarch64", str(hb.get("opencv", "")).startswith("5.") and hb.get("arch") == "aarch64", 0)
    check("  X-Request-Id header", bool(h.get("x-request-id")), 0, h.get("x-request-id", ""))
    warm = []
    for _ in range(3):
        _, _, _, ms = req("GET", "/health")
        warm.append(ms)
    check("GET /health (warm x3)", True, sorted(warm)[1], "median")

    # --- web page
    s, p, h, ms = req("GET", "/")
    check("GET / (web demo HTML)", s == 200 and b"Cofounder Vision" in p and "text/html" in h.get("content-type", ""), ms, f"{len(p)} bytes")

    # --- samples
    s, p, _, ms = req("GET", "/api/samples")
    samples = jbody(p)
    check("GET /api/samples", s == 200 and samples.get("locate") and samples.get("verify") and samples.get("segment"), ms,
          f"{len(samples.get('locate', []))} locate, {len(samples.get('verify', []))} verify, {len(samples.get('segment', []))} recordings")

    # --- locate
    for smp in samples.get("locate", []):
        body = {"screenshot_b64": sample_b64(smp["screenshot"]), "template_b64": sample_b64(smp["template"])}
        s, p, _, ms = req("POST", "/api/locate", body)
        r = jbody(p)
        ok = s == 200 and r.get("found") is True and bool(r.get("annotated_png_b64"))
        check(f"POST /api/locate [{smp['id']}]", ok, ms,
              f"found={r.get('found')} xy=({r.get('x')},{r.get('y')}) score={r.get('score')} scale={r.get('scale')} "
              f"method={r.get('method')} ambiguous={r.get('ambiguous')} cv={r.get('elapsed_ms')}ms" if s == 200 else p[:200].decode(errors="replace"))

    # --- verify
    expectations = {"verify1": True, "verify2": False}
    for smp in samples.get("verify", []):
        body = {"before_b64": sample_b64(smp["before"]), "after_b64": sample_b64(smp["after"])}
        s, p, _, ms = req("POST", "/api/verify", body)
        r = jbody(p)
        exp = expectations.get(smp["id"])
        ok = s == 200 and bool(r.get("annotated_png_b64")) and (exp is None or r.get("changed") is exp)
        check(f"POST /api/verify [{smp['id']}]", ok, ms,
              f"changed={r.get('changed')} verdict={r.get('verdict')} ssim_local={r.get('ssim_local')} ssim_global={r.get('ssim_global')} "
              f"regions={len(r.get('regions') or [])}" if s == 200 else p[:200].decode(errors="replace"))

    # --- segment via the full presigned-upload flow
    for smp in samples.get("segment", [])[:1]:
        _, video, _, _ = req("GET", f"/samples/{smp['video']}")
        s, p, _, ms = req("POST", "/api/upload-url", {"content_type": "video/mp4", "size": len(video)})
        u = jbody(p)
        check("POST /api/upload-url", s == 200 and u.get("url", "").startswith("https://") and u.get("key", "").startswith("uploads/"), ms,
              u.get("key", p[:200].decode(errors="replace")))
        if s != 200:
            continue
        s, p, _, ms = req("PUT", "", raw=video, headers={"Content-Type": "video/mp4"}, full_url=u["url"])
        check("PUT presigned S3 upload", s == 200, ms, f"{len(video)} bytes")
        s, p, _, ms = req("POST", "/api/segment", {"key": u["key"]}, timeout=120)
        r = jbody(p)
        steps = r.get("steps") or []
        ok = s == 200 and len(steps) >= 1
        detail = (f"{len(steps)} steps, {r.get('frames_analyzed')} frames, {r.get('duration_s')} s video, {len(p) / 1e6:.2f} MB response; actions="
                  + ",".join(str(x.get("action")) for x in steps)) if s == 200 else p[:300].decode(errors="replace")
        check(f"POST /api/segment [upload {smp['id']}]", ok, ms, detail)
        s, p, _, ms = req("POST", "/api/segment", {"key": u["key"]})
        check("  upload deleted after analysis", s == 404 and jbody(p).get("error") == "upload_not_found", ms, f"status={s}")
        # Exactly the takeover agent's flow: no size, then full_targets=true.
        s, p, _, ms = req("POST", "/api/upload-url", {"content_type": "video/mp4"})
        u = jbody(p)
        check("POST /api/upload-url [agent, no size]", s == 200 and {"url", "key", "content_type"} <= set(u), ms, u.get("key", ""))
        s, p, _, ms = req("PUT", "", raw=video, headers={"Content-Type": "video/mp4"}, full_url=u["url"])
        check("PUT presigned S3 upload [agent]", s == 200, ms)
        s, p, _, ms = req("POST", "/api/segment", {"key": u["key"], "full_targets": True}, timeout=120)
        r = jbody(p)
        full_ok = True
        for st in r.get("steps") or []:
            if st.get("target_png_b64") and st.get("target_bbox"):
                png = base64.b64decode(st["target_png_b64"])
                full_ok &= int.from_bytes(png[16:20], "big") == int(st["target_bbox"][2])
        check("POST /api/segment [agent, full_targets]", s == 200 and len(r.get("steps") or []) >= 1 and full_ok, ms,
              f"{len(r.get('steps') or [])} steps, target crops full-res={full_ok}, clicks=" +
              ";".join(f"({st['click']['x']},{st['click']['y']})" for st in (r.get("steps") or []) if st.get("click")))
        s, p, _, ms = req("POST", "/api/segment", {"sample": smp["video"]}, timeout=120)
        check(f"POST /api/segment [bundled {smp['id']}]", s == 200 and len(jbody(p).get("steps") or []) >= 1, ms)

    # --- validation / error handling
    s, p, h, ms = req("POST", "/api/locate", raw=b"{not json", headers={"content-type": "application/json"})
    check("invalid JSON -> 400", s == 400 and jbody(p).get("error") == "invalid_json", ms, jbody(p).get("message", ""))
    s, p, _, ms = req("POST", "/api/locate", {"screenshot_b64": "@@@", "template_b64": "AAAA"})
    check("invalid base64 -> 400", s == 400 and jbody(p).get("error") == "invalid_base64", ms, jbody(p).get("message", ""))
    s, p, _, ms = req("POST", "/api/verify", {"before_b64": base64.b64encode(b"not an image").decode(), "after_b64": "AAAA"})
    check("undecodable image -> 400", s == 400 and jbody(p).get("error") == "invalid_image", ms, jbody(p).get("message", ""))
    big = base64.b64encode(tiny_png(4100, 2)).decode()
    s, p, _, ms = req("POST", "/api/verify", {"before_b64": big, "after_b64": big})
    check("image > 4096 px -> 413", s == 413 and jbody(p).get("error") == "image_too_large", ms, jbody(p).get("message", ""))
    s, p, _, ms = req("POST", "/api/upload-url", {"content_type": "application/x-msdownload", "size": 10})
    check("bad upload content type -> 400", s == 400, ms, jbody(p).get("message", ""))
    s, p, _, ms = req("POST", "/api/upload-url", {"content_type": "video/mp4", "size": 500 * 2**20})
    check("upload > 40 MB -> 413", s == 413, ms, jbody(p).get("message", ""))
    s, p, _, ms = req("POST", "/api/segment", {"key": "uploads/../../etc/passwd"})
    check("path-traversal key -> 400", s == 400 and jbody(p).get("error") == "invalid_key", ms)
    s, p, _, ms = req("GET", "/nope")
    check("unknown route -> 404", s == 404, ms)
    s, p, h, ms = req("OPTIONS", "/api/locate", headers={"Origin": "https://example.org", "Access-Control-Request-Method": "POST",
                                                         "Access-Control-Request-Headers": "content-type"})
    check("CORS preflight", s in (200, 204) and "access-control-allow-origin" in h, ms, f"allow-origin={h.get('access-control-allow-origin')}")

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:", ", ".join(r[0].strip() for r in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
