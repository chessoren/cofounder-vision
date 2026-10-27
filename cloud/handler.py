"""Cofounder Vision - AWS Lambda handler (Function URL, payload v2).

Routes
  GET  /                 web demo (single self-contained HTML page, inlined at build time)
  GET  /health           runtime + OpenCV build info
  GET  /api/samples      bundled sample inputs (metadata)
  GET  /samples/<file>   bundled sample file (png / mp4)
  POST /api/locate       {screenshot_b64, template_b64, min_score?}
  POST /api/verify       {before_b64, after_b64, roi?}
  POST /api/upload-url   {content_type, size?} -> {url, key, content_type} presigned S3 PUT (uploads/, 10 min)
  POST /api/segment      {key | video_b64 | sample, full_targets?, sample_fps?}

Every request emits exactly one structured JSON log line (no image data, no IPs).
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import platform
import re
import sys
import time
import traceback
import uuid
from pathlib import Path

_T0 = time.perf_counter()
import cv2  # noqa: E402
import numpy as np  # noqa: E402

_IMPORT_MS = round((time.perf_counter() - _T0) * 1000, 1)

try:  # the perception package; the service still answers /health without it
    import cofounder_vision as cv_pkg  # type: ignore

    _VISION_ERR = None if hasattr(cv_pkg, "locate") else "package present but incomplete"
except Exception as exc:  # pragma: no cover - only when packaging is broken
    cv_pkg = None
    _VISION_ERR = f"{type(exc).__name__}: {exc}"

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "samples"
REGION = os.environ.get("AWS_REGION", "unknown")
BUCKET = os.environ.get("BUCKET", "")

# Replaced by build.py with the contents of cloud/web/index.html.
INDEX_HTML = "__INDEX_HTML__"
BUILD_INFO = {"build_id": "dev", "built_at": None}
try:
    BUILD_INFO = json.loads((ROOT / "build_info.json").read_text())
except Exception:
    pass

# ---- limits -------------------------------------------------------------
MAX_BODY_BYTES = 6 * 1024 * 1024 - 64 * 1024  # Function URL sync payload limit is 6 MB
MAX_IMAGE_BYTES = 4 * 1024 * 1024  # per decoded image
MAX_IMAGE_SIDE = 4096
MAX_VIDEO_BYTES = 40 * 1024 * 1024  # via presigned upload
MAX_INLINE_VIDEO_BYTES = 4 * 1024 * 1024  # via video_b64
MAX_RESPONSE_BYTES = 5_800_000
KEYFRAME_MAX_W = 960
MAX_STEPS = 40
VIDEO_TYPES = {"video/mp4": ".mp4", "video/quicktime": ".mov"}
UPLOAD_KEY_RE = re.compile(r"^uploads/[0-9a-f]{32}\.(mp4|mov)$")
SAMPLE_RE = re.compile(r"^[a-z0-9_\-]+\.(png|mp4|json)$")

_COLD = True
_s3 = None


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


# ---- helpers ------------------------------------------------------------
def _s3_client():
    global _s3
    if _s3 is None:
        import boto3
        from botocore.config import Config

        _s3 = boto3.client(
            "s3",
            region_name=REGION,
            endpoint_url=f"https://s3.{REGION}.amazonaws.com",
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )
    return _s3


def _require_vision():
    if cv_pkg is None or not hasattr(cv_pkg, "locate"):
        raise ApiError(503, "vision_unavailable", f"perception package not loaded ({_VISION_ERR})")
    return cv_pkg


def _b64decode(value, field: str, max_bytes: int) -> bytes:
    if not isinstance(value, str) or not value:
        raise ApiError(400, "invalid_field", f"'{field}' must be a non-empty base64 string")
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    if len(value) > (max_bytes * 4) // 3 + 8:
        raise ApiError(413, "too_large", f"'{field}' exceeds {max_bytes // (1024 * 1024)} MB")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ApiError(400, "invalid_base64", f"'{field}' is not valid base64")
    if len(raw) > max_bytes:
        raise ApiError(413, "too_large", f"'{field}' exceeds {max_bytes // (1024 * 1024)} MB")
    return raw


def _image(body: dict, field: str, metrics: dict):
    raw = _b64decode(body.get(field), field, MAX_IMAGE_BYTES)
    metrics.setdefault("in_bytes", {})[field] = len(raw)
    dims = _sniff_dims(raw)
    if dims and (dims[0] > MAX_IMAGE_SIDE or dims[1] > MAX_IMAGE_SIDE):
        raise ApiError(413, "image_too_large", f"'{field}' is {dims[0]}x{dims[1]}; max is {MAX_IMAGE_SIDE}x{MAX_IMAGE_SIDE}")
    vis = _require_vision()
    try:
        img = vis.decode_image(raw)
    except (ValueError, TypeError, cv2.error):
        img = None
    if img is None or getattr(img, "size", 0) == 0:
        raise ApiError(400, "invalid_image", f"'{field}' is not a decodable PNG/JPEG/WebP image")
    h, w = img.shape[:2]
    if h > MAX_IMAGE_SIDE or w > MAX_IMAGE_SIDE:
        raise ApiError(413, "image_too_large", f"'{field}' is {w}x{h}; max is {MAX_IMAGE_SIDE}x{MAX_IMAGE_SIDE}")
    metrics.setdefault("in_dims", {})[field] = [int(w), int(h)]
    return img


def _sniff_dims(raw: bytes):
    """Read width/height from PNG/JPEG headers before decoding (decompression-bomb guard)."""
    try:
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
        if raw[:2] == b"\xff\xd8":
            i = 2
            while i + 9 < len(raw):
                if raw[i] != 0xFF:
                    i += 1
                    continue
                marker = raw[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    return int.from_bytes(raw[i + 7:i + 9], "big"), int.from_bytes(raw[i + 5:i + 7], "big")
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                i += 2 + int.from_bytes(raw[i + 2:i + 4], "big")
    except Exception:
        return None
    return None


def _roi(value, name="roi"):
    if value is None:
        return None
    if isinstance(value, dict):
        value = [value.get(k) for k in ("x", "y", "w", "h")]
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or not all(isinstance(v, (int, float)) and v >= 0 for v in value)
    ):
        raise ApiError(400, "invalid_roi", f"'{name}' must be [x, y, w, h] or {{x,y,w,h}} with non-negative numbers")
    return tuple(int(v) for v in value)


def _png_b64(img) -> str:
    return base64.b64encode(_require_vision().encode_png(img)).decode()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bytes):
        return base64.b64encode(o).decode()
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def _shrink_png_b64(b64: str | None, max_w: int) -> str | None:
    if not b64:
        return b64
    try:
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w <= max_w:
            return b64
        if w > max_w:
            img = cv2.resize(img, (max_w, max(1, round(h * max_w / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        return base64.b64encode(buf.tobytes()).decode() if ok else None
    except Exception:
        return None


# ---- routes -------------------------------------------------------------
def route_health(event, ctx, metrics):
    return 200, {
        "ok": True,
        "service": "cofounder-vision",
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "arch": platform.machine(),
        "runtime": f"python{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "memory_mb": int(getattr(ctx, "memory_limit_in_mb", 0) or 0),
        "region": REGION,
        "version": BUILD_INFO,
        "vision_package": cv_pkg is not None and hasattr(cv_pkg, "locate"),
        "vision_error": _VISION_ERR,
        "cold_start": metrics.get("cold", False),
        "import_ms": _IMPORT_MS,
    }


def route_samples(event, ctx, metrics):
    manifest = SAMPLES_DIR / "manifest.json"
    data = json.loads(manifest.read_text()) if manifest.exists() else {"locate": [], "verify": [], "segment": []}
    return 200, data


def route_sample_file(name: str):
    if not SAMPLE_RE.match(name) or not (SAMPLES_DIR / name).is_file():
        raise ApiError(404, "not_found", "sample not found")
    data = (SAMPLES_DIR / name).read_bytes()
    ctype = {"png": "image/png", "mp4": "video/mp4", "json": "application/json"}[name.rsplit(".", 1)[1]]
    return data, ctype


def route_locate(body, metrics):
    vis = _require_vision()
    shot = _image(body, "screenshot_b64", metrics)
    tmpl = _image(body, "template_b64", metrics)
    if tmpl.shape[0] > shot.shape[0] or tmpl.shape[1] > shot.shape[1]:
        raise ApiError(400, "template_too_large", "template must be smaller than the screenshot")
    min_score = body.get("min_score", 0.80)
    if not isinstance(min_score, (int, float)) or not 0.0 < float(min_score) <= 1.0:
        raise ApiError(400, "invalid_min_score", "'min_score' must be a number in (0, 1]")
    kwargs = {"min_score": float(min_score)}
    if body.get("search_roi") is not None:
        kwargs["search_roi"] = _roi(body.get("search_roi"), "search_roi")
    t = time.perf_counter()
    res = _jsonable(vis.locate(shot, tmpl, **kwargs))
    metrics["cv_ms"] = {"locate": round((time.perf_counter() - t) * 1000, 1)}
    t = time.perf_counter()
    if res.get("found"):
        # locate() returns the match centre; annotate() takes top-left boxes.
        w, h = int(res["w"]), int(res["h"])
        cx, cy = int(res["x"]), int(res["y"])
        res["bbox"] = [cx - w // 2, cy - h // 2, w, h]
        ann = vis.annotate(shot, boxes=[res["bbox"]], points=[[cx, cy]], labels=[f"match {res.get('score', 0):.2f}"])
    else:
        ann = shot
    res["annotated_png_b64"] = _png_b64(ann)
    metrics["cv_ms"]["annotate"] = round((time.perf_counter() - t) * 1000, 1)
    metrics["result"] = {"found": res.get("found"), "score": res.get("score")}
    return 200, res


def route_verify(body, metrics):
    vis = _require_vision()
    before = _image(body, "before_b64", metrics)
    after = _image(body, "after_b64", metrics)
    kwargs = {}
    if body.get("roi") is not None:
        kwargs["roi"] = _roi(body.get("roi"))
    expect = body.get("expect")
    if expect is not None:
        if expect not in ("any", "local", "global"):
            raise ApiError(400, "invalid_expect", "'expect' must be one of any, local, global")
        kwargs["expect"] = expect
    t = time.perf_counter()
    res = _jsonable(vis.verify_change(before, after, **kwargs))
    metrics["cv_ms"] = {"verify": round((time.perf_counter() - t) * 1000, 1)}
    t = time.perf_counter()
    boxes = []
    for r in res.get("regions") or []:
        if isinstance(r, dict):
            boxes.append(tuple(int(r[k]) for k in ("x", "y", "w", "h")))
        elif isinstance(r, (list, tuple)) and len(r) >= 4:
            boxes.append(tuple(int(v) for v in r[:4]))
    boxes = boxes[:50]
    labels = [f"change {i + 1}" for i in range(len(boxes))]
    if kwargs.get("roi"):
        boxes.append(kwargs["roi"])
        labels.append("roi")
    ann = vis.annotate(after, boxes=boxes or None, labels=labels or None)
    res["annotated_png_b64"] = _png_b64(ann)
    metrics["cv_ms"]["annotate"] = round((time.perf_counter() - t) * 1000, 1)
    metrics["result"] = {"verdict": res.get("verdict"), "regions": len(boxes)}
    return 200, res


def route_upload_url(body, metrics):
    if not BUCKET:
        raise ApiError(503, "uploads_disabled", "upload bucket not configured")
    ctype = body.get("content_type", "video/mp4")
    if ctype not in VIDEO_TYPES:
        raise ApiError(400, "invalid_content_type", "content_type must be video/mp4 or video/quicktime")
    size = body.get("size")  # optional; when given it is signed into the URL (exact Content-Length)
    if size is not None:
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ApiError(400, "invalid_size", "'size' must be a positive integer (bytes)")
        if size > MAX_VIDEO_BYTES:
            raise ApiError(413, "too_large", f"recordings are limited to {MAX_VIDEO_BYTES // (1024 * 1024)} MB")
    key = f"uploads/{uuid.uuid4().hex}{VIDEO_TYPES[ctype]}"
    params = {"Bucket": BUCKET, "Key": key, "ContentType": ctype}
    if size is not None:
        params["ContentLength"] = size
    url = _s3_client().generate_presigned_url(
        "put_object",
        Params=params,
        ExpiresIn=600,
        HttpMethod="PUT",
    )
    metrics["upload"] = {"size": size, "content_type": ctype}
    return 200, {
        "key": key,
        "url": url,
        "content_type": ctype,
        "method": "PUT",
        "headers": {"Content-Type": ctype},
        "expires_in": 600,
        "max_bytes": MAX_VIDEO_BYTES,
    }


def _segment_file(path: str, body, metrics):
    vis = _require_vision()
    sample_fps = body.get("sample_fps", 5)
    if not isinstance(sample_fps, (int, float)) or not 1 <= sample_fps <= 10:
        raise ApiError(400, "invalid_sample_fps", "'sample_fps' must be between 1 and 10")
    full_targets = body.get("full_targets", False)
    if not isinstance(full_targets, bool):
        raise ApiError(400, "invalid_full_targets", "'full_targets' must be a boolean")
    t = time.perf_counter()
    try:
        res = vis.segment_recording(path, sample_fps=sample_fps, keyframe_max_width=KEYFRAME_MAX_W)
    except (ValueError, IOError) as exc:
        raise ApiError(422, "unreadable_video", f"could not analyse the recording: {exc}")
    metrics["cv_ms"] = {"segment": round((time.perf_counter() - t) * 1000, 1)}
    res = _jsonable(res)
    steps = res.get("steps") or []
    truncated = len(steps) > MAX_STEPS
    steps = steps[:MAX_STEPS]
    for s in steps:
        s["keyframe_png_b64"] = _shrink_png_b64(s.get("keyframe_png_b64"), KEYFRAME_MAX_W)
        if not full_targets:  # the takeover agent asks for full-resolution target crops
            s["target_png_b64"] = _shrink_png_b64(s.get("target_png_b64"), 320)
    res["steps"] = steps
    res["truncated"] = truncated
    res["full_targets"] = full_targets
    # Stay under the 6 MB Function URL response limit.
    for max_w in (720, 480, 320):
        if len(json.dumps(res)) <= MAX_RESPONSE_BYTES:
            break
        for s in steps:
            s["keyframe_png_b64"] = _shrink_png_b64(s.get("keyframe_png_b64"), max_w)
    metrics["result"] = {"steps": len(steps), "duration_s": res.get("duration_s"), "frames": res.get("frames_analyzed")}
    return 200, res


def route_segment(body, metrics):
    tmp = Path("/tmp") / f"rec-{uuid.uuid4().hex}"
    key = body.get("key")
    sample = body.get("sample")
    try:
        if sample is not None:
            if not isinstance(sample, str) or not SAMPLE_RE.match(sample) or not sample.endswith(".mp4"):
                raise ApiError(400, "invalid_sample", "unknown sample")
            src = SAMPLES_DIR / sample
            if not src.is_file():
                raise ApiError(404, "not_found", "sample not found")
            metrics["source"] = "sample"
            return _segment_file(str(src), body, metrics)
        if key is not None:
            if not BUCKET:
                raise ApiError(503, "uploads_disabled", "upload bucket not configured")
            if not isinstance(key, str) or not UPLOAD_KEY_RE.match(key):
                raise ApiError(400, "invalid_key", "'key' must be a key returned by /api/upload-url")
            s3 = _s3_client()
            try:
                head = s3.head_object(Bucket=BUCKET, Key=key)
            except Exception:
                raise ApiError(404, "upload_not_found", "upload not found (expired or never uploaded)")
            if head["ContentLength"] > MAX_VIDEO_BYTES:
                s3.delete_object(Bucket=BUCKET, Key=key)
                raise ApiError(413, "too_large", "recording too large")
            tmp = tmp.with_suffix(Path(key).suffix)
            t = time.perf_counter()
            try:
                s3.download_file(BUCKET, key, str(tmp))
            finally:
                s3.delete_object(Bucket=BUCKET, Key=key)  # delete immediately, success or not
            metrics["source"] = "upload"
            metrics["in_bytes"] = {"video": head["ContentLength"]}
            metrics["s3_ms"] = round((time.perf_counter() - t) * 1000, 1)
            return _segment_file(str(tmp), body, metrics)
        if body.get("video_b64") is not None:
            raw = _b64decode(body.get("video_b64"), "video_b64", MAX_INLINE_VIDEO_BYTES)
            tmp = tmp.with_suffix(".mp4")
            tmp.write_bytes(raw)
            metrics["source"] = "inline"
            metrics["in_bytes"] = {"video": len(raw)}
            return _segment_file(str(tmp), body, metrics)
        raise ApiError(400, "missing_input", "provide one of 'key', 'video_b64' or 'sample'")
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


POST_ROUTES = {
    "/api/locate": route_locate,
    "/api/verify": route_verify,
    "/api/upload-url": route_upload_url,
    "/api/segment": route_segment,
}


KNOWN_ROUTES = {"/", "/index.html", "/health", "/api/samples", *POST_ROUTES}


# ---- entry point --------------------------------------------------------
def _resp(status, payload, request_id, ctype="application/json", binary=False, cache=None):
    headers = {
        "Content-Type": ctype,
        "X-Request-Id": request_id,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": cache or "no-store",
    }
    if ctype.startswith("text/html"):
        headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline' "
            "https://fonts.googleapis.com; font-src https://fonts.gstatic.com; script-src 'unsafe-inline'; "
            "connect-src 'self' https://*.amazonaws.com; frame-ancestors 'none'; base-uri 'none'"
        )
        headers["X-Frame-Options"] = "DENY"
    if binary:
        return {"statusCode": status, "headers": headers, "body": base64.b64encode(payload).decode(), "isBase64Encoded": True}
    body = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return {"statusCode": status, "headers": headers, "body": body, "isBase64Encoded": False}


def lambda_handler(event, context):
    global _COLD
    t0 = time.perf_counter()
    http = (event.get("requestContext") or {}).get("http") or {}
    method = http.get("method", "GET").upper()
    path = event.get("rawPath") or http.get("path") or "/"
    request_id = getattr(context, "aws_request_id", None) or uuid.uuid4().hex
    metrics = {"cold": _COLD}
    _COLD = False
    status, out_bytes = 500, 0
    try:
        if method == "OPTIONS":
            resp = _resp(204, "", request_id)
        elif method == "GET" and path in ("/", "/index.html"):
            resp = _resp(200, INDEX_HTML, request_id, "text/html; charset=utf-8", cache="public, max-age=60")
        elif method == "GET" and path == "/health":
            resp = _resp(*route_health(event, context, metrics), request_id)
        elif method == "GET" and path == "/api/samples":
            resp = _resp(*route_samples(event, context, metrics), request_id, cache="public, max-age=300")
        elif method == "GET" and path.startswith("/samples/"):
            data, ctype = route_sample_file(path[len("/samples/"):])
            resp = _resp(200, data, request_id, ctype, binary=True, cache="public, max-age=3600")
        elif path in POST_ROUTES:
            if method != "POST":
                raise ApiError(405, "method_not_allowed", f"{path} accepts POST")
            raw = event.get("body") or ""
            if event.get("isBase64Encoded"):
                try:
                    raw = base64.b64decode(raw).decode("utf-8")
                except Exception:
                    raise ApiError(400, "invalid_body", "body must be UTF-8 JSON")
            metrics["req_bytes"] = len(raw)
            if len(raw) > MAX_BODY_BYTES:
                raise ApiError(413, "too_large", "request body exceeds 6 MB; use /api/upload-url for videos")
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError as exc:
                raise ApiError(400, "invalid_json", f"body is not valid JSON ({exc.msg} at char {exc.pos})")
            if not isinstance(body, dict):
                raise ApiError(400, "invalid_json", "body must be a JSON object")
            resp = _resp(*POST_ROUTES[path](body, metrics), request_id)
        else:
            raise ApiError(404, "not_found", f"no route for {method} {path}")
    except ApiError as e:
        resp = _resp(e.status, {"error": e.code, "message": e.message, "request_id": request_id}, request_id)
    except Exception as e:  # unexpected: log the stack (no payloads), return a generic error
        metrics["exception"] = f"{type(e).__name__}: {str(e)[:300]}"
        metrics["trace"] = traceback.format_exc(limit=6)[-1500:]
        resp = _resp(500, {"error": "internal", "message": "internal error", "request_id": request_id}, request_id)
    status = resp["statusCode"]
    out_bytes = len(resp.get("body") or "")
    if out_bytes > 6_000_000:
        metrics["oversize_response"] = out_bytes
        resp = _resp(500, {"error": "response_too_large", "message": "result exceeds 6 MB", "request_id": request_id}, request_id)
        status = 500
    log = {
        "msg": "request",
        "request_id": request_id,
        "method": method,
        "route": path if path in KNOWN_ROUTES else ("/samples/*" if path.startswith("/samples/") else "(unmatched)"),
        "status": status,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "resp_bytes": out_bytes,
        "build": BUILD_INFO.get("build_id"),
        **metrics,
    }
    print(json.dumps(_jsonable(log), separators=(",", ":")), flush=True)
    return resp
