# Cofounder Vision on AWS Lambda (Graviton / arm64)

This is the public web endpoint for **Cofounder Vision**, the OpenCV 5 perception layer of Cofounder. One Lambda function serves two things:

- a web demo at `/`, with three tabs: *Learn from a recording*, *Find on screen* and *Did it work?*;
- a JSON API that the Cofounder takeover agent calls (`takeover/learn.py`).

The live URL is in `cloud/endpoint.json`, under `url`.

## Architecture

```
 browser / agent ──HTTPS──▶ Lambda Function URL (AuthType NONE, CORS GET/POST)
                              │
                              ▼
               Lambda "cofounder-vision"  python3.12 · arm64 (Graviton) · 3008 MB · 60 s · 2 GB /tmp
               handler.py ──▶ cofounder_vision (opencv-python-headless 5.0.0.93, numpy 2.2.6)
                 │   │
                 │   └── one structured JSON log line per request ──▶ CloudWatch Logs (14-day retention)
                 │
                 └── presign PUT / GET / DELETE on uploads/* only
                              ▼
               S3 "cofounder-vision-uploads-<hash>"  private · Block Public Access · SSE-S3 ·
                                                     TLS-only policy · 1-day expiry lifecycle
```

The video upload flow works like this:

1. `POST /api/upload-url` returns a presigned S3 PUT URL. It is valid for 10 minutes, covers one random `uploads/<uuid>.mp4` key, and has its content type pinned.
2. The client PUTs the video directly to S3, so the video never passes through the 6 MB Function URL payload limit.
3. `POST /api/segment {key}` checks the object's size (40 MB max). It then downloads the video to `/tmp`, runs `segment_recording`, and **deletes both the S3 object and the tmp file** in a `finally` block. A second call with the same key returns `404 upload_not_found`.

## API

| Route | Body | Returns |
|---|---|---|
| `GET /` | – | the web demo (one self-contained HTML page, inlined into the handler at build time) |
| `GET /health` | – | `{ok, opencv, arch, runtime, memory_mb, region, version{build_id,built_at,requirements}, vision_package, cold_start}` |
| `GET /api/samples` | – | bundled sample inputs (`locate`, `verify`, `segment`); the files are at `GET /samples/<file>` |
| `POST /api/locate` | `{screenshot_b64, template_b64, min_score?, search_roi?}` | `locate()` output (`x,y` = match **centre**), plus `bbox` `[x,y,w,h]` (top-left) and `annotated_png_b64` |
| `POST /api/verify` | `{before_b64, after_b64, roi?, expect?}` | `verify_change()` output, plus `annotated_png_b64` (change regions drawn on *after*) |
| `POST /api/upload-url` | `{content_type: "video/mp4" \| "video/quicktime", size?}` | `{url, key, content_type, method:"PUT", headers, expires_in, max_bytes}`. If you send `size`, it is signed in as the exact Content-Length. |
| `POST /api/segment` | `{key}` \| `{video_b64}` (≤ 4 MB) \| `{sample}`, plus optional `full_targets`, `sample_fps` | `segment_recording()` output. Keyframes are downscaled to ≤ 960 px wide. Target crops are downscaled to 320 px unless `full_targets: true`. |

Errors are JSON: `{error, message, request_id}`, with the status codes 400, 404, 405, 413, 422, 500 and 503. Every response carries an `X-Request-Id` header, which equals the Lambda request id and appears in the log line for that request.

Input validation:
- Images must be base64. They are limited to 4 MB per image and 4096×4096 px. The dimensions are read from the PNG or JPEG header *before* decoding, and `OPENCV_IO_MAX_IMAGE_PIXELS` is set as a second guard against decompression bombs.
- Request bodies are limited to 6 MB.
- Upload keys must match `^uploads/[0-9a-f]{32}\.(mp4|mov)$`, which rules out path traversal.
- `content_type` must be `video/mp4` or `video/quicktime`.

Example:

```bash
URL=$(python3 -c "import json;print(json.load(open('cloud/endpoint.json'))['url'].rstrip('/'))")
curl -s $URL/health
curl -s -X POST $URL/api/locate -H 'content-type: application/json' \
  -d "{\"screenshot_b64\":\"$(base64 -i screen.png)\",\"template_b64\":\"$(base64 -i button.png)\"}"
```

## Build and deploy

You need `uv` and AWS credentials in the default boto3 chain. You do **not** need Docker or the AWS CLI. The region is `us-west-2`.

```bash
uv run --with boto3 python cloud/deploy.py              # build + idempotent create-or-update
uv run --with boto3 python cloud/deploy.py --no-build   # redeploy the existing zip
python3 cloud/smoke_test.py                             # end-to-end test (stdlib only)
```

`cloud/build.py` does the following:

1. Reads the exact pins from `vision/pyproject.toml`.
2. Runs `uv pip install --python-platform aarch64-manylinux_2_28 --python-version 3.12 --only-binary=:all:`, which cross-installs the Linux arm64 wheels from any host.
3. Checks that every native wheel is tagged `manylinux_*_aarch64` and that `cv2*.so` is an AArch64 ELF binary.
4. Strips tests, `__pycache__` and `.pyi` files.
5. Copies `cofounder_vision`, the handler (with `web/index.html` inlined) and `samples/`.
6. Writes `build_info.json`, which holds a build id made from the UTC timestamp and a content hash. No git is needed.

The zip is about 52 MB, which is over the 50 MB direct-upload limit, so `deploy.py` uploads it to the bucket under `deploy/` and deletes it once Lambda has its own copy. The unzipped size is about 130 MB, well under the 250 MB limit. `/health` reporting `opencv: 5.0.0` and `arch: aarch64` proves that the wheel imports on Graviton.

`cloud/deploy.py` is idempotent. Running it again only reconciles configuration. It creates or updates:

- **IAM role** `cofounder-vision-lambda`, with `AWSLambdaBasicExecutionRole` and the inline policy `uploads-prefix-only`. That policy allows `s3:PutObject`, `s3:GetObject` and `s3:DeleteObject` on `arn:aws:s3:::<bucket>/uploads/*` and nothing else.
- **S3 bucket** `cofounder-vision-uploads-<sha256(account)[:10]>`, with:
  - Block Public Access (all four settings) and BucketOwnerEnforced;
  - SSE-S3 with a bucket key;
  - a lifecycle rule that expires all objects after 1 day and aborts incomplete multipart uploads;
  - a bucket policy that denies non-TLS access;
  - CORS that allows only `PUT` from the Function URL origin.
- **Log group** `/aws/lambda/cofounder-vision`, with 14-day retention.
- **Lambda** `cofounder-vision`: python3.12, `arm64`, 3008 MB, 60 s timeout, 2048 MB ephemeral storage. The environment sets `BUCKET`, `OPENCV_IO_MAX_IMAGE_PIXELS` and `OMP_NUM_THREADS`.
- **Function URL**: `AuthType NONE`, BUFFERED, CORS (`*`, GET/POST, `content-type`, exposes `x-request-id`). It has two resource-policy statements: `lambda:InvokeFunctionUrl` with `FunctionUrlAuthType=NONE`, and `lambda:InvokeFunction` with `InvokedViaFunctionUrl=true`.
- **Reserved concurrency 5.** The script requests it, but new accounts with a 10-execution limit reject any reservation. Lambda must keep at least 10 executions unreserved. When that happens, the script logs a warning and the account limit (10) is the effective cap. Raising the account quota lets the reservation apply on the next deploy.

`deploy.py` then writes `cloud/endpoint.json` with the URL and ARNs. This is the only file that contains the account id.

## Security and responsible use

- **No accounts and no stored data.** Images are processed in memory and never written to disk or S3. Videos live in S3 only between the PUT and the analysis, and are then deleted immediately. Anything left over, such as abandoned uploads, expires within 24 h through the lifecycle rule. No data is used for training.
- **Least privilege.** The function can touch only `uploads/*` in its own bucket. The bucket is private, encrypted, TLS-only and never public. Uploads use a presigned URL that is valid for 10 minutes and covers one random key.
- **Abuse and cost guards:**
  - strict input-size limits (images, request body, 40 MB videos);
  - a 60 s timeout;
  - a concurrency cap (see above);
  - at most 40 steps per recording, with responses kept under the 6 MB limit.
- **Hardened responses.** The HTML page is served with a strict CSP (no third-party scripts; only Google Fonts), `X-Frame-Options: DENY`, `nosniff` and `no-referrer`. API errors never echo stack traces; those go only to the logs.
- **Privacy-preserving logs.** No image or video bytes, no IP addresses, no user agents, and no presigned URLs are logged.
- **Synthetic sample data.** The bundled samples come from the scripted demo app and from synthetic screens drawn with OpenCV (`samples_gen.py`). They contain no personal data.
- The page asks users not to upload recordings that contain other people's personal data.

## Observability

Every request writes exactly one JSON line to `/aws/lambda/cofounder-vision`. It looks like this:

```json
{"msg":"request","request_id":"db3f7100-…","method":"POST","route":"/api/locate","status":200,
 "latency_ms":551.2,"resp_bytes":1322366,"build":"20261026.2135-c519fe43","cold":false,
 "req_bytes":1090727,"in_bytes":{"screenshot_b64":814722,"template_b64":3280},
 "in_dims":{"screenshot_b64":[2048,1400],"template_b64":[155,32]},
 "cv_ms":{"locate":450.2,"annotate":58.4},"result":{"found":true,"score":0.9565}}
```

Unexpected errors add `exception` and a truncated `trace` field. Lambda's own `REPORT` lines add the init duration and the memory used.

Here are some CloudWatch Logs Insights queries (Console → CloudWatch → Logs Insights → `/aws/lambda/cofounder-vision`):

```
# latency per route
filter msg = "request"
| stats count() as n, pct(latency_ms, 50) as p50, pct(latency_ms, 95) as p95 by route, status

# OpenCV time inside locate / verify / segment
filter msg = "request" and ispresent(cv_ms.locate) | stats avg(cv_ms.locate), max(cv_ms.locate)

# cold starts
filter @type = "REPORT" and ispresent(@initDuration) | stats count(), avg(@initDuration), max(@maxMemoryUsed/1048576)

# errors with their request id (match a client's X-Request-Id)
filter msg = "request" and status >= 500 | fields @timestamp, request_id, route, exception
```

## Measured performance (us-west-2; client in Europe)

These are server-side numbers from the request log lines:

| Route | Server latency |
|---|---|
| `/health` | < 1 ms |
| `/api/locate` (1280×800) | ~110–150 ms |
| `/api/locate` (2048×1400 Retina, 2× scale) | ~550 ms |
| `/api/verify` (1280×800) | ~75 ms |
| `/api/segment` (12.5 s, 1180×760 recording, 63 frames analysed) | ~4 s |

Other measurements:
- **Cold start:** Lambda init takes 530–650 ms (importing numpy and cv2 takes about 540 ms of that). The first `/health` call from Europe took about 1.8 s; warm calls take about 0.5–0.7 s, almost all of which is network round trip.
- **Peak memory:** about 275 MB. The 3008 MB setting is there for CPU: Lambda allocates CPU in proportion to memory.

## Cost

- **Lambda (arm64):** about $0.0000133 per GB-s, roughly 20% cheaper than x86. A locate call costs about 0.15 s × 2.94 GB ≈ $0.000006. A 12 s recording costs about 4 s × 2.94 GB ≈ $0.00016. Requests cost $0.20 per million.
- **S3:** objects live for seconds, and the lifecycle rule caps storage at 1 day, so the cost is effectively $0.
- **CloudWatch Logs:** about 1 KB per request and 14-day retention.
- **Totals:** idle costs $0. A judging night with a few thousand requests costs well under $1.

## Files

| File | Purpose |
|---|---|
| `handler.py` | Lambda handler (routing, validation, logging) |
| `web/index.html` | web demo, inlined at build time |
| `build.py` | builds the cross-platform arm64 zip |
| `samples_gen.py` | generates / collects the bundled samples |
| `samples/` | sample inputs and `manifest.json` |
| `deploy.py` | idempotent infrastructure and code deploy |
| `smoke_test.py` | end-to-end test of every route, including the S3 upload flow and the validation errors |
| `endpoint.json` | deploy output (URL, ARNs); the only file with the account id |
