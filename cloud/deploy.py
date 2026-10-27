"""Idempotent create-or-update deploy of Cofounder Vision to AWS Lambda (arm64 / Graviton).

  uv run --with boto3 python cloud/deploy.py            # build + deploy
  uv run --with boto3 python cloud/deploy.py --no-build # deploy the existing zip

Creates or updates exactly:
  - IAM role       cofounder-vision-lambda  (AWSLambdaBasicExecutionRole + inline S3 policy on uploads/*)
  - S3 bucket      cofounder-vision-uploads-<sha256(account)[:10]>  (Block Public Access, SSE-S3,
                   1-day lifecycle, CORS PUT from the Function URL origin)
  - Log group      /aws/lambda/cofounder-vision (14-day retention)
  - Lambda         cofounder-vision (python3.12, arm64, 3008 MB, 60 s, 2048 MB /tmp, reserved concurrency 5)
  - Function URL   AuthType NONE, CORS GET/POST, public-invoke permission
Credentials come from the default boto3 chain and are never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REGION = "us-west-2"
FUNCTION = "cofounder-vision"
ROLE = "cofounder-vision-lambda"
LOG_RETENTION_DAYS = 14
MEMORY_MB = 3008
TIMEOUT_S = 60
EPHEMERAL_MB = 2048
RESERVED_CONCURRENCY = 5
CLOUD = Path(__file__).resolve().parent
ZIP = CLOUD / "build" / "cofounder-vision.zip"
TAGS = {"project": "cofounder", "component": "vision", "managed-by": "cloud/deploy.py"}

session = boto3.Session(region_name=REGION)
iam = session.client("iam")
s3 = session.client("s3")
lam = session.client("lambda")
logs = session.client("logs")


def log(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


def account_id() -> str:
    return session.client("sts").get_caller_identity()["Account"]


def mask(s: str, acct: str) -> str:
    return s.replace(acct, "*" * 8 + acct[-4:])


# ---------------------------------------------------------------- IAM
def ensure_role(bucket: str) -> str:
    trust = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}],
    }
    created = False
    try:
        arn = iam.get_role(RoleName=ROLE)["Role"]["Arn"]
        iam.update_assume_role_policy(RoleName=ROLE, PolicyDocument=json.dumps(trust))
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(
            RoleName=ROLE,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Execution role for the cofounder-vision Lambda",
            Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()],
        )["Role"]["Arn"]
        created = True
    iam.attach_role_policy(RoleName=ROLE, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    inline = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "TempUploadsOnly",
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
                "Resource": f"arn:aws:s3:::{bucket}/uploads/*",
            }
        ],
    }
    iam.put_role_policy(RoleName=ROLE, PolicyName="uploads-prefix-only", PolicyDocument=json.dumps(inline))
    log(f"role {'created' if created else 'up to date'}: {ROLE}")
    if created:
        time.sleep(10)  # IAM propagation before Lambda can assume the role
    return arn


# ---------------------------------------------------------------- S3
def ensure_bucket(bucket: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        log(f"bucket exists: {bucket}")
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchBucket", "NotFound"):
            raise
        s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": REGION})
        s3.get_waiter("bucket_exists").wait(Bucket=bucket)
        log(f"bucket created: {bucket}")
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_ownership_controls(Bucket=bucket, OwnershipControls={"Rules": [{"ObjectOwnership": "BucketOwnerEnforced"}]})
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}, "BucketKeyEnabled": True}]},
    )
    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket,
        LifecycleConfiguration={
            "Rules": [
                {"ID": "expire-everything-after-1-day", "Status": "Enabled", "Filter": {"Prefix": ""},
                 "Expiration": {"Days": 1}, "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}},
            ]
        },
    )
    # Deny any non-TLS access.
    policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "DenyInsecureTransport", "Effect": "Deny", "Principal": "*", "Action": "s3:*",
            "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            "Condition": {"Bool": {"aws:SecureTransport": "false"}},
        }],
    }
    s3.put_bucket_policy(Bucket=bucket, Policy=json.dumps(policy))
    s3.put_bucket_tagging(Bucket=bucket, Tagging={"TagSet": [{"Key": k, "Value": v} for k, v in TAGS.items()]})


def set_bucket_cors(bucket: str, origin: str) -> None:
    s3.put_bucket_cors(
        Bucket=bucket,
        CORSConfiguration={"CORSRules": [{
            "AllowedOrigins": [origin], "AllowedMethods": ["PUT"], "AllowedHeaders": ["content-type"],
            "ExposeHeaders": ["ETag"], "MaxAgeSeconds": 3000,
        }]},
    )
    log(f"bucket CORS: PUT from {origin}")


# ---------------------------------------------------------------- Logs
def ensure_log_group() -> None:
    name = f"/aws/lambda/{FUNCTION}"
    try:
        logs.create_log_group(logGroupName=name, tags=TAGS)
        log(f"log group created: {name}")
    except logs.exceptions.ResourceAlreadyExistsException:
        pass
    logs.put_retention_policy(logGroupName=name, retentionInDays=LOG_RETENTION_DAYS)


# ---------------------------------------------------------------- Lambda
def wait_ready() -> None:
    for _ in range(120):
        c = lam.get_function_configuration(FunctionName=FUNCTION)
        if c.get("State") == "Active" and c.get("LastUpdateStatus") in ("Successful", None):
            return
        if c.get("LastUpdateStatus") == "Failed" or c.get("State") == "Failed":
            sys.exit(f"[deploy] function update failed: {c.get('LastUpdateStatusReason') or c.get('StateReason')}")
        time.sleep(2)
    sys.exit("[deploy] timed out waiting for the function")


def code_location(bucket: str) -> dict:
    size = ZIP.stat().st_size
    if size <= 50 * 2**20 - 1024 * 64:
        return {"ZipFile": ZIP.read_bytes()}
    digest = hashlib.sha256(ZIP.read_bytes()).hexdigest()[:12]
    key = f"deploy/cofounder-vision-{digest}.zip"  # expires via the bucket's 1-day lifecycle
    log(f"zip is {size / 2**20:.1f} MB (> 50 MB direct limit), uploading to s3://{bucket}/{key}")
    s3.upload_file(str(ZIP), bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"})
    return {"S3Bucket": bucket, "S3Key": key}


def ensure_function(role_arn: str, bucket: str) -> str:
    cfg = dict(
        Runtime="python3.12",
        Handler="handler.lambda_handler",
        Role=role_arn,
        MemorySize=MEMORY_MB,
        Timeout=TIMEOUT_S,
        EphemeralStorage={"Size": EPHEMERAL_MB},
        Environment={"Variables": {"BUCKET": bucket, "OPENCV_LOG_LEVEL": "ERROR", "OPENCV_IO_MAX_IMAGE_PIXELS": str(4096 * 4096), "OMP_NUM_THREADS": "2"}},
        Description="Cofounder Vision: OpenCV 5 perception API + web demo",
        LoggingConfig={"LogFormat": "Text", "LogGroup": f"/aws/lambda/{FUNCTION}"},
    )
    code = code_location(bucket)
    try:
        lam.get_function(FunctionName=FUNCTION)
        exists = True
    except lam.exceptions.ResourceNotFoundException:
        exists = False
    if not exists:
        for attempt in range(6):
            try:
                arn = lam.create_function(
                    FunctionName=FUNCTION, Code=code, Architectures=["arm64"], PackageType="Zip",
                    Tags=TAGS, Publish=False, **cfg,
                )["FunctionArn"]
                break
            except lam.exceptions.InvalidParameterValueException as e:
                if "role" in str(e).lower() and attempt < 5:
                    time.sleep(5)  # role not yet assumable
                    continue
                raise
        log("function created")
    else:
        wait_ready()
        lam.update_function_configuration(FunctionName=FUNCTION, **cfg)
        wait_ready()
        arn = lam.update_function_code(FunctionName=FUNCTION, Architectures=["arm64"], **code)["FunctionArn"]
        log("function code + configuration updated")
    wait_ready()
    if "S3Key" in code:
        try:
            s3.delete_object(Bucket=code["S3Bucket"], Key=code["S3Key"])  # Lambda has its own copy now
        except ClientError:
            pass
    try:
        lam.put_function_concurrency(FunctionName=FUNCTION, ReservedConcurrentExecutions=RESERVED_CONCURRENCY)
        log(f"reserved concurrency = {RESERVED_CONCURRENCY}")
    except ClientError as e:
        # New accounts have a 10-execution account limit and Lambda keeps >= 10 unreserved,
        # so any reservation is rejected there; the account limit itself then caps concurrency at 10.
        log(f"WARNING reserved concurrency not applied: {e.response['Error']['Message']}")
    return arn


def ensure_url() -> str:
    cors = {
        "AllowOrigins": ["*"], "AllowMethods": ["GET", "POST"], "AllowHeaders": ["content-type"],
        "ExposeHeaders": ["x-request-id"], "MaxAge": 3600, "AllowCredentials": False,
    }
    try:
        url = lam.get_function_url_config(FunctionName=FUNCTION)["FunctionUrl"]
        lam.update_function_url_config(FunctionName=FUNCTION, AuthType="NONE", Cors=cors)
    except lam.exceptions.ResourceNotFoundException:
        url = lam.create_function_url_config(FunctionName=FUNCTION, AuthType="NONE", Cors=cors, InvokeMode="BUFFERED")["FunctionUrl"]
        log("function URL created")
    perms = [
        dict(StatementId="public-function-url", Action="lambda:InvokeFunctionUrl", Principal="*", FunctionUrlAuthType="NONE"),
        dict(StatementId="public-function-url-invoke", Action="lambda:InvokeFunction", Principal="*", InvokedViaFunctionUrl=True),
    ]
    for p in perms:
        try:
            lam.add_permission(FunctionName=FUNCTION, **p)
            log(f"permission added: {p['Action']}")
        except lam.exceptions.ResourceConflictException:
            pass
        except Exception as e:  # older SDKs without InvokedViaFunctionUrl
            log(f"note: {p['StatementId']} not added ({type(e).__name__}: {e})")
    return url


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    if not args.no_build:
        subprocess.run(
            ["uv", "run", "--no-project", "--with", "opencv-python-headless==5.0.0.93", "--with", "numpy",
             "python", str(CLOUD / "build.py")],
            check=True,
        )
    acct = account_id()
    bucket = f"cofounder-vision-uploads-{hashlib.sha256(acct.encode()).hexdigest()[:10]}"
    ensure_bucket(bucket)
    role_arn = ensure_role(bucket)
    ensure_log_group()
    fn_arn = ensure_function(role_arn, bucket)
    url = ensure_url()
    set_bucket_cors(bucket, url.rstrip("/"))
    build = json.loads(subprocess.run(
        [sys.executable, "-c", f"import zipfile,sys;print(zipfile.ZipFile({str(ZIP)!r}).read('build_info.json').decode())"],
        capture_output=True, text=True, check=True).stdout)
    endpoint = {
        "url": url,
        "region": REGION,
        "account_id": acct,
        "function": FUNCTION,
        "function_arn": fn_arn,
        "role_arn": role_arn,
        "bucket": bucket,
        "log_group": f"/aws/lambda/{FUNCTION}",
        "build_id": build.get("build_id"),
        "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (CLOUD / "endpoint.json").write_text(json.dumps(endpoint, indent=2) + "\n")
    log(f"function: {mask(fn_arn, acct)}")
    log(f"role:     {mask(role_arn, acct)}")
    log(f"bucket:   {bucket}")
    log(f"build:    {endpoint['build_id']}")
    print(f"\nFunction URL: {url}\n")


if __name__ == "__main__":
    main()
