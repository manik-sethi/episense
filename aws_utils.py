"""
aws_utils.py  – central place to build a boto3 session / clients

• Reads AWS keys from a `.env` file *or* environment variables
• Exposes ready‑made `s3_client`, `s3_resource`, and `bucket_name`
• Nothing is written to stdout unless credentials are missing
"""
from pathlib import Path
import os
from dotenv import dotenv_values, load_dotenv
import boto3
import s3fs

# 1) -----------------------------------------------------------------
# load .env if present (does nothing when file absent)
env = dotenv_values(".env")   

# 2) -----------------------------------------------------------------
# pull creds (ENV overrides .env → nice for prod where ENV is set)
api_key    = os.getenv("aws_access_key_id")     or env.get("aws_access_key_id")
secret_key = os.getenv("aws_secret_access_key") or env.get("aws_secret_access_key")
region     = os.getenv("aws_default_region")    or env.get("aws_default_region") or "us-west-2"


if not (api_key and secret_key):
    raise RuntimeError(
        "AWS credentials not found in environment variables or .env file.\n"
        "Add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
    )

# 3) -----------------------------------------------------------------
# build one boto3 session and reuse
_session = boto3.Session(
    aws_access_key_id     = api_key,
    aws_secret_access_key = secret_key,
    region_name           = region
)

# public singletons --------------------------------------------------
s3_client   = _session.client("s3")
s3_resource = _session.resource("s3")
fs          = s3fs.S3FileSystem(key=api_key, secret=secret_key, client_kwargs={"region_name": region})

bucket_name = "maniks-chb-mit"        # change once, import everywhere
