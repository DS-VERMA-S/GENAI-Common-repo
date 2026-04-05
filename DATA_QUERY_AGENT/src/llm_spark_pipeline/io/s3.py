from __future__ import annotations

import os
import tempfile
from urllib.parse import urlparse

import boto3


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Not an S3 URI: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def upload_file(local_path: str, s3_uri: str) -> str:
    bucket, key = _parse_s3_uri(s3_uri)
    client = boto3.client("s3")
    client.upload_file(local_path, bucket, key)
    return s3_uri


def download_file(s3_uri: str) -> str:
    bucket, key = _parse_s3_uri(s3_uri)
    client = boto3.client("s3")
    fd, local_path = tempfile.mkstemp()
    os.close(fd)
    client.download_file(bucket, key, local_path)
    return local_path

