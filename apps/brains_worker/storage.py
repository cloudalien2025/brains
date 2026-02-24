from __future__ import annotations

from pathlib import Path


def build_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def upload_transcript(path: Path, bucket: str, key: str) -> str:
    return build_s3_uri(bucket, key)
