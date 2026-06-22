from __future__ import annotations

import json
import mimetypes
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".avif"}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class R2Config:
    bucket: str
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    public_base_url: str = ""

    @classmethod
    def from_env(cls) -> "R2Config":
        endpoint_url = os.environ.get("R2_ENDPOINT_URL", "").strip()
        account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
        if not endpoint_url and account_id:
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

        values = {
            "R2_BUCKET": os.environ.get("R2_BUCKET", "").strip(),
            "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", "").strip(),
            "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", "").strip(),
            "R2_ENDPOINT_URL/R2_ACCOUNT_ID": endpoint_url,
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise RuntimeError("Missing R2 env values: " + ", ".join(missing))

        return cls(
            bucket=values["R2_BUCKET"],
            endpoint_url=endpoint_url,
            access_key_id=values["R2_ACCESS_KEY_ID"],
            secret_access_key=values["R2_SECRET_ACCESS_KEY"],
            public_base_url=os.environ.get("R2_PUBLIC_BASE_URL", "").strip().rstrip("/"),
        )


def client(config: R2Config | None = None):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for R2. Run: pip install boto3") from exc

    config = config or R2Config.from_env()
    return boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        region_name="auto",
    )


def normalize_key(*parts: str) -> str:
    return "/".join(str(part).replace("\\", "/").strip("/") for part in parts if str(part).strip("/"))


def public_url(config: R2Config, key: str) -> str:
    return f"{config.public_base_url}/{key}" if config.public_base_url else ""


def content_type_for(path: Path) -> str:
    if path.suffix.lower() == ".webp":
        return "image/webp"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def iter_files(root: Path, images_only: bool = False) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if images_only and path.suffix.lower() not in IMAGE_EXTS:
            continue
        files.append(path)
    return files


def upload_file(path: Path, key: str, config: R2Config | None = None) -> dict[str, str | int]:
    config = config or R2Config.from_env()
    s3 = client(config)
    key = normalize_key(key)
    s3.upload_file(
        str(path),
        config.bucket,
        key,
        ExtraArgs={"ContentType": content_type_for(path)},
    )
    return {
        "key": key,
        "url": public_url(config, key),
        "bytes": path.stat().st_size,
    }


def put_json(key: str, payload: dict, config: R2Config | None = None) -> dict[str, str | int]:
    config = config or R2Config.from_env()
    s3 = client(config)
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    key = normalize_key(key)
    s3.put_object(Bucket=config.bucket, Key=key, Body=body, ContentType="application/json; charset=utf-8")
    return {"key": key, "url": public_url(config, key), "bytes": len(body)}


def upload_directory(local_dir: Path, prefix: str, config: R2Config | None = None) -> dict:
    config = config or R2Config.from_env()
    local_dir = local_dir.resolve()
    items = []
    for path in iter_files(local_dir, images_only=True):
        rel = path.relative_to(local_dir).as_posix()
        items.append(upload_file(path, normalize_key(prefix, rel), config))

    manifest = {
        "prefix": normalize_key(prefix),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "count": len(items),
        "items": items,
    }
    manifest["manifest"] = put_json(normalize_key(prefix, "manifest.json"), manifest, config)
    return manifest


def download_prefix(prefix: str, local_dir: Path, config: R2Config | None = None) -> list[Path]:
    config = config or R2Config.from_env()
    s3 = client(config)
    prefix = normalize_key(prefix)
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=config.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/") or key.endswith("manifest.json"):
                continue
            if Path(key).suffix.lower() not in IMAGE_EXTS:
                continue
            rel = key[len(prefix):].lstrip("/")
            target = local_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(config.bucket, key, str(target))
            downloaded.append(target)

    def _num_key(p: Path) -> tuple:
        m = re.search(r"(\d+)", p.stem)
        return (0, int(m.group(1)), p.name) if m else (1, 0, p.name)

    return sorted(downloaded, key=_num_key)


def delete_prefix(prefix: str, config: R2Config | None = None) -> int:
    config = config or R2Config.from_env()
    s3 = client(config)
    prefix = normalize_key(prefix)
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")
    batch = []

    for page in paginator.paginate(Bucket=config.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            batch.append({"Key": item["Key"]})
            if len(batch) == 1000:
                s3.delete_objects(Bucket=config.bucket, Delete={"Objects": batch, "Quiet": True})
                deleted += len(batch)
                batch = []

    if batch:
        s3.delete_objects(Bucket=config.bucket, Delete={"Objects": batch, "Quiet": True})
        deleted += len(batch)
    return deleted
