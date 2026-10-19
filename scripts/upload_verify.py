"""Make sure a translated chapter really is in storage / on the CDN.

local_batch.py: after upload, compare the bucket listing against the local seo-upload files (name + size),
re-upload what is missing once, and refuse to delete raw pages when pages are still missing.
worker.py: after the DB page list is written, HEAD every public page URL and keep the job out of DONE when a
page is definitively broken (404, HTML instead of an image, empty file).

No package-relative imports: worker.py imports this as a top-level module, local_batch.py as scripts.upload_verify.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".avif"}
USER_AGENT = "TrendManga-LinkCheck/1.0"  # Cloudflare (1010) rejects the default Python-urllib UA
MIN_IMAGE_BYTES = 512


def _flag(name: str, default: str = "true") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off")


def upload_verify_enabled() -> bool:
    return _flag("R2_VERIFY_UPLOAD")


def publish_verify_enabled() -> bool:
    return _flag("TRANSLATION_VERIFY_PUBLISHED")


def _key(*parts: str) -> str:
    return "/".join(str(p).replace("\\", "/").strip("/") for p in parts if str(p).strip("/"))


def expected_objects(upload_dir: Path, prefix: str) -> dict[str, int]:
    upload_dir = Path(upload_dir).resolve()
    return {_key(prefix, p.relative_to(upload_dir).as_posix()): p.stat().st_size
            for p in sorted(upload_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def missing_objects(expected: dict[str, int], listed: dict[str, int]) -> list[str]:
    """Keys that are absent, empty or have a different size than the local file."""
    return [key for key, size in expected.items() if not size or listed.get(key) != size]


def stale_objects(expected: dict[str, int], listed: dict[str, int]) -> list[str]:
    """Images under the chapter folder that the new upload does not contain (old page names / extra pages)."""
    if not expected:  # never wipe a chapter because the local upload folder came out empty
        return []
    return sorted(k for k in listed if k not in expected and Path(k).suffix.lower() in IMAGE_EXTS)


def verify_and_repair(upload_dir: Path, prefix: str, list_objects: Callable[[str], dict[str, int]],
                      upload_file: Callable[[Path, str], object], retries: int = 1, log=print) -> list[str]:
    """Returns the keys still missing after `retries` re-upload rounds ([] = upload verified)."""
    upload_dir = Path(upload_dir).resolve()
    expected = expected_objects(upload_dir, prefix)
    if not expected:
        return [f"{_key(prefix)}/<no pages in {upload_dir.name}>"]
    missing = missing_objects(expected, list_objects(prefix))
    for attempt in range(1, retries + 1):
        if not missing:
            break
        log(f"R2 verify: {len(missing)} missing/short objects, re-upload attempt {attempt}: {missing[:5]}", flush=True)
        root = _key(prefix)
        for key in missing:
            try:
                upload_file(upload_dir / key[len(root) + 1:], key)
            except Exception as exc:  # the re-check below reports it as still missing
                log(f"R2 verify: re-upload failed for {key}: {exc}", flush=True)
        missing = missing_objects(expected, list_objects(prefix))
    return missing


def head_url(url: str, timeout: float = 10.0) -> tuple[int, str, int | None]:
    """(status, content-type, content-length); status 0 = network error/timeout."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return response.status, response.headers.get("Content-Type") or "", int(length) if length else None
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type") or "" if exc.headers else "", None
    except Exception:
        return 0, "", None


def classify(status: int, content_type: str, length: int | None) -> str | None:
    """None = ok, 'network' = inconclusive, anything else = definitively broken."""
    if status == 0 or status == 429 or status >= 500:
        return "network"
    if status != 200:
        return f"http_{status}"
    if not content_type.lower().startswith("image/"):
        return "not_image"
    if length is not None and length < MIN_IMAGE_BYTES:
        return "too_small"
    return None


def check_urls(urls: list[str], head=head_url, concurrency: int = 8) -> list[dict]:
    """HEAD every URL (one retry for inconclusive results); returns only the problem pages."""
    def check(url: str) -> dict:
        status, content_type, length = head(url)
        reason = classify(status, content_type, length)
        if reason == "network":
            status, content_type, length = head(url)
            reason = classify(status, content_type, length)
        return {"url": url, "status": status, "reason": reason}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(check, urls))
    return [r for r in results if r["reason"]]
