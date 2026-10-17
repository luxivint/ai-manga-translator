"""Local on-disk cache for gateway (9router / OpenAI-compatible) answers, so retries and re-runs never re-spend tokens.

The key is the SHA-256 of (cache version, call kind, URL path, exact request body). The body already holds the model, the
system + user prompt (= the prompt version), the source text and the images, so any change of text, prompt, model or
image is a different key. Headers (API keys) and the host are NOT part of the key and nothing secret is ever written:
a cache file only holds the answer JSON the gateway returned.

Only answers the CALLER accepted are stored (OCR: parsed list; translation: parsed JSON with all keys), see
`gateway.resilient_post(cache="manual")` + `gateway.cache_commit()` / `cache_discard()`. A hit carries no `usage`
block, so usage/cost logging does not count it twice.

Knobs: GATEWAY_CACHE (true), GATEWAY_CACHE_DIR (<repo>/work/cache/gateway), GATEWAY_CACHE_TTL_DAYS (30).
Delete the directory (or set GATEWAY_CACHE=false) to force fresh answers.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

CACHE_VERSION = 1
_PURGED: set[str] = set()
_LOCK = threading.Lock()


def enabled() -> bool:
    return os.environ.get("GATEWAY_CACHE", "true").strip().lower() not in ("false", "0", "no", "off")


def cache_dir() -> Path:
    raw = os.environ.get("GATEWAY_CACHE_DIR", "").strip()
    return Path(raw) if raw else Path(__file__).resolve().parents[2] / "work" / "cache" / "gateway"


def _ttl_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("GATEWAY_CACHE_TTL_DAYS", "30") or 30)) * 86400
    except ValueError:
        return 30 * 86400


def make_key(kind: str, url: str, data) -> str:
    body = bytes(data) if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")
    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}\0{kind}\0{urlparse(url).path}\0".encode("utf-8"))
    h.update(body)
    return h.hexdigest()


def _path(key: str) -> Path:
    return cache_dir() / key[:2] / f"{key}.json"


def _purge_old(root: Path) -> None:
    """Once per process and directory: drop entries older than the TTL."""
    ttl = _ttl_seconds()
    with _LOCK:
        if str(root) in _PURGED:
            return
        _PURGED.add(str(root))
    if not ttl or not root.is_dir():
        return
    cutoff = time.time() - ttl
    for f in root.glob("*/*.json"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


class CachedResponse:
    """The slice of requests.Response the callers use."""
    status_code = 200
    from_cache = True
    headers: dict = {}

    def __init__(self, body: dict):
        self.text = json.dumps(body, ensure_ascii=False)

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        return None


def get(key: str) -> CachedResponse | None:
    try:
        root = cache_dir()
        _purge_old(root)
        p = _path(key)
        if not p.is_file():
            return None
        entry = json.loads(p.read_text(encoding="utf-8"))
        if entry.get("v") != CACHE_VERSION or not isinstance(entry.get("body"), dict):
            return None
        if time.time() - float(entry.get("ts", 0)) > _ttl_seconds():
            return None
        body = dict(entry["body"])
        body.pop("usage", None)  # a hit spends no tokens: never log the original usage again
        body["_gateway_cache"] = True
        return CachedResponse(body)
    except (OSError, ValueError, TypeError):
        return None


def put(key: str, kind: str, body: dict) -> bool:
    try:
        p = _path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"v": CACHE_VERSION, "kind": kind, "ts": time.time(), "body": body},
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
        return True
    except (OSError, TypeError, ValueError):
        return False


def drop(key: str) -> None:
    try:
        _path(key).unlink()
    except OSError:
        pass
