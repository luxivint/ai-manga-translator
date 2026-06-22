"""Local-disk storage backend with the same interface as r2_storage.py.

Lets local_batch.py run without any R2/cloud credentials: pages are read from
and written to a local directory tree instead of an S3-compatible bucket.
Select it via STORAGE_BACKEND=local (see scripts/storage.py).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .r2_storage import IMAGE_EXTS, normalize_key  # re-exported for callers


def _storage_root() -> Path:
    root = os.environ.get("LOCAL_STORAGE_ROOT", "").strip()
    return Path(root).resolve() if root else Path.cwd() / "storage"


def _prefix_dir(prefix: str) -> Path:
    return _storage_root() / normalize_key(prefix)


def download_prefix(prefix: str, local_dir: Path, config=None) -> list[Path]:
    source = _prefix_dir(prefix)
    local_dir.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        return []

    downloaded = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(source)
        target = local_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        downloaded.append(target)

    def _num_key(p: Path) -> tuple:
        m = re.search(r"(\d+)", p.stem)
        return (0, int(m.group(1)), p.name) if m else (1, 0, p.name)

    return sorted(downloaded, key=_num_key)


def upload_directory(local_dir: Path, prefix: str, config=None) -> dict:
    destination = _prefix_dir(prefix)
    destination.mkdir(parents=True, exist_ok=True)
    local_dir = local_dir.resolve()

    items = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(local_dir)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        items.append({"key": normalize_key(prefix, rel.as_posix()), "url": "", "bytes": path.stat().st_size})

    manifest = {
        "prefix": normalize_key(prefix),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "count": len(items),
        "items": items,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def delete_prefix(prefix: str, config=None) -> int:
    target = _prefix_dir(prefix)
    if not target.exists():
        return 0
    count = sum(1 for p in target.rglob("*") if p.is_file())
    shutil.rmtree(target)
    return count
