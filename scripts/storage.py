"""Storage backend dispatcher.

Selects the R2 (Cloudflare/S3-compatible) backend or a plain local-disk
backend based on STORAGE_BACKEND (r2|local, default: r2). Both backends
expose the same download_prefix/upload_directory/delete_prefix interface,
so local_batch.py never needs to know which one is active.
"""

from __future__ import annotations

import os

from .r2_storage import normalize_key

_BACKEND = os.environ.get("STORAGE_BACKEND", "r2").strip().lower()

if _BACKEND == "local":
    from .local_storage import delete_prefix, download_prefix, upload_directory
else:
    from .r2_storage import delete_prefix, download_prefix, upload_directory

__all__ = ["delete_prefix", "download_prefix", "upload_directory", "normalize_key"]
