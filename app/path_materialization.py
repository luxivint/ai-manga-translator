from __future__ import annotations

import os


def ensure_path_materialized(path: str) -> bool:
    """Confirm an image path is a real file on disk.

    The desktop app supports opening images lazily from archives or
    project-file blob storage; this worker only ever operates on plain
    files already downloaded to local disk, so materialization here is
    just an existence check.
    """
    if not path:
        return False
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False
