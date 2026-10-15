"""Container startup: unique worker identity and drain the current job on stop."""
import os
import re
from pathlib import Path
import signal
import socket

ROOT = Path(__file__).resolve().parents[1]


def configure_runtime():
    if not os.environ.get("WORKER_INSTANCE_ID", "").strip():
        os.environ["WORKER_INSTANCE_ID"] = "translator-server-" + socket.gethostname()
    identity = re.sub(r"[^a-zA-Z0-9_.-]", "_", os.environ["WORKER_INSTANCE_ID"])
    path = Path(os.environ.get("WORKER_DRAIN_FILE") or str(ROOT / "work/control" / (identity + ".drain")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    os.environ["WORKER_DRAIN_FILE"] = str(path)
    return path


def request_drain(path):
    path.touch()
    print("Container stopping: finishing current job before exit", flush=True)


def main():
    path = configure_runtime()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: request_drain(path))
    import worker
    return worker.main()


if __name__ == "__main__":
    raise SystemExit(main())
