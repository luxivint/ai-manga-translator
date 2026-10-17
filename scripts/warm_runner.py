"""Warm job runner: one long-lived child of scripts/worker.py that runs chapter jobs IN-PROCESS, one at a time.

Why: every chapter used to be a fresh `python scripts/local_batch.py` process, so each job paid ~25 s to load the
detector, the YOLO segmenter, the LaMa ORT session (~23 s) and the fonts again. Here local_batch is imported once and
its module-level model singletons survive from job to job. Opt-in with WORKER_INPROCESS=true (worker.py).

The worker keeps the supervision (stall timeout, cancel, kill tree, log file); this child only executes. A job
failure, a cancel or a crash costs at most this child, never the worker: the worker respawns it (cold) for the next job.

Protocol: JSON lines. worker -> child on stdin, child -> worker on a private duplicate of the original stdout
(fd 1 itself is pointed at the job log during a job, and at NUL between jobs, so stray prints cannot corrupt it):
  {"op":"run","script":..,"args":[..],"env":{..},"cwd":..,"log":..}  ->  {"event":"done","code":N,...}
  {"op":"unload"} -> {"event":"unloaded",...}        {"op":"quit"}
  child first says {"event":"ready"}; with --idle-seconds it unloads the models after that long without a job
  ({"event":"unloaded","reason":"idle"}) and exits after the same time again ({"event":"exiting"}).
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

_MODULES: dict[str, object] = {}
_BUSY = threading.Lock()  # held while a job runs or models are being unloaded
_LAST_ACTIVITY = time.monotonic()
_CONTROL = None
_CONTROL_LOCK = threading.Lock()


def send(event: dict) -> None:
    with _CONTROL_LOCK:
        _CONTROL.write(json.dumps(event) + "\n")
        _CONTROL.flush()


def load_script(script: Path):
    """Import a job script once (as a module, not __main__) so its module-level state, e.g. loaded models, is kept."""
    key = str(script.resolve())
    if key not in _MODULES:
        for path in (str(script.parent), str(ROOT)):
            if path not in sys.path:
                sys.path.insert(0, path)
        spec = importlib.util.spec_from_file_location(f"warm_{script.stem}", key)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


def _std_handles(target_fd: int | None):
    """Point the Win32 STD handles (native libraries write to them, not to the CRT fds) at target_fd."""
    if os.name != "nt" or target_fd is None:
        return
    import ctypes
    import msvcrt
    handle = msvcrt.get_osfhandle(target_fd)
    kernel32 = ctypes.windll.kernel32
    kernel32.SetStdHandle(-11, handle)
    kernel32.SetStdHandle(-12, handle)


class Redirect:
    """fd 1 and fd 2 (and the Win32 STD handles) -> the job log for the duration of a job, back to NUL/stderr log after."""

    def __init__(self, log: str, idle_out: int, idle_err: int):
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0))
        self.idle_out, self.idle_err = idle_out, idle_err

    def __enter__(self):
        _flush_std()
        os.dup2(self.fd, 1)
        os.dup2(self.fd, 2)
        _std_handles(self.fd)
        return self

    def __exit__(self, *exc):
        _flush_std()
        os.dup2(self.idle_out, 1)
        os.dup2(self.idle_err, 2)
        _std_handles(self.idle_err)
        os.close(self.fd)
        return False


def _flush_std():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def memory_report() -> dict:
    report: dict = {}
    try:
        import psutil
        report["rss_mb"] = round(psutil.Process().memory_info().rss / 2**20)
    except Exception:
        pass
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            report["vram_used_mb"] = round((total - free) / 2**20)  # whole device, not just this process
    except Exception:
        pass
    return report


def models_loaded() -> list[str]:
    return [name for module in _MODULES.values() if hasattr(module, "loaded_models") for name in module.loaded_models()]


def unload_models() -> list[str]:
    released: list[str] = []
    for module in _MODULES.values():
        if hasattr(module, "unload_models"):
            released += list(module.unload_models() or [])
    gc.collect()
    return released


def run_job(req: dict, idle_out: int, idle_err: int) -> int:
    script = Path(req["script"])
    os.environ.clear()
    os.environ.update({str(k): str(v) for k, v in req["env"].items()})
    os.chdir(req["cwd"])
    sys.argv = [str(script), *req["args"]]
    with Redirect(req["log"], idle_out, idle_err):
        try:
            module = load_script(script)
            if hasattr(module, "reset_job_state"):
                module.reset_job_state()
            code = module.main()
        except SystemExit as exc:
            if exc.code is None:
                code = 0
            elif isinstance(exc.code, int):
                code = exc.code
            else:
                print(exc.code, file=sys.stderr)
                code = 1
        except BaseException:
            traceback.print_exc()
            code = 1
        _flush_std()
    return int(code or 0)


def watch_idle(idle_seconds: float) -> None:
    """Two idle stages, so an idle worker holds neither the GPU nor the RAM of the warm models:
    after idle_seconds without a job the models are unloaded (LaMa, YOLO, detector, torch cache); after another
    idle_seconds this child exits, which also drops the CUDA contexts and the imported libraries. The worker simply
    starts a fresh child for the next job."""
    handled = _LAST_ACTIVITY  # the activity stamp we already unloaded for (nothing is loaded at start)
    while True:
        time.sleep(min(5.0, max(0.2, idle_seconds / 4)))
        stamp = _LAST_ACTIVITY
        idle_for = time.monotonic() - stamp
        if idle_for < idle_seconds or (stamp == handled and idle_for < 2 * idle_seconds):
            continue
        if _BUSY.acquire(blocking=False):
            try:
                if _LAST_ACTIVITY != stamp:  # a job finished while we were deciding
                    continue
                if stamp != handled:
                    released = unload_models()
                    handled = stamp
                    send({"event": "unloaded", "reason": "idle", "released": released, **memory_report()})
                else:
                    send({"event": "exiting", "reason": "idle"})
                    _flush_std()
                    os._exit(0)  # holding _BUSY: no job is running; a request already in the pipe is re-sent by the worker
            finally:
                _BUSY.release()


def main() -> int:
    global _CONTROL, _LAST_ACTIVITY
    parser = argparse.ArgumentParser()
    parser.add_argument("--idle-seconds", type=float, default=0.0)
    opts = parser.parse_args()
    _CONTROL = os.fdopen(os.dup(1), "w", encoding="utf-8")
    null = os.open(os.devnull, os.O_WRONLY)
    idle_err = os.dup(2)
    os.dup2(null, 1)  # stray prints between jobs go nowhere; the control channel is the duplicate above
    _std_handles(null)
    idle_out = os.dup(1)
    if opts.idle_seconds > 0:
        threading.Thread(target=watch_idle, args=(opts.idle_seconds,), daemon=True, name="idle-unload").start()
    send({"event": "ready", "pid": os.getpid(), **memory_report()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        op = req.get("op")
        if op == "quit":
            break
        with _BUSY:
            if op == "run":
                started = time.monotonic()
                code = run_job(req, idle_out, idle_err)
                gc.collect()
                _LAST_ACTIVITY = time.monotonic()
                send({"event": "done", "code": code, "seconds": round(_LAST_ACTIVITY - started, 2),
                      "threads": threading.active_count(), "models": models_loaded(), **memory_report()})
            elif op == "unload":
                released = unload_models()
                _LAST_ACTIVITY = time.monotonic()
                send({"event": "unloaded", "reason": "request", "released": released, **memory_report()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
