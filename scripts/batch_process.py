"""Supervise a batch without blocking the worker heartbeat or a tray output pipe."""
import os
import subprocess
import time
from pathlib import Path


def stop_tree(process):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        process.kill()
    process.wait(timeout=10)


def run_batch_process(command, cwd, env, log_path, tick=None, stall_seconds=600):
    path = Path(log_path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as output:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
        previous = -1; last_activity = time.monotonic()
        try:
            while True:
                try:
                    code = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    size = path.stat().st_size
                    if size != previous:
                        previous = size; last_activity = time.monotonic()
                    if tick: tick()
                    if time.monotonic() - last_activity > stall_seconds:
                        raise RuntimeError(f'Translation produced no output for {stall_seconds}s; batch stopped, raw images kept')
        except BaseException:
            stop_tree(process)
            raise
    if code:
        with path.open('rb') as output:
            output.seek(max(0, path.stat().st_size - 8000))
            tail = output.read().decode('utf-8', errors='replace')
        raise subprocess.CalledProcessError(code, command, output=tail)


def batch_progress(log_path):
    path=Path(log_path)
    try:
        with path.open('rb') as log:
            log.seek(max(0,path.stat().st_size-8192))
            tail=log.read().decode('utf-8',errors='replace')
    except OSError:
        return 0
    markers={'R2 download:':5,'[RECUT]':15,'Faz 1/3':30,'Faz 2/3':55,'Faz 3/3':75,'R2 upload:':90,'R2 uploaded:':95}
    return max((progress for text,progress in markers.items() if text in tail),default=0)
