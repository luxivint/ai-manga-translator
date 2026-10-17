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


# ---------------------------------------------------------------------------------------------------------------
# Warm mode (WORKER_INPROCESS=true): chapters run inside ONE long-lived child (scripts/warm_runner.py) so the models
# stay loaded between jobs. Same contract as run_batch_process: output goes to the job log, tick() is called every
# 5 s, a silent log trips the stall timeout, cancel/errors kill the process tree, a non-zero exit raises
# CalledProcessError. The difference: a finished job normally leaves the child alive for the next one.
# ---------------------------------------------------------------------------------------------------------------
import hashlib
import json
import queue
import sys
import threading

WARM_RUNNER = Path(__file__).resolve().parent / 'warm_runner.py'
# Differ per chapter by design; everything else in the environment is part of the child's identity.
PER_JOB_ENV = ('PIPELINE_JOB_ID', 'TRANSLATION_MANGA_ID', 'TRANSLATION_CHAPTER_ID', 'TRANSLATION_CHAPTER_NUMBER',
               'TRANSLATION_MANGA_SLUG', 'TRANSLATION_RAW_PREFIX', 'TRANSLATION_OUTPUT_PREFIX',
               'OPENAI_USAGE_OPERATION', 'TRANSLATION_SOURCE_FILE')


def env_fingerprint(env, dotenv_path=None):
    """Hash of the job environment without the per-chapter keys, plus the .env file content. Module constants read
    at import time (render sizes, storage backend, ...) are frozen in a warm child, so a changed fingerprint means
    'start a fresh child' rather than 'run with stale knobs'."""
    digest = hashlib.sha1()
    for key in sorted(k for k in env if k not in PER_JOB_ENV):
        digest.update(f'{key}={env[key]}'.encode('utf-8', 'replace') + b'\0')
    try:
        digest.update(Path(dotenv_path).read_bytes())
    except (OSError, TypeError):
        pass
    return digest.hexdigest()


def _tail(path, size=8000):
    try:
        with Path(path).open('rb') as output:
            output.seek(max(0, Path(path).stat().st_size - size))
            return output.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


class WarmRunner:
    def __init__(self, cwd, idle_seconds=600, recycle_jobs=20, recycle_rss_mb=0, recycle_vram_mb=0,
                 python=None, runner=WARM_RUNNER, child_log=None, popen=subprocess.Popen, poll=5):
        self.cwd = cwd
        self.idle_seconds = idle_seconds
        self.recycle_jobs = recycle_jobs
        self.recycle_rss_mb = recycle_rss_mb
        self.recycle_vram_mb = recycle_vram_mb
        self.python = python or sys.executable
        self.runner = str(runner)
        self.child_log = child_log
        self._popen = popen
        self.poll = poll  # seconds between tick()/stall checks
        self.process = None
        self.events = None
        self.fingerprint = None
        self.jobs_in_child = 0
        self.spawned = 0
        self.last = {}

    # -- child lifecycle -------------------------------------------------------------------------------------
    def alive(self):
        return self.process is not None and self.process.poll() is None

    def _spawn(self, fingerprint):
        command = [self.python, '-u', self.runner, '--idle-seconds', str(self.idle_seconds)]
        err = subprocess.DEVNULL
        if self.child_log:
            Path(self.child_log).parent.mkdir(parents=True, exist_ok=True)
            err = open(self.child_log, 'ab')
        try:
            self.process = self._popen(command, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=err, env=os.environ.copy())
        finally:
            if self.child_log:
                err.close()
        self.events = queue.Queue()
        threading.Thread(target=self._read, args=(self.process, self.events), daemon=True, name='warm-reader').start()
        self.fingerprint = fingerprint
        self.jobs_in_child = 0
        self.spawned += 1

    @staticmethod
    def _read(process, events):
        for raw in process.stdout:
            try:
                events.put(json.loads(raw.decode('utf-8', 'replace')))
            except ValueError:
                continue
        events.put(None)  # EOF: the child is gone

    def close(self):
        """Ask the child to exit; kill its tree if it does not. Safe to call twice."""
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is not None:
            self._release(process)
            return
        try:
            process.stdin.write(b'{"op":"quit"}\n')
            process.stdin.flush()
            process.stdin.close()
            process.wait(timeout=10)
        except Exception:
            stop_tree(process)
        self._release(process)

    def _kill(self):
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            stop_tree(process)
        if process is not None:
            self._release(process)

    @staticmethod
    def _release(process):
        for pipe in (process.stdin, process.stdout):
            try:
                pipe.close()
            except Exception:
                pass

    def _retired(self):
        """Drain old events; True if the child announced it is exiting (idle exit) or is already gone."""
        retired = False
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return retired
            if event is None or event.get('event') == 'exiting':
                retired = True

    def _ensure(self, fingerprint):
        if self.alive() and not self._retired() and self.fingerprint == fingerprint:
            return
        if self.alive():
            print('warm runner: starting a fresh child (environment changed or child retiring)', flush=True)
        self.close()
        self._spawn(fingerprint)

    def request(self, payload):
        self.process.stdin.write((json.dumps(payload) + '\n').encode('utf-8'))
        self.process.stdin.flush()

    def unload(self):
        """Ask an idle child to release its models now."""
        if self.alive():
            self.request({'op': 'unload'})

    # -- one job ---------------------------------------------------------------------------------------------
    def run(self, command, cwd, env, log_path, tick=None, stall_seconds=600, fingerprint=None):
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = fingerprint if fingerprint is not None else env_fingerprint(env)
        self._ensure(fingerprint)
        events = self.events
        process = self.process
        resent = False
        try:
            self.request({'op': 'run', 'script': str(command[1]), 'args': [str(a) for a in command[2:]],
                          'env': dict(env), 'cwd': str(cwd), 'log': str(path)})
            previous = -1
            last_activity = time.monotonic()
            while True:
                try:
                    event = events.get(timeout=self.poll)
                except queue.Empty:
                    try:
                        size = path.stat().st_size
                    except OSError:
                        size = 0
                    if size != previous:
                        previous = size
                        last_activity = time.monotonic()
                    if tick:
                        tick()
                    if time.monotonic() - last_activity > stall_seconds:
                        raise RuntimeError(f'Translation produced no output for {stall_seconds}s; batch stopped, raw images kept')
                    continue
                if event is not None and event.get('event') == 'exiting' and not resent:
                    # idle exit raced with our request: the job never started, hand it to a fresh child
                    resent = True
                    self.close()
                    self._ensure(fingerprint)
                    events, process = self.events, self.process
                    self.request({'op': 'run', 'script': str(command[1]), 'args': [str(a) for a in command[2:]],
                                  'env': dict(env), 'cwd': str(cwd), 'log': str(path)})
                    continue
                if event is None:  # the child died mid-job (native crash, killed): same as a failing subprocess
                    code = process.wait(timeout=10)
                    self.process = None
                    self._release(process)
                    raise subprocess.CalledProcessError(code or -1, command, output=_tail(path))
                if event.get('event') == 'done':
                    break
        except subprocess.CalledProcessError:
            raise
        except BaseException:
            self._kill()
            raise
        self.last = event
        self.jobs_in_child += 1
        code = int(event.get('code') or 0)
        if code:
            self.close()  # state after a failed job is not trusted: the next job gets a fresh child
            raise subprocess.CalledProcessError(code, command, output=_tail(path))
        reason = self._recycle_reason(event)
        if reason:
            print(f'warm runner: recycling child ({reason})', flush=True)
            self.close()

    def _recycle_reason(self, event):
        if self.recycle_jobs and self.jobs_in_child >= self.recycle_jobs:
            return f'{self.jobs_in_child} jobs done'
        rss = event.get('rss_mb')
        if self.recycle_rss_mb and rss and rss >= self.recycle_rss_mb:
            return f'RSS {rss} MB >= {self.recycle_rss_mb} MB'
        vram = event.get('vram_used_mb')
        if self.recycle_vram_mb and vram and vram >= self.recycle_vram_mb:
            return f'VRAM {vram} MB >= {self.recycle_vram_mb} MB'
        return None
