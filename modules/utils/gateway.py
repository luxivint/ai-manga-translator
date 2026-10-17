"""Resilient transport for the OpenAI-compatible gateway (9router -> ChatGPT/Codex plan).

The upstream drops streams under concurrent load and answers 503 ("Failed to convert streaming response to JSON").
Every OCR / translate / vision call goes through `resilient_post`:

* transient failures (HTTP 5xx, 429, timeout, connection error, 200 with an undecodable body) are retried with
  exponential backoff + jitter (GATEWAY_BACKOFFS, default "3,8,20" seconds -> 3 retries after the first try);
* a process-wide circuit breaker counts consecutive transient failures; after GATEWAY_BREAKER_THRESHOLD (3) of
  them EVERY call waits a cool-down (20 s, doubling to 60 s) before trying again; one success resets it;
* OCR calls additionally hold a process-wide slot (OCR_CONCURRENCY, default 2), independent of BATCH_WORKERS.

After the last attempt `GatewayError` is raised: a failed call is never reported as an empty answer.
HTTP 4xx (except 429) is not transient: the response is returned for the caller to judge.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import contextlib
import os
import random
import threading
import time
from typing import Callable

import requests

from modules.utils import gateway_cache


class GatewayError(RuntimeError):
    """The gateway call failed after all retries (as opposed to answering with nothing)."""


class GatewayQuotaError(GatewayError):
    """Account usage quota is exhausted; repeated page calls cannot recover it."""


def quota_path():
    return Path(os.environ.get("GATEWAY_QUOTA_FILE", str(Path(__file__).resolve().parents[2] / "work/gateway-quota.json")))


def quota_remaining():
    try:
        return max(0, json.loads(quota_path().read_text())["until"] - time.time())
    except (OSError, ValueError, KeyError):
        return 0


def raise_if_quota():
    remaining = quota_remaining()
    if remaining:
        raise GatewayQuotaError(f"Model usage limit reached; retry after {remaining:.0f} seconds")


def remember_quota(text):
    if not any(x in text.lower() for x in ("usage limit has been reached", "quota exceeded")):
        return False
    reset = text.lower().split("reset after", 1)
    seconds = sum(int(n) * {"h":3600,"m":60,"s":1}[u] for n,u in re.findall(r"(\d+)\s*([hms])", reset[-1])) if len(reset)>1 else 1800
    seconds = max(60, min(seconds + 30, 86400))
    path = quota_path(); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps({"until": time.time()+max(seconds, quota_remaining()), "reason":"Model usage limit reached"}))
    os.replace(temp, path)
    return True


class OCRFailedError(GatewayError):
    """OCR could not be obtained for a page (distinct from a page that genuinely has no readable text)."""


def _post(url, **kwargs):
    return requests.post(url, **kwargs)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _rand() -> float:
    return random.random()


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def default_backoffs() -> tuple[float, ...]:
    raw = os.environ.get("GATEWAY_BACKOFFS", "3,8,20")
    try:
        values = tuple(float(x) for x in raw.split(",") if x.strip())
    except ValueError:
        values = ()
    return values or (3.0, 8.0, 20.0)


class CircuitBreaker:
    """Consecutive-failure breaker. While open, `wait_if_open` blocks until the cool-down is over.

    The first failure after a cool-down re-opens it at once (with a longer cool-down) because the counter is
    only reset by a success. Failures reported while it is already open (calls that were in flight) neither
    count nor extend the cool-down."""

    def __init__(self, threshold: int = 3, cooldown_start: float = 20.0, cooldown_max: float = 60.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.threshold = max(1, int(threshold))
        self.cooldown_start = float(cooldown_start)
        self.cooldown_max = float(cooldown_max)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._fails = 0
        self._open_until = 0.0
        self._cooldown = self.cooldown_start

    def is_open(self) -> bool:
        with self._lock:
            return self._clock() < self._open_until

    def wait_if_open(self) -> float:
        waited = 0.0
        while True:
            with self._lock:
                remaining = self._open_until - self._clock()
            if remaining <= 0:
                return waited
            self._sleep(remaining)
            waited += remaining

    def record_success(self) -> None:
        with self._lock:
            self._fails = 0
            self._open_until = 0.0
            self._cooldown = self.cooldown_start

    def record_failure(self) -> None:
        with self._lock:
            if self._clock() < self._open_until:
                return
            self._fails += 1
            if self._fails >= self.threshold:
                self._open_until = self._clock() + self._cooldown
                print(f"  [gateway] circuit open: {self._fails} consecutive failures, cooling down "
                      f"{self._cooldown:.0f}s", flush=True)
                self._cooldown = min(self.cooldown_max, self._cooldown * 2)


_BREAKER: CircuitBreaker | None = None
_BREAKER_LOCK = threading.Lock()


def get_breaker() -> CircuitBreaker:
    """The process-wide breaker shared by OCR, translate and vision calls (they hit the same gateway)."""
    global _BREAKER
    with _BREAKER_LOCK:
        if _BREAKER is None:
            _BREAKER = CircuitBreaker(
                threshold=int(_float_env("GATEWAY_BREAKER_THRESHOLD", 3)),
                cooldown_start=_float_env("GATEWAY_BREAKER_COOLDOWN", 20),
                cooldown_max=_float_env("GATEWAY_BREAKER_COOLDOWN_MAX", 60),
            )
        return _BREAKER


class AdaptiveLimiter:
    """At most `max_n` holders at once; only 1 while the (optional) breaker is open."""

    def __init__(self, max_n: int, breaker: CircuitBreaker | None = None, poll: float = 0.5):
        self.max_n = max(1, int(max_n))
        self._breaker = breaker
        self._poll = poll
        self._cond = threading.Condition()
        self._active = 0

    def allowed(self) -> int:
        return 1 if (self._breaker is not None and self._breaker.is_open()) else self.max_n

    @contextlib.contextmanager
    def slot(self):
        with self._cond:
            while self._active >= self.allowed():
                self._cond.wait(timeout=self._poll)
            self._active += 1
        try:
            yield
        finally:
            with self._cond:
                self._active -= 1
                self._cond.notify_all()


_OCR_LIMITER: AdaptiveLimiter | None = None
_OCR_LIMITER_LOCK = threading.Lock()


def ocr_limiter() -> AdaptiveLimiter:
    """Process-wide OCR slot pool (OCR_CONCURRENCY, default 2); rebuilt only if the env value changes."""
    global _OCR_LIMITER
    try:
        n = max(1, int(os.environ.get("OCR_CONCURRENCY", "2")))
    except ValueError:
        n = 2
    with _OCR_LIMITER_LOCK:
        if _OCR_LIMITER is None or _OCR_LIMITER.max_n != n:
            _OCR_LIMITER = AdaptiveLimiter(n)
        return _OCR_LIMITER


def _short(text: str, limit: int = 200) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


_CACHE_STATE = threading.local()


def cache_commit() -> bool:
    """Store the answer of this thread's last `cache="manual"` call (the caller has validated it)."""
    pending = getattr(_CACHE_STATE, "pending", None)
    if not pending:
        return False
    key, kind, body = pending
    _CACHE_STATE.pending = None
    _CACHE_STATE.committed = key
    return gateway_cache.put(key, kind, body)


def cache_discard() -> None:
    """The caller rejected the last answer (missing keys, suspicious OCR): forget it so a retry asks again."""
    pending = getattr(_CACHE_STATE, "pending", None)
    committed = getattr(_CACHE_STATE, "committed", None)
    _CACHE_STATE.pending = None
    _CACHE_STATE.committed = None
    for key in (pending[0] if pending else None, committed):
        if key:
            gateway_cache.drop(key)


def resilient_post(url: str, *, headers: dict, data, timeout: float, kind: str = "api",
                   limiter: AdaptiveLimiter | None = None, breaker: CircuitBreaker | None = None,
                   backoffs: tuple[float, ...] | None = None, post=None, sleep=None, rand=None,
                   cache: str | None = None):
    """POST with retry/backoff + circuit breaker. Returns a response (2xx with a decodable JSON body, or a
    non-retryable 4xx). Raises GatewayError once every attempt failed transiently.

    cache=None: no local cache. cache="auto": a stored answer for the identical request is returned without any network
    call and every good answer is stored. cache="manual": same lookup, but the answer is stored only when the caller
    calls `cache_commit()` after validating it (`cache_discard()` if it turns out bad). See gateway_cache."""
    _CACHE_STATE.pending = None
    _CACHE_STATE.committed = None
    cache_key = None
    if cache and gateway_cache.enabled():
        cache_key = gateway_cache.make_key(kind, url, data)
        hit = gateway_cache.get(cache_key)
        if hit is not None:
            _CACHE_STATE.committed = cache_key
            print(f"  [gateway-cache] hit kind={kind} (no request sent)", flush=True)
            return hit
    resp = _resilient_post(url, headers=headers, data=data, timeout=timeout, kind=kind, limiter=limiter,
                           breaker=breaker, backoffs=backoffs, post=post, sleep=sleep, rand=rand)
    if cache_key is not None and getattr(resp, "status_code", 0) == 200:
        try:
            body = resp.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            if cache == "auto":
                gateway_cache.put(cache_key, kind, body)
                _CACHE_STATE.committed = cache_key
            else:
                _CACHE_STATE.pending = (cache_key, kind, body)
    return resp


def _resilient_post(url: str, *, headers: dict, data, timeout: float, kind: str = "api",
                    limiter: AdaptiveLimiter | None = None, breaker: CircuitBreaker | None = None,
                    backoffs: tuple[float, ...] | None = None, post=None, sleep=None, rand=None):
    post = post or _post
    sleep = sleep or _sleep
    rand = rand or _rand
    breaker = breaker or get_breaker()
    delays = tuple(backoffs) if backoffs is not None else default_backoffs()
    attempts = len(delays) + 1
    last = "no attempt made"
    for attempt in range(1, attempts + 1):
        raise_if_quota()
        breaker.wait_if_open()
        problem = None
        resp = None
        try:
            with (limiter.slot() if limiter is not None else contextlib.nullcontext()):
                resp = post(url, headers=headers, data=data, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            problem = f"{type(exc).__name__}: {_short(str(exc), 400)}"
        else:
            status = resp.status_code
            if status >= 400 and remember_quota(getattr(resp, "text", "")):
                raise_if_quota()
            if status >= 500 or status == 429:
                problem = f"HTTP {status} {_short(getattr(resp, 'text', ''))}"
            elif status == 200:
                try:
                    resp.json()
                except ValueError:
                    problem = f"HTTP 200 with undecodable body {_short(getattr(resp, 'text', ''), 80)!r}"
        if problem is None:
            breaker.record_success()
            return resp
        breaker.record_failure()
        last = problem
        if attempt >= attempts:
            break
        delay = delays[attempt - 1] * (0.75 + 0.5 * rand())
        print(f"  [gateway] {kind} attempt {attempt}/{attempts} failed: {problem}; "
              f"retry in {delay:.1f}s", flush=True)
        if not breaker.is_open():  # an open breaker already imposes its (longer) cool-down on the next try
            sleep(delay)
    raise GatewayError(f"{kind} failed after {attempts} attempts: {last}")
