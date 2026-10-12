"""Transport resilience for the OpenAI-compatible gateway (9router): retry/backoff, circuit breaker,
OCR concurrency limit, OCR failed-vs-empty signalling. HTTP is mocked; time is injected (no real sleeps)."""
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.utils import gateway
from modules.utils.gateway import AdaptiveLimiter, CircuitBreaker, GatewayError, OCRFailedError, resilient_post


class FakeClock:
    """sleep() advances the clock, so waits are asserted without real time passing."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self.ok = status < 400
        self._body = {"ok": True} if body is None else body
        self.text = text if text is not None else json.dumps(self._body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def ok(content="[]"):
    return FakeResponse(200, {"choices": [{"message": {"content": content}}], "usage": {}})


def r503():
    return FakeResponse(503, {"error": {"message": "Failed to convert streaming response to JSON"}})


def make_breaker(fc: FakeClock, **kw):
    return CircuitBreaker(threshold=3, cooldown_start=20, cooldown_max=60, clock=fc.clock, sleep=fc.sleep, **kw)


class ResilientPostTests(unittest.TestCase):
    def call(self, responses, fc=None, backoffs=(3, 8, 20), **kw):
        fc = fc or FakeClock()
        seq = list(responses)
        calls = []

        def fake_post(url, **k):
            calls.append(url)
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        breaker = kw.pop("breaker", None) or make_breaker(fc)
        resp = resilient_post("http://x/v1/chat/completions", headers={}, data="{}", timeout=5, kind="test",
                              breaker=breaker, backoffs=backoffs, post=fake_post, sleep=fc.sleep,
                              rand=lambda: 0.5, **kw)
        return resp, calls, fc, breaker

    def test_503_then_success_retries_with_backoff(self):
        resp, calls, fc, breaker = self.call([r503(), r503(), ok()])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(calls), 3)
        # jitter factor with rand=0.5 is 1.0 -> exactly the configured delays
        self.assertEqual([round(s, 2) for s in fc.sleeps], [3.0, 8.0])
        self.assertFalse(breaker.is_open())

    def test_timeout_and_connection_error_are_transient(self):
        resp, calls, _fc, _b = self.call([requests.Timeout("t"), requests.ConnectionError("c"), ok()])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(calls), 3)

    def test_malformed_200_body_is_transient(self):
        bad = FakeResponse(200, ValueError("not json"), text="<html>")
        resp, calls, _fc, _b = self.call([bad, ok()])
        self.assertEqual(len(calls), 2)
        self.assertEqual(resp.status_code, 200)

    def test_persistent_503_raises_after_all_attempts(self):
        with self.assertRaises(GatewayError) as ctx:
            self.call([r503(), r503(), r503(), r503()])
        self.assertIn("503", str(ctx.exception))

    def test_client_error_is_not_retried_and_returned(self):
        resp, calls, fc, _b = self.call([FakeResponse(400, {"error": "bad"})])
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(len(calls), 1)
        self.assertEqual(fc.sleeps, [])

    def test_jitter_stays_within_bounds(self):
        for rnd, lo, hi in ((0.0, 2.2, 2.3), (1.0, 3.7, 3.8)):
            fc = FakeClock()
            seq = [r503(), ok()]
            resilient_post("http://x", headers={}, data="{}", timeout=1, kind="t", breaker=make_breaker(fc),
                           backoffs=(3,), post=lambda *a, **k: seq.pop(0), sleep=fc.sleep, rand=lambda r=rnd: r)
            self.assertTrue(lo <= fc.sleeps[0] <= hi, fc.sleeps)


class CircuitBreakerTests(unittest.TestCase):
    def test_opens_after_three_consecutive_failures_and_waits_cooldown(self):
        fc = FakeClock()
        b = make_breaker(fc)
        b.record_failure(); b.record_failure()
        self.assertFalse(b.is_open())
        b.record_failure()
        self.assertTrue(b.is_open())
        waited = b.wait_if_open()
        self.assertAlmostEqual(waited, 20.0)
        self.assertFalse(b.is_open())

    def test_success_resets_counter_and_cooldown(self):
        fc = FakeClock()
        b = make_breaker(fc)
        for _ in range(3):
            b.record_failure()
        b.wait_if_open()
        b.record_failure()          # first failure after cool-down re-trips at once, cooldown grows
        self.assertAlmostEqual(b.wait_if_open(), 40.0)
        b.record_success()
        for _ in range(2):
            b.record_failure()
        self.assertFalse(b.is_open())   # counter was reset
        for _ in range(1):
            b.record_failure()
        self.assertAlmostEqual(b.wait_if_open(), 20.0)  # cooldown back to the start value

    def test_cooldown_grows_to_cap(self):
        fc = FakeClock()
        b = make_breaker(fc)
        waits = []
        for _ in range(3):
            b.record_failure()
        for _ in range(4):
            waits.append(b.wait_if_open())
            b.record_failure()
        self.assertEqual([round(w) for w in waits], [20, 40, 60, 60])

    def test_failures_while_open_do_not_extend_cooldown(self):
        fc = FakeClock()
        b = make_breaker(fc)
        for _ in range(6):          # six in-flight calls all fail at the same moment
            b.record_failure()
        self.assertAlmostEqual(b.wait_if_open(), 20.0)

    def test_every_call_waits_while_open(self):
        fc = FakeClock()
        b = make_breaker(fc)
        for _ in range(3):
            b.record_failure()
        seq = [ok()]
        resilient_post("http://x", headers={}, data="{}", timeout=1, kind="ocr", breaker=b, backoffs=(3,),
                       post=lambda *a, **k: seq.pop(0), sleep=fc.sleep, rand=lambda: 0.5)
        self.assertEqual(fc.sleeps, [20.0])   # waited the cool-down before the first attempt
        self.assertFalse(b.is_open())

    def test_resilient_post_trips_breaker_on_repeated_5xx(self):
        fc = FakeClock()
        b = make_breaker(fc)
        seq = [r503(), r503(), r503(), ok()]
        resilient_post("http://x", headers={}, data="{}", timeout=1, kind="ocr", breaker=b, backoffs=(3, 8, 20),
                       post=lambda *a, **k: seq.pop(0), sleep=fc.sleep, rand=lambda: 0.5)
        # 3 failures -> breaker opened; the 4th attempt first waits max(backoff, cooldown) via the breaker
        self.assertIn(20.0, [round(s, 1) for s in fc.sleeps])
        self.assertFalse(b.is_open())


class LimiterTests(unittest.TestCase):
    def test_limit_is_never_exceeded(self):
        limiter = AdaptiveLimiter(2)
        active = peak = 0
        lock = threading.Lock()

        def work():
            nonlocal active, peak
            with limiter.slot():
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.03)
                with lock:
                    active -= 1

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(peak, 2)

    def test_drops_to_one_while_breaker_open(self):
        fc = FakeClock()
        b = make_breaker(fc)
        limiter = AdaptiveLimiter(3, breaker=b)
        self.assertEqual(limiter.allowed(), 3)
        for _ in range(3):
            b.record_failure()
        self.assertEqual(limiter.allowed(), 1)
        fc.now += 25
        self.assertEqual(limiter.allowed(), 3)

    def test_ocr_limiter_follows_env(self):
        with mock.patch.dict("os.environ", {"OCR_CONCURRENCY": "5"}):
            self.assertEqual(gateway.ocr_limiter().max_n, 5)
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("OCR_CONCURRENCY", None)
            self.assertEqual(gateway.ocr_limiter().max_n, 2)


class OCREngineTests(unittest.TestCase):
    def make_engine(self):
        from modules.ocr.qwen_grid_ocr import OpenAICompatibleGridOCR
        eng = OpenAICompatibleGridOCR()
        with mock.patch.dict("os.environ", {"OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}):
            eng.initialize(api_key="k", model="m")
        return eng

    def blocks(self, n=1):
        from modules.utils.textblock import TextBlock
        out = []
        for i in range(n):
            b = TextBlock(text_bbox=np.array([10, 10 + 60 * i, 120, 50 + 60 * i]))
            b.text = ""
            out.append(b)
        return out

    def run_ocr(self, responses, n=1):
        eng = self.make_engine()
        fc = FakeClock()
        seq = list(responses)
        breaker = make_breaker(fc)
        img = np.full((400, 300, 3), 255, dtype=np.uint8)
        blks = self.blocks(n)

        def fake_post(url, **k):
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        patches = [
            mock.patch.object(gateway, "_post", fake_post),
            mock.patch.object(gateway, "_sleep", fc.sleep),
            mock.patch.object(gateway, "get_breaker", lambda: breaker),
            mock.patch.object(gateway, "_rand", lambda: 0.5),
        ]
        for p in patches: p.start()
        try:
            return eng.process_image(img, blks), fc
        finally:
            for p in patches: p.stop()

    def test_ocr_503_then_success_reads_text(self):
        out, fc = self.run_ocr([r503(), ok('[{"id":1,"text":"HELLO THERE"}]')])
        self.assertEqual(out[0].text, "HELLO THERE")
        self.assertEqual(len(fc.sleeps), 1)

    def test_persistent_503_raises_ocr_failed_not_empty(self):
        with self.assertRaises(OCRFailedError):
            self.run_ocr([r503()] * 4)

    def test_genuinely_empty_ocr_stays_empty(self):
        out, fc = self.run_ocr([ok('[{"id":1,"text":""}]')])
        self.assertEqual(out[0].text, "")
        self.assertEqual(fc.sleeps, [])

    def test_empty_array_is_empty_not_failure(self):
        out, _fc = self.run_ocr([ok("[]")])
        self.assertEqual(out[0].text, "")

    def test_garbage_content_is_failure(self):
        with self.assertRaises(OCRFailedError):
            self.run_ocr([ok("sorry, I cannot"), ok("sorry again")])

    def test_client_error_is_failure(self):
        with self.assertRaises(OCRFailedError):
            self.run_ocr([FakeResponse(401, {"error": "no"})])


if __name__ == "__main__":
    unittest.main()
