"""Gateway answer cache: retries / re-runs must not re-spend tokens (fake transport only, synthetic text)."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.utils import gateway, gateway_cache
from modules.utils.gateway import CircuitBreaker, resilient_post


class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body if body is not None else {"choices": [{"message": {"content": "hello"}}],
                                                    "usage": {"prompt_tokens": 7, "completion_tokens": 2}}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def raise_for_status(self):
        return None


class GatewayCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"GATEWAY_CACHE_DIR": self.tmp.name, "GATEWAY_CACHE": "true"})
        self.env.start()
        self.calls = []
        gateway_cache._PURGED.clear()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def post(self, status=200, body=None):
        def fake(url, **kw):
            self.calls.append((url, kw))
            return FakeResp(status, body)
        return fake

    def call(self, data='{"m":"x"}', cache="manual", headers=None, kind="translate", fake=None):
        return resilient_post("http://127.0.0.1:20128/v1/chat/completions", headers=headers or {"Authorization": "Bearer SECRET-1"},
                              data=data, timeout=5, kind=kind, breaker=CircuitBreaker(), backoffs=(),
                              post=fake or self.post(), cache=cache)

    def test_manual_answer_is_stored_only_after_commit(self):
        self.call()
        self.assertEqual(len(self.calls), 1)
        self.call()  # nothing committed -> asks again
        self.assertEqual(len(self.calls), 2)
        self.call()
        self.assertTrue(gateway.cache_commit())
        self.assertEqual(len(self.calls), 3)
        hit = self.call()
        self.assertEqual(len(self.calls), 3)  # served from disk, no request
        self.assertTrue(getattr(hit, "from_cache", False))
        self.assertEqual(hit.json()["choices"][0]["message"]["content"], "hello")

    def test_hit_has_no_usage_so_tokens_are_not_logged_twice(self):
        self.call(cache="auto")
        hit = self.call(cache="auto")
        self.assertNotIn("usage", hit.json())
        self.assertEqual(len(self.calls), 1)

    def test_key_ignores_headers_and_host_but_not_the_body(self):
        self.call(cache="auto", headers={"Authorization": "Bearer A"})
        self.call(cache="auto", headers={"Authorization": "Bearer B"})  # other key, same request
        self.assertEqual(len(self.calls), 1)
        self.call(data='{"m":"y"}', cache="auto")  # other text / prompt / model
        self.assertEqual(len(self.calls), 2)

    def test_discard_forgets_a_rejected_answer(self):
        self.call()
        gateway.cache_commit()
        self.call()  # hit
        gateway.cache_discard()  # caller found the cached answer unusable
        self.call()
        self.assertEqual(len(self.calls), 2)

    def test_errors_and_non_200_are_never_cached(self):
        self.call(cache="auto", fake=self.post(status=400, body={"error": "bad"}))
        self.call(cache="auto", fake=self.post(status=400, body={"error": "bad"}))
        self.assertEqual(len(self.calls), 2)

    def test_disabled_by_knob(self):
        with patch.dict(os.environ, {"GATEWAY_CACHE": "false"}):
            self.call(cache="auto")
            self.call(cache="auto")
        self.assertEqual(len(self.calls), 2)

    def test_no_cache_param_means_no_caching(self):
        self.call(cache=None)
        self.call(cache=None)
        self.assertEqual(len(self.calls), 2)

    def test_expired_entries_are_not_served(self):
        self.call(cache="auto")
        for f in Path(self.tmp.name).glob("*/*.json"):
            old = time.time() - 40 * 86400
            os.utime(f, (old, old))
            entry = json.loads(f.read_text())
            entry["ts"] = old
            f.write_text(json.dumps(entry))
        gateway_cache._PURGED.clear()
        self.call(cache="auto")
        self.assertEqual(len(self.calls), 2)

    def test_cache_files_hold_no_secret(self):
        self.call(cache="auto", headers={"Authorization": "Bearer SECRET-1"})
        blobs = "".join(f.read_text() for f in Path(self.tmp.name).glob("*/*.json"))
        self.assertTrue(blobs)
        self.assertNotIn("SECRET-1", blobs)

    def test_hit_works_while_quota_is_blocked(self):
        self.call(cache="auto")
        with patch.object(gateway, "raise_if_quota", side_effect=gateway.GatewayQuotaError("blocked")):
            self.call(cache="auto")  # no network, no quota check
        self.assertEqual(len(self.calls), 1)


class FakeGatewayTranslationRerun(unittest.TestCase):
    """translate_keyed through a fake gateway: the second run costs zero requests; a bad answer is not replayed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"GATEWAY_CACHE_DIR": self.tmp.name, "GATEWAY_CACHE": "true",
                                           "OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1"})
        self.env.start()
        gateway_cache._PURGED.clear()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def engine(self):
        from types import SimpleNamespace
        from modules.translation.llm.gpt import GPTTranslation
        eng = GPTTranslation()
        eng.api_key, eng.api_base_url, eng.model, eng.timeout = "k", "http://127.0.0.1:1/v1/chat/completions", "m", 5
        eng.source_lang, eng.target_lang = "English", "Turkish"
        eng.temperature, eng.top_p, eng.max_tokens = 0.2, 1.0, 1000
        eng.supports_images = False
        eng.img_as_llm_input = False
        return eng

    def test_second_run_makes_no_request_and_bad_answer_is_dropped(self):
        answers = ['{"p0_b0": "Merhaba"}', "no json at all", '{"p0_b0": "Merhaba"}']
        sent = []

        def fake_post(url, **kw):
            sent.append(1)
            return FakeResp(200, {"choices": [{"message": {"content": answers[len(sent) - 1]}}]})

        eng = self.engine()
        with patch.object(gateway, "_post", fake_post):
            out = eng.translate_keyed({"p0_b0": "Hello"})
            self.assertEqual(out, {"p0_b0": "Merhaba"})
            self.assertEqual(eng.translate_keyed({"p0_b0": "Hello"}), out)  # cached
            self.assertEqual(len(sent), 1)
            # another text -> bad answer -> RuntimeError, nothing stored, next call asks again
            with self.assertRaises(RuntimeError):
                eng.translate_keyed({"p0_b0": "Bye"})
            self.assertEqual(eng.translate_keyed({"p0_b0": "Bye"}), {"p0_b0": "Merhaba"})
            self.assertEqual(len(sent), 3)


if __name__ == "__main__":
    unittest.main()
