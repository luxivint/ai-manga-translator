"""Concurrent requests on one translation engine must not mutate token caps."""
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from modules.translation.llm.base import BaseLLMTranslation


class FakeEngine(BaseLLMTranslation):
    def __init__(self):
        super().__init__()
        self.source_lang = "English"
        self.target_lang = "Turkish"
        self.max_tokens = 5000
        self.barrier = threading.Barrier(2)
        self.observed = []

    def get_system_prompt(self, *args, **kwargs):
        return "Translate as JSON."

    def _perform_translation(self, user_prompt, system_prompt, image):
        self.barrier.wait(timeout=5)
        cap = getattr(self._request_local, "max_tokens", None)
        self.observed.append(cap)
        raw = json.loads(user_prompt.split("Translate this:\n", 1)[1])
        return json.dumps({key: "ok" for key in raw})


class ConcurrentTokenLimitsTests(unittest.TestCase):
    def test_concurrent_requests_have_independent_limits(self):
        engine = FakeEngine()
        small = {"p0_b0": "Hello"}
        large = {f"p1_b{i}": "Hello" for i in range(40)}
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(engine.translate_keyed, small)
            b = pool.submit(engine.translate_keyed, large)
            self.assertEqual(len(a.result(timeout=10)), 1)
            self.assertEqual(len(b.result(timeout=10)), 40)
        self.assertEqual(sorted(engine.observed), [8000, 16000])
        self.assertEqual(engine.max_tokens, 5000)


if __name__ == "__main__":
    unittest.main()
