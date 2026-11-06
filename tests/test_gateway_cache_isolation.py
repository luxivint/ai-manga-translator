"""Provider-separated gateway response cache regression tests."""
import unittest

from modules.utils.gateway_cache import make_key


class GatewayCacheIsolationTests(unittest.TestCase):
    def test_different_hosts_never_share_cached_translation(self):
        body = '{"model":"shared","messages":[{"role":"user","content":"hello"}]}'
        a = make_key("translate", "https://gateway-one.example/v1/chat/completions", body)
        b = make_key("translate", "https://gateway-two.example/v1/chat/completions", body)
        self.assertNotEqual(a, b)

    def test_different_models_never_share_cached_translation(self):
        endpoint = "https://gateway.example/v1/chat/completions"
        self.assertNotEqual(make_key("translate", endpoint, '{"model":"A"}'),
                            make_key("translate", endpoint, '{"model":"B"}'))

    def test_auth_query_token_does_not_become_cache_path(self):
        body = '{"model":"x"}'
        self.assertEqual(
            make_key("translate", "https://gateway.example/v1/chat/completions?token=a", body),
            make_key("translate", "https://gateway.example/v1/chat/completions?token=b", body),
        )

    def test_same_request_is_stable(self):
        args = ("ocr", "https://gateway.example/v1/chat/completions", '{"model":"x"}')
        self.assertEqual(make_key(*args), make_key(*args))


if __name__ == "__main__":
    unittest.main()
