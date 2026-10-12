import os
import unittest
from unittest.mock import patch

from modules.utils.openai_endpoint import (
    get_openai_chat_completions_url,
    get_openai_model,
)


class OpenAIEndpointTests(unittest.TestCase):
    def test_defaults_to_openai_api(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                get_openai_chat_completions_url(),
                "https://api.openai.com/v1/chat/completions",
            )

    def test_appends_chat_completions_to_custom_base_url(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://localhost:20128/v1/"}, clear=True):
            self.assertEqual(
                get_openai_chat_completions_url(),
                "http://localhost:20128/v1/chat/completions",
            )

    def test_does_not_duplicate_chat_completions_path(self):
        with patch.dict(
            os.environ,
            {"OPENAI_BASE_URL": "http://localhost:20128/v1/chat/completions"},
            clear=True,
        ):
            self.assertEqual(
                get_openai_chat_completions_url(),
                "http://localhost:20128/v1/chat/completions",
            )

    def test_openai_model_override_takes_precedence(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": "cx/gpt-6-luna"}, clear=True):
            self.assertEqual(get_openai_model("gpt-5.4-mini"), "cx/gpt-6-luna")

    def test_selected_model_is_used_without_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_openai_model("gpt-4.1-mini"), "gpt-4.1-mini")

    def test_rejects_non_http_base_url(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "file:///tmp/endpoint"}, clear=True):
            with self.assertRaises(ValueError):
                get_openai_chat_completions_url()


if __name__ == "__main__":
    unittest.main()
