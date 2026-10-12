import os
import unittest
from unittest.mock import patch

from modules.translation.llm.gpt import _estimate_cost_usd


class OpenAIUsageTests(unittest.TestCase):
    def test_subscription_mode_keeps_token_counts_without_fake_api_cost(self):
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
        }
        with patch.dict(os.environ, {"OPENAI_USAGE_BILLING_MODE": "subscription"}, clear=True):
            self.assertEqual(_estimate_cost_usd("cx/gpt-6-luna", usage), (1000, 200, 0, 1200, 0.0))

    def test_api_mode_retains_api_cost_estimate(self):
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
        }
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_estimate_cost_usd("gpt-5.4-mini", usage), (1000, 200, 0, 1200, 0.00165))


if __name__ == "__main__":
    unittest.main()
