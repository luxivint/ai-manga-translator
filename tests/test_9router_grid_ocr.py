import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from modules.ocr.factory import OCRFactory
from modules.ocr.qwen_grid_ocr import OpenAICompatibleGridOCR


class NineRouterGridOCRTests(unittest.TestCase):
    def test_factory_selects_9router_grid_ocr(self):
        settings = SimpleNamespace(
            get_credentials=lambda service: {"api_key": "router-key"}
        )

        with patch.dict(
            os.environ,
            {
                "OPENAI_BASE_URL": "http://127.0.0.1:20128/v1",
                "OPENAI_MODEL": "cx/gpt-6-luna",
            },
            clear=True,
        ):
            engine = OCRFactory._create_new_engine(settings, "9Router Vision Grid OCR")

        self.assertIsInstance(engine, OpenAICompatibleGridOCR)
        self.assertEqual(engine.api_base_url, "http://127.0.0.1:20128/v1/chat/completions")
        self.assertEqual(engine.model, "cx/gpt-6-luna")
        self.assertEqual(engine.api_key, "router-key")

    def test_ocr_request_uses_router_endpoint_and_model(self):
        block = SimpleNamespace(xyxy=[10, 10, 90, 55], text="")
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '[{"id":1,"text":"Hello there"}]'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            },
        )
        with patch.dict(
            os.environ,
            {
                "OPENAI_BASE_URL": "http://127.0.0.1:20128/v1",
                "OPENAI_MODEL": "cx/gpt-6-luna",
                "OPENAI_API_KEY": "router-key",
                "QWEN_GRID_OCR_USAGE_PATH": "",
                "GATEWAY_CACHE": "false",
            },
            clear=True,
        ):
            engine = OpenAICompatibleGridOCR()
            engine.initialize()
            with patch("modules.ocr.qwen_grid_ocr.requests.post", return_value=response) as post:
                engine.process_image(np.zeros((80, 120, 3), dtype=np.uint8), [block])

        self.assertEqual(block.text, "Hello there")
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:20128/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer router-key")
        payload = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(payload["model"], "cx/gpt-6-luna")
        self.assertIn("max_completion_tokens", payload)
        self.assertTrue(payload["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
