"""A hallucinated Latin reading of Korean graphics must never reach clean/render."""
import json
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from modules.ocr.qwen_grid_ocr import QwenGridOCR
from modules.utils.gateway import OCRFailedError
from modules.utils.textblock import TextBlock


def response(items):
    result = Mock(status_code=200)
    result.json.return_value = {'choices': [{'message': {'content': json.dumps(items)}}]}
    return result


class SourceScriptTests(unittest.TestCase):
    def run_ocr(self, items, followup=None, count=1, box=None):
        engine = QwenGridOCR(); engine.api_key = 'test-only'
        blocks = [TextBlock(text_bbox=np.array(box or [0, 0, 15, 15]), text_class='text_free') for _ in range(count)]
        replies = [response(items)] + [response(followup)]*2 if followup is not None else [response(items)]
        with patch.object(engine, '_make_sheet', side_effect=lambda image, bs:
                (Image.new('RGB', (20, 20)), {i+1: i for i in range(len(bs))})), \
             patch('modules.ocr.qwen_grid_ocr.gateway.resilient_post', side_effect=replies) as post, \
             patch('modules.ocr.qwen_grid_ocr.gateway.cache_commit'), \
             patch('modules.ocr.qwen_grid_ocr.gateway.cache_discard'):
            engine.process_image(np.full((20, 20, 3), 255, dtype=np.uint8), blocks)
        return blocks, post.call_count

    def test_hangul_misread_as_of_is_preserved(self):
        blocks, calls = self.run_ocr([{'id': 1, 'text': 'OF', 'original_text': '아', 'script': 'hangul', 'kind': 'sfx'}])
        self.assertTrue(blocks[0].preserve_original)
        self.assertEqual(blocks[0].text, '')
        self.assertEqual(calls, 1)

    def test_legacy_cache_requires_independent_source_verification(self):
        blocks, calls = self.run_ocr([{'id': 1, 'text': 'FOOTBALL TEAM'}],
            [{'id': 1, 'original_text': '쾅', 'script': 'hangul', 'kind': 'sfx'}])
        self.assertTrue(blocks[0].preserve_original)
        self.assertEqual(calls, 2)

    def test_explicit_latin_dialogue_survives(self):
        blocks, _ = self.run_ocr([{'id': 1, 'text': 'HELLO', 'original_text': 'HELLO', 'script': 'latin', 'kind': 'dialogue'}])
        self.assertEqual(blocks[0].text, 'HELLO')
        self.assertFalse(blocks[0].preserve_original)

    def test_confident_but_suspicious_large_free_read_is_checked_again(self):
        blocks, calls = self.run_ocr(
            [{'id': 1, 'text': 'OF', 'original_text': 'OF', 'script': 'latin', 'kind': 'sign'}],
            [{'id': 1, 'original_text': '아', 'script': 'hangul', 'kind': 'sfx'}], box=[0, 0, 120, 200])
        self.assertTrue(blocks[0].preserve_original)
        self.assertEqual(calls, 2)

    def test_genuine_large_latin_title_survives_independent_check(self):
        blocks, calls = self.run_ocr(
            [{'id': 1, 'text': 'CHAPTER ONE', 'original_text': 'CHAPTER ONE', 'script': 'latin', 'kind': 'narration'}],
            [{'id': 1, 'original_text': 'CHAPTER ONE', 'script': 'latin', 'kind': 'narration'}], box=[0, 0, 120, 200])
        self.assertFalse(blocks[0].preserve_original)
        self.assertEqual(blocks[0].text, 'CHAPTER ONE')
        self.assertEqual(calls, 2)

    def test_verified_sfx_only_sheet_is_not_a_failed_blank_read(self):
        blocks, calls = self.run_ocr([{'id': i, 'text': '', 'original_text': '아', 'script': 'hangul', 'kind': 'sfx'}
                                    for i in range(1, 4)], count=3)
        self.assertEqual(calls, 1)
        self.assertTrue(all(b.preserve_original for b in blocks))

    def test_missing_verifier_ids_fail_instead_of_guessing(self):
        with self.assertRaises(OCRFailedError):
            self.run_ocr([{'id': 1, 'text': 'OF'}], [])

    def test_preserved_blocks_survive_filters_without_empty_bubble_recovery(self):
        import local_batch as lb
        block = TextBlock(text_bbox=np.array([0, 0, 15, 15]), text_class='text_bubble')
        block.preserve_original = True
        with patch.dict(os.environ, {'GPT_OCR_EMPTY_BUBBLES': 'true', 'OPENAI_API_KEY': 'test-only'}), \
             patch.object(lb.gateway, 'resilient_post') as post:
            self.assertEqual(lb.prune_ocr_blocks(lb.drop_non_english_ocr_blocks([block])), [block])
            lb.recover_empty_bubble_text_with_gpt(np.zeros((20, 20, 3), dtype=np.uint8), [block], None)
            post.assert_not_called()


if __name__ == '__main__':
    unittest.main()
