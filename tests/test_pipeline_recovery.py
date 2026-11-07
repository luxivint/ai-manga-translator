"""Recovery must reuse verified work without accepting incomplete OCR or stale pixels."""
import json
import contextlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from modules.ocr.qwen_grid_ocr import QwenGridOCR
from modules.utils.gateway import OCRFailedError
from modules.utils.textblock import TextBlock
from scripts import safe_recut
from scripts.page_checkpoint import PageCheckpoints


def reply(items):
    items = [dict(script='latin', kind='dialogue', original_text=item.get('text', ''), **item) for item in items]
    response = Mock(status_code=200)
    response.json.return_value = {'choices': [{'message': {'content': json.dumps(items)}}]}
    return response


class EmptyOCRRecoveryTests(unittest.TestCase):
    def run_ocr(self, replies):
        engine = QwenGridOCR()
        engine.api_key = 'test-only'
        blocks = [TextBlock(text_bbox=np.array([0, 0, 8, 8])) for _ in range(3)]
        with patch.object(engine, '_make_sheet', side_effect=lambda image, bs:
                          (Image.new('RGB', (12, 12)), {i + 1: i for i in range(len(bs))})), \
             patch('modules.ocr.qwen_grid_ocr.gateway.resilient_post', side_effect=replies) as post, \
             patch('modules.ocr.qwen_grid_ocr.gateway.cache_commit'), \
             patch('modules.ocr.qwen_grid_ocr.gateway.cache_discard'), \
             patch.dict(os.environ, {'QWEN_GRID_OCR_RETRIES': '2'}):
            result = engine.process_image(np.zeros((10, 10, 3), dtype=np.uint8), blocks)
        return result, post.call_count

    def test_complete_but_blank_sheet_is_verified_individually(self):
        empty = [{'id': i, 'text': ''} for i in range(1, 4)]
        result, calls = self.run_ocr([reply(empty), reply(empty)] +
                                   [reply([{'id': 1, 'text': ''}]) for _ in range(3)])
        self.assertEqual([b.text for b in result], ['', '', ''])
        self.assertEqual(calls, 5)

    def test_individual_recovery_can_find_dialogue(self):
        empty = [{'id': i, 'text': ''} for i in range(1, 4)]
        result, _ = self.run_ocr([reply(empty), reply(empty), reply([{'id': 1, 'text': 'HELLO'}]),
                                  reply([{'id': 1, 'text': ''}]), reply([{'id': 1, 'text': ''}])])
        self.assertEqual(result[0].text, 'HELLO')

    def test_missing_ids_are_still_errors(self):
        with self.assertRaisesRegex(OCRFailedError, 'missing crop IDs'):
            self.run_ocr([reply([{'id': 1, 'text': ''}])] * 2)


class RecutResumeTests(unittest.TestCase):
    def test_long_namespace_keeps_generated_windows_paths_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / '001.png'
            Image.new('RGB', (30, 100), 'white').save(source)
            plan = safe_recut.prepare_cached([source], root / ('namespace-' * 4),
                lambda _: {'texts': [], 'bubbles': []}, detector_identity='v1')
            self.assertLess(max(len(str(p)) for p in plan.images), 248)

    def test_resume_verifies_source_settings_and_generated_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / '001.png'
            Image.new('RGB', (30, 100), 'white').save(source)
            detector = Mock(return_value={'texts': [], 'bubbles': []})
            first = safe_recut.prepare_cached([source], root / 'cache', detector, detector_identity='v1')
            detector.reset_mock()
            second = safe_recut.prepare_cached([source], root / 'cache', detector, detector_identity='v1')
            detector.assert_not_called()
            self.assertEqual(first.images, second.images)
            first.images[0].write_bytes(b'corrupt')
            safe_recut.prepare_cached([source], root / 'cache', detector, detector_identity='v1')
            self.assertTrue(detector.called)
            detector.reset_mock()
            Image.new('RGB', (30, 100), 'black').save(source)
            safe_recut.prepare_cached([source], root / 'cache', detector, detector_identity='v1')
            self.assertTrue(detector.called)
            detector.reset_mock()
            safe_recut.prepare_cached([source], root / 'cache', detector, detector_identity='v2')
            self.assertTrue(detector.called)


class PageResumeTests(unittest.TestCase):
    def test_retry_archives_previous_log(self):
        from scripts.batch_process import run_batch_process
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'job.log'
            log.write_text('old failure evidence', encoding='utf8')
            run_batch_process([sys.executable, '-c', 'print("new attempt")'], tmp, os.environ.copy(), log)
            self.assertIn('new attempt', log.read_text())
            self.assertEqual(next((Path(tmp) / 'history').glob('*.log')).read_text(), 'old failure evidence')

    def test_render_failure_reuses_ocr_and_translation_on_retry(self):
        import local_batch as lb
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / '1.png'
            Image.new('RGB', (10, 10), 'white').save(source)
            second = root / '2.png'
            Image.new('RGB', (10, 10), 'blue').save(second)
            args = SimpleNamespace(checkpoint_dir=root / 'checkpoints', source_lang='English',
                target_lang='Turkish', translator='test', ocr='test', output_format='webp',
                suffix='_translated', detector='test')
            block = TextBlock(text_bbox=np.array([0, 0, 8, 8]), text='hello')
            def result(task):
                return (task[0], task[1], np.full((10, 10, 3), 255, dtype=np.uint8), [block.deep_copy()],
                        {'ocr_failed': False, 'detected': 1, 'kept': 1, 'empty': 0})
            def translate(pages, args):
                for _, _, blocks in pages:
                    for b in blocks:
                        b.translation = 'merhaba'
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {'PAGE_MAX_RETRIES': '1',
                    'BATCH_OCR_WORKERS': '1', 'BATCH_WORKERS_AUTO': 'false',
                    'SKIP_FULL_AD_PAGES': 'false', 'TRANSLATION_QUALITY_SCORE': 'false'}))
                stack.enter_context(patch.object(lb, 'LocalSettings'))
                stack.enter_context(patch.object(lb, 'LocalMain'))
                stack.enter_context(patch('modules.detection.factory.DetectionEngineFactory.create_engine',
                                          return_value=SimpleNamespace(reuse_identity='test')))
                stack.enter_context(patch.object(lb, 'detect_ad_regions', return_value={}))
                stack.enter_context(patch.object(lb, 'detect_banner_strips', return_value={}))
                detect = stack.enter_context(patch.object(lb, '_stage_detect_ocr', side_effect=result))
                translated = stack.enter_context(patch.object(lb, '_batch_translate_all', side_effect=translate))
                stack.enter_context(patch.object(lb, '_stage_clean_safe',
                    side_effect=lambda t: (t[0], t[1], t[2], t[3], t[2])))
                def render_page(idx, total, path, image, blocks, cleaned, output, settings, main):
                    if path == second:
                        raise ValueError('temporary render problem')
                    target = lb._output_path(path, output, settings)
                    Image.fromarray(image).save(target)
                    lb._mark_output_complete(path, target, settings)
                    return target
                render = stack.enter_context(patch.object(lb, '_do_render_and_save', side_effect=render_page))
                with self.assertRaises(lb.BatchValidationError):
                    lb.process_all_batched([source, second], root / 'out', args)
                render.side_effect = None
                lb.process_all_batched([source, second], root / 'out', args)
                self.assertEqual(detect.call_count, 2)
                self.assertEqual(translated.call_count, 1)

    def test_clean_failure_does_not_become_a_completed_passthrough(self):
        import local_batch as lb
        with patch.object(lb, '_stage_clean', side_effect=ValueError('temporary clean problem')):
            with self.assertRaisesRegex(lb.BatchValidationError, 'clean failed'):
                lb._stage_clean_safe((0, Path('1.png'), np.zeros((5, 5, 3), dtype=np.uint8), [], None))

    def test_uniform_fast_path_matches_original(self):
        a = np.random.default_rng(42).integers(0, 256, (100, 60, 3), dtype=np.uint8)
        a[:20] = 255
        a[20:40] = np.random.default_rng(3).integers(100, 108, (20, 60, 3), dtype=np.uint8)
        original = a.astype(np.int16)
        expected = ((original.max(axis=1)-original.min(axis=1)).max(axis=1) <= 22) & \
                   (original.std(axis=1).max(axis=1) <= 3.5)
        np.testing.assert_array_equal(safe_recut.uniform_rows(a), expected)

    def test_checkpoint_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PageCheckpoints(Path(tmp))
            block = TextBlock(text_bbox=np.array([1, 2, 5, 8]), text='hello', translation='merhaba')
            block.custom_flag = True
            pixels = np.full((10, 8, 3), 200, dtype=np.uint8)
            store.save('ocr', Path('1.png'), 'source-settings-v1', pixels, [block], {'ocr_failed': False})
            restored = store.load('ocr', Path('1.png'), 'source-settings-v1')
            np.testing.assert_array_equal(restored[0], pixels)
            np.testing.assert_array_equal(restored[1][0].xyxy, block.xyxy)
            self.assertEqual(restored[1][0].translation, 'merhaba')
            self.assertTrue(restored[1][0].custom_flag)
            self.assertIsNone(store.load('ocr', Path('1.png'), 'changed-source'))
            next(Path(tmp).glob('*.png')).write_bytes(b'corrupt')
            self.assertIsNone(store.load('ocr', Path('1.png'), 'source-settings-v1'))

    def test_failed_ocr_is_never_checkpointed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PageCheckpoints(Path(tmp))
            store.save('ocr', Path('1.png'), 'v1', np.zeros((10, 10, 3), dtype=np.uint8), [],
                       {'ocr_failed': True})
            self.assertIsNone(store.load('ocr', Path('1.png'), 'v1'))

    def test_successful_ocr_page_is_skipped_on_retry(self):
        import local_batch as lb
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / '1.png'
            Image.new('RGB', (10, 10), 'white').save(source)
            args = SimpleNamespace(checkpoint_dir=root / 'checkpoints', source_lang='English',
                target_lang='Turkish', translator='test', ocr='test', output_format='webp',
                suffix='_translated', detector='test')
            task = (0, source, args, None)
            result = (0, source, np.full((10, 10, 3), 255, dtype=np.uint8),
                      [TextBlock(text_bbox=np.array([0, 0, 8, 8]), text='hello')], {'ocr_failed': False})
            with patch.object(lb, '_stage_detect_ocr', return_value=result) as process:
                lb._stage_detect_ocr_resumable(task)
                restored = lb._stage_detect_ocr_resumable((5, source, args, None))
                self.assertEqual(process.call_count, 1)
                self.assertEqual(restored[0], 5)
                self.assertEqual(restored[3][0].text, 'hello')
                lb._stage_detect_ocr_resumable((0, source, args, (10, 0)))
                self.assertEqual(process.call_count, 2)


if __name__ == '__main__':
    unittest.main()
