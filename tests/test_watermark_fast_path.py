"""OCR-local watermark cleaning must not do another chapter scan or any model inference."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import local_batch as lb
from modules.utils import watermark as wm


def fixture():
    image = np.full((240, 360, 3), 235, np.uint8)
    image[80:120, 40:300] = (60, 60, 65)
    cv2.putText(image, 'VORTEXSCANS.COM', (65, 106), cv2.FONT_HERSHEY_SIMPLEX,
                .5, (250, 250, 250), 1, cv2.LINE_AA)
    block = SimpleNamespace(text='VORTEXSCANS.COM', translation='', is_ad=True,
                            xyxy=np.array([65, 87, 235, 110]), text_class='text_free')
    return image, block


class FastWatermarkTests(unittest.TestCase):
    def test_chapter_never_learns_templates_even_with_legacy_flags_enabled(self):
        image, block = fixture()
        args = SimpleNamespace(source_lang='English', target_lang='Turkish', detector='rtdetr')
        def detected(task):
            index, path, _, _ = task
            return index, path, image.copy(), [block], {'ocr_failed': False, 'detected': 1, 'kept': 1, 'empty': 0}
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(mock.patch.dict(os.environ, {'BATCH_WORKERS': '1', 'BATCH_WORKERS_AUTO': 'false',
                'WATERMARK_POLICY': 'clean', 'WATERMARK_PLATES': 'true', 'WATERMARK_PLATE_MINE': 'true',
                'CLEAN_CV2_REFINE_MASK': 'false'}))
            learn = stack.enter_context(mock.patch.object(wm, 'learn_plates', create=True))
            stack.enter_context(mock.patch.object(lb, '_already_done', return_value=False))
            stack.enter_context(mock.patch.object(lb, 'detect_ad_regions', return_value={}))
            stack.enter_context(mock.patch.object(lb, 'detect_banner_strips', return_value={}))
            stack.enter_context(mock.patch.object(lb, '_stage_detect_ocr', side_effect=detected))
            stack.enter_context(mock.patch.object(lb, '_batch_translate_all'))
            stack.enter_context(mock.patch.object(lb, '_rate_translation_quality', return_value=0))
            stack.enter_context(mock.patch.object(lb, '_do_render_and_save'))
            stack.enter_context(mock.patch('modules.detection.factory.DetectionEngineFactory.create_engine',
                                           return_value=SimpleNamespace(reuse_identity='api')))
            stack.enter_context(mock.patch.object(lb, '_tiled_segment', side_effect=AssertionError('extra model pass')))
            stack.enter_context(mock.patch.object(cv2, 'matchTemplate', side_effect=AssertionError('chapter search')))
            result = lb.process_all_batched([Path('001.png'), Path('002.png')], Path(directory), args)
            self.assertEqual(result, 0)
            learn.assert_not_called()

    def test_watermark_only_page_cleans_without_segmenter_or_network(self):
        image, block = fixture()
        args = SimpleNamespace(source_lang='English')
        with mock.patch.dict(os.environ, {'WATERMARK_POLICY': 'clean', 'CLEAN_CV2_REFINE_MASK': 'false'}), \
             mock.patch.object(lb, '_tiled_segment', side_effect=AssertionError('model not allowed')), \
             mock.patch.object(cv2, 'matchTemplate', side_effect=AssertionError('search not allowed')), \
             contextlib.redirect_stdout(io.StringIO()):
            result = lb._stage_clean((0, Path('001.png'), image, [block], args))[-1]
            mask = lb.watermark_clear_mask(image, [block])
        self.assertLess(float(np.abs(result[87:110, 65:235].astype(float) - 235).mean()), 2)
        self.assertTrue(np.array_equal(result[mask == 0], image[mask == 0]))

    def test_missing_ocr_does_not_guess_or_reuse_other_pages(self):
        image, _ = fixture()
        with mock.patch.dict(os.environ, {'WATERMARK_POLICY': 'clean'}):
            self.assertFalse(lb.watermark_clear_mask(image, []).any())
            mask = lb.build_segmenter_mask(image, [], require_translation=True)
            self.assertFalse(mask.any())


if __name__ == '__main__':
    unittest.main()
