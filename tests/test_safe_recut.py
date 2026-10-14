import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from scripts.safe_recut import VerticalStrip, RecutError, prepare, choose_cuts, safe_bands


class RecutTests(unittest.TestCase):
    def test_virtual_strip_crop_across_seam_and_cache_bound(self):
        with tempfile.TemporaryDirectory() as work:
            paths = []
            expected = Image.new('RGB', (40, 240))
            for i, color in enumerate(['red', 'green', 'blue', 'yellow']):
                im = Image.new('RGB', (40, 60), color)
                path = Path(work) / f'{i}.png'; im.save(path); paths.append(path)
                expected.paste(im, (0, i * 60))
            strip = VerticalStrip(paths)
            try:
                for a, b in [(0, 240), (60, 120), (55, 185), (239, 240)]:
                    with strip.crop(a, b) as crop:
                        self.assertEqual(crop.tobytes(), expected.crop((0, a, 40, b)).tobytes())
                    self.assertLessEqual(len(strip.cache), 2)
            finally:
                strip.close()
            self.assertFalse(strip.cache)

    def test_model_protection_moves_cut_away_from_source_seam(self):
        with tempfile.TemporaryDirectory() as work:
            paths = []
            full = Image.new('RGB', (80, 1600), 'white')
            ImageDraw.Draw(full).rectangle((20, 700, 60, 950), fill='black')
            for i, (a, b) in enumerate([(0, 800), (800, 1600)]):
                p = Path(work) / f'{i}.png'; full.crop((0, a, 80, b)).save(p); paths.append(p)
            plan = prepare(paths, Path(work) / 'recut', lambda a: [[10, 650, 70, 1000]],
                           target=800, minimum=200, soft_max=1000, hard_max=2000)
            self.assertFalse(any(626 <= y <= 1024 for y in plan.cuts[1:-1]))
            out = hashlib.sha256()
            for p in plan.images:
                with Image.open(p) as im: out.update(im.tobytes())
            self.assertEqual(out.hexdigest(), hashlib.sha256(full.tobytes()).hexdigest())
            self.assertEqual(plan.height, 1600)

    def test_no_gap_never_forces_cut_and_hard_limit_fails(self):
        self.assertEqual(choose_cuts(12000, []), [0, 12000])
        with tempfile.TemporaryDirectory() as work:
            p = Path(work) / 'input.png'
            im = Image.new('RGB', (80, 2000), 'white')
            ImageDraw.Draw(im).line((40, 0, 40, 1999), fill='black'); im.save(p)
            before = p.read_bytes()
            plan = prepare([p], Path(work) / 'recut', lambda a: [],
                           target=500, minimum=200, soft_max=800, hard_max=1000)
            self.assertLessEqual(max(b-a for a,b in zip(plan.cuts,plan.cuts[1:])),1000)
            self.assertEqual(p.read_bytes(), before)
            self.assertTrue(list((Path(work) / 'recut').glob('*.png')))

    def test_thin_edges_and_empty_balloons_block_gaps(self):
        with tempfile.TemporaryDirectory() as work:
            p = Path(work) / 'input.png'; Image.new('RGB', (80, 2000), 'white').save(p)
            with self.assertRaisesRegex(RecutError, 'No safe cut'):
                prepare([p], Path(work) / 'recut', lambda a: [[0, 0, 80, len(a)]],
                        target=500, minimum=200, soft_max=800, hard_max=1000)

    def test_width_mismatch_and_stale_work_fail(self):
        with tempfile.TemporaryDirectory() as work:
            a, b = Path(work) / 'a.png', Path(work) / 'b.png'
            Image.new('RGB', (40, 100)).save(a); Image.new('RGB', (41, 100)).save(b)
            with self.assertRaises(RecutError): VerticalStrip([a, b])
            with self.assertRaisesRegex(RecutError, 'empty'):
                prepare([a], Path(work), lambda a: [])

    def test_short_blank_chapter_black_band_and_invalid_settings(self):
        self.assertEqual(safe_bands(np.ones(100, dtype=bool)), [(0, 100)])
        with tempfile.TemporaryDirectory() as work:
            a = Path(work) / 'a.png'; Image.new('RGB', (40, 100), 'black').save(a)
            plan = prepare([a], Path(work) / 'out', lambda a: [])
            self.assertEqual(plan.cuts, [0, 100])
            with self.assertRaises(RecutError): prepare([a], Path(work) / 'bad', lambda a: [], hard_max=20000)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import local_batch
        cls.lb = local_batch

    def args(self):
        with mock.patch.object(sys, 'argv', ['local_batch.py', '--no-gpu', '--stamp-enabled', 'false']):
            args = self.lb.parse_args()
        args.resolved_font = 'Arial'
        return args

    def test_recut_failure_cannot_run_ocr_upload_or_delete_raw(self):
        with tempfile.TemporaryDirectory() as work:
            args = self.args()
            args.r2_input_prefix = 'raw/test/chapter'; args.r2_output_prefix = 'translated/test/chapter'
            args.r2_workdir = work
            engine = mock.Mock(confidence_threshold=0.3)
            with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'fake'}), \
                 mock.patch.object(self.lb, 'download_prefix') as download, \
                 mock.patch.object(self.lb, 'iter_images', return_value=[Path('source.png')]), \
                 mock.patch('modules.detection.factory.DetectionEngineFactory.create_engine', return_value=engine), \
                 mock.patch.object(self.lb, 'prepare_safe_recut', side_effect=RecutError('no safe gap')), \
                 mock.patch.object(self.lb, 'process_all_batched') as ocr, \
                 mock.patch.object(self.lb, 'upload_directory') as upload, \
                 mock.patch.object(self.lb, 'delete_prefix') as delete:
                code = self.lb.run_batch(args, mock.Mock(), Path(work) / 'temp')
            self.assertEqual(code, self.lb.EXIT_VALIDATION_FAILED)
            ocr.assert_not_called(); upload.assert_not_called(); delete.assert_not_called()
            self.assertEqual(engine.confidence_threshold, 0.3)

    def test_enabled_pipeline_processes_new_pieces_and_replaces_stale_local_outputs(self):
        from scripts.safe_recut import RecutPlan
        with tempfile.TemporaryDirectory() as work:
            root = Path(work); source = root / 'in'; dest = root / 'out'
            source.mkdir(); dest.mkdir()
            Image.new('RGB', (40, 100), 'blue').save(source / '001.png')
            Image.new('RGB', (40, 100), 'red').save(dest / 'old.png')
            args = self.args(); args.r2_input_prefix = ''; args.input = str(source); args.output = str(dest)
            args.zip_output = False
            plan = RecutPlan([root / 'recut-00001.png'], [0, 100], 40, 100, 'hash')
            def process(images, out, settings):
                self.assertEqual(images, plan.images)
                self.assertNotEqual(out, dest)
                Image.new('RGB', (40, 100), 'green').save(out / 'recut-00001_translated.png')
                return 0
            with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'fake', 'BATCH_PIPELINE': 'true'}), \
                 mock.patch('modules.detection.factory.DetectionEngineFactory.create_engine', return_value=mock.Mock(confidence_threshold=0.3)), \
                 mock.patch.object(self.lb, 'prepare_safe_recut', return_value=plan), \
                 mock.patch.object(self.lb, 'process_all_batched', side_effect=process):
                self.assertEqual(self.lb.run_batch(args, mock.Mock(), root / 'temp'), 0)
            self.assertEqual([p.name for p in dest.iterdir()], ['recut-00001_translated.png'])
            self.assertTrue((source / '001.png').is_file())

    def test_ocr_failure_preserves_existing_outputs(self):
        from scripts.safe_recut import RecutPlan
        with tempfile.TemporaryDirectory() as work:
            root = Path(work); source = root / 'in'; dest = root / 'out'
            source.mkdir(); dest.mkdir()
            Image.new('RGB', (40, 100), 'blue').save(source / '001.png')
            Image.new('RGB', (40, 100), 'red').save(dest / 'old.png')
            before = (dest / 'old.png').read_bytes()
            args = self.args(); args.r2_input_prefix = ''; args.input = str(source); args.output = str(dest)
            with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'fake', 'BATCH_PIPELINE': 'true'}), \
                 mock.patch('modules.detection.factory.DetectionEngineFactory.create_engine', return_value=mock.Mock(confidence_threshold=0.3)), \
                 mock.patch.object(self.lb, 'prepare_safe_recut', return_value=RecutPlan([root / 'recut.png'], [0, 100], 40, 100, 'hash')), \
                 mock.patch.object(self.lb, 'process_all_batched', side_effect=self.lb.BatchValidationError('OCR failed')):
                self.assertEqual(self.lb.run_batch(args, mock.Mock(), root / 'temp'), self.lb.EXIT_VALIDATION_FAILED)
            self.assertEqual((dest / 'old.png').read_bytes(), before)

    def test_default_enabled_and_explicit_disable(self):
        with mock.patch.dict(os.environ, {'WEBTOON_SAFE_RECUT': 'true'}), \
             mock.patch.object(sys, 'argv', ['local_batch.py', '--no-gpu']):
            self.assertTrue(self.lb.parse_args().safe_recut)
        with mock.patch.object(sys, 'argv', ['local_batch.py', '--no-safe-recut', '--no-gpu']):
            self.assertFalse(self.lb.parse_args().safe_recut)

    def test_same_input_output_folder_never_deletes_source(self):
        with tempfile.TemporaryDirectory() as work:
            p = Path(work) / '001.png'; Image.new('RGB', (40, 100), 'red').save(p)
            before = p.read_bytes()
            args = self.args(); args.r2_input_prefix = ''; args.input = work; args.output = work
            with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'fake'}), \
                 mock.patch.object(self.lb, 'prepare_safe_recut') as recut:
                self.assertEqual(self.lb.run_batch(args, mock.Mock(), Path(work) / 'temp'), self.lb.EXIT_VALIDATION_FAILED)
            recut.assert_not_called()
            self.assertEqual(p.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
