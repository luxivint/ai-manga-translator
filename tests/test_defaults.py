import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import local_batch as lb  # noqa: E402
from modules.utils import compute_device as cd  # noqa: E402
from modules.utils import device as dev  # noqa: E402


class UseGpuTests(unittest.TestCase):
    def test_unset_and_auto_follow_cuda(self):
        for v in (None, "", "auto", "AUTO", "weird"):
            self.assertTrue(cd.resolve_use_gpu(v, True))
            self.assertFalse(cd.resolve_use_gpu(v, False))

    def test_explicit_off(self):
        for v in ("false", "0", "no", "off", " False "):
            self.assertFalse(cd.resolve_use_gpu(v, True))

    def test_explicit_on(self):
        for v in ("true", "1", "yes", "on"):
            self.assertTrue(cd.resolve_use_gpu(v, False))


class DynamicWorkersTests(unittest.TestCase):
    def _run(self, cpu, base, env):
        fake = mock.Mock()
        fake.cpu_percent.return_value = cpu
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.dict(sys.modules, {"psutil": fake}), \
                mock.patch.object(os, "cpu_count", return_value=16):
            return lb._dynamic_workers(base)

    def test_idle_cpu_raises_by_one_up_to_cap(self):
        self.assertEqual(self._run(10, 3, {"BATCH_WORKERS_AUTO": "true", "BATCH_WORKERS_MAX": "4"}), 4)
        self.assertEqual(self._run(10, 4, {"BATCH_WORKERS_AUTO": "true", "BATCH_WORKERS_MAX": "4"}), 4)

    def test_busy_cpu_lowers(self):
        self.assertEqual(self._run(95, 3, {"BATCH_WORKERS_AUTO": "true"}), 2)
        self.assertEqual(self._run(95, 1, {"BATCH_WORKERS_AUTO": "true"}), 1)

    def test_auto_off_is_exact(self):
        self.assertEqual(self._run(5, 3, {"BATCH_WORKERS_AUTO": "false"}), 3)

    def test_explicit_base_above_cap_not_lowered(self):
        self.assertEqual(self._run(10, 6, {"BATCH_WORKERS_AUTO": "true", "BATCH_WORKERS_MAX": "4"}), 6)


class TensorRtProviderTests(unittest.TestCase):
    AVAIL = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_tensorrt_dropped_by_default(self):
        with mock.patch.dict(os.environ, {"USE_TENSORRT": "false"}), \
                mock.patch.object(dev.ort, "get_available_providers", return_value=self.AVAIL):
            self.assertEqual(dev.get_providers("cuda"), ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_tensorrt_opt_in(self):
        with mock.patch.dict(os.environ, {"USE_TENSORRT": "true"}), \
                mock.patch.object(dev.ort, "get_available_providers", return_value=self.AVAIL):
            names = [p[0] if isinstance(p, tuple) else p for p in dev.get_providers("cuda")]
            self.assertEqual(names[0], "TensorrtExecutionProvider")


class ArgsAndSaveTests(unittest.TestCase):
    def _parse(self, argv, env=None):
        with mock.patch.object(sys, "argv", ["local_batch.py"] + argv), \
                mock.patch.dict(os.environ, env or {}, clear=False):
            return lb.parse_args()

    def test_empty_suffix_forms(self):
        self.assertEqual(self._parse(["--suffix="]).suffix, "")
        self.assertEqual(self._parse(["--suffix", ""]).suffix, "")
        self.assertEqual(self._parse([], {"OUTPUT_SUFFIX": ""}).suffix, "")

    def test_gpu_flags(self):
        with mock.patch.object(cd, "ort_cuda_available", return_value=True):
            self.assertFalse(self._parse(["--no-gpu"]).gpu)
            self.assertTrue(self._parse(["--gpu"]).gpu)
            self.assertTrue(self._parse([], {"USE_GPU": "auto"}).gpu)
            self.assertFalse(self._parse([], {"USE_GPU": "false"}).gpu)

    def test_default_output_format_webp(self):
        env = os.environ.copy()
        env.pop("OUTPUT_FORMAT", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(self._parse([]).output_format, "webp")

    def test_webp_saved_high_quality_with_exact_dims(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, (120, 80, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            lo, hi = Path(d) / "lo.webp", Path(d) / "hi.webp"
            with mock.patch.dict(os.environ, {"IMAGE_QUALITY": "30"}):
                lb._write_array(lo, arr)
            with mock.patch.dict(os.environ, {"IMAGE_QUALITY": "95"}):
                lb._write_array(hi, arr)
            self.assertGreater(hi.stat().st_size, lo.stat().st_size)
            with Image.open(hi) as im:
                self.assertEqual(im.size, (80, 120))
            lb.verify_output_dims([hi], arr)

    def test_bad_quality_env_falls_back(self):
        with mock.patch.dict(os.environ, {"IMAGE_QUALITY": "abc"}):
            self.assertEqual(lb.image_quality(), 95)


if __name__ == "__main__":
    unittest.main()
