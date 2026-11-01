import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import local_batch as lb  # noqa: E402


class FakeYolo:
    def __init__(self, fail_first=None):
        self.calls = []
        self.fail_first = fail_first

    def predict(self, crop, **kw):
        self.calls.append(kw)
        if self.fail_first is not None and len(self.calls) == 1:
            raise self.fail_first
        return ["ok"]


class SegPredictTests(unittest.TestCase):
    def test_cpu_path_uses_cpu_device_without_half(self):
        m = FakeYolo()
        with mock.patch.object(lb, "_SEG_DEVICE", "cpu"):
            self.assertEqual(lb._seg_predict(m, None, conf=0.25), ["ok"])
        self.assertEqual(m.calls[0]["device"], "cpu")
        self.assertNotIn("half", m.calls[0])

    def test_cuda_path_passes_cuda_device(self):
        m = FakeYolo()
        with mock.patch.object(lb, "_SEG_DEVICE", "cuda"), mock.patch.object(lb, "_SEG_HALF", False):
            lb._seg_predict(m, None, conf=0.25)
        self.assertEqual(m.calls[0]["device"], "cuda")

    def test_cuda_oom_retries_on_cpu_copy(self):
        gpu = FakeYolo(fail_first=RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
        cpu = FakeYolo()
        with mock.patch.object(lb, "_SEG_DEVICE", "cuda"), mock.patch.object(lb, "_SEG_HALF", False), \
                mock.patch.object(lb, "_SEGMENTER_CPU", cpu):
            self.assertEqual(lb._seg_predict(gpu, None, conf=0.25), ["ok"])
        self.assertEqual(cpu.calls[0]["device"], "cpu")

    def test_non_oom_error_propagates(self):
        gpu = FakeYolo(fail_first=ValueError("bad"))
        with mock.patch.object(lb, "_SEG_DEVICE", "cuda"), mock.patch.object(lb, "_SEG_HALF", False):
            with self.assertRaises(ValueError):
                lb._seg_predict(gpu, None, conf=0.25)




if __name__ == "__main__":
    unittest.main()
