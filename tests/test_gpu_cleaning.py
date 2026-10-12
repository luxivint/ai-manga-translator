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
from modules.inpainting import lama as lama_mod  # noqa: E402
from modules.inpainting.lama import LaMa  # noqa: E402


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


class LamaGpuTests(unittest.TestCase):
    def test_missing_cuda_provider_gives_none(self):
        with mock.patch.object(lama_mod.compute_device, "cuda_providers", return_value=None):
            self.assertIsNone(LaMa._make_cuda_session("x.onnx"))

    def test_session_creation_failure_gives_none(self):
        with mock.patch.object(lama_mod.compute_device, "cuda_providers", return_value=["CUDAExecutionProvider"]), \
                mock.patch.object(lama_mod, "make_session", side_effect=RuntimeError("no cudnn")):
            self.assertIsNone(LaMa._make_cuda_session("x.onnx"))

    def test_silent_cpu_fallback_by_ort_gives_none(self):
        sess = mock.Mock()
        sess.get_providers.return_value = ["CPUExecutionProvider"]
        with mock.patch.object(lama_mod.compute_device, "cuda_providers", return_value=["CUDAExecutionProvider"]), \
                mock.patch.object(lama_mod, "make_session", return_value=sess):
            self.assertIsNone(LaMa._make_cuda_session("x.onnx"))

    def test_bucket_selection(self):
        lm = object.__new__(LaMa)
        lm._buckets = (512, 768, 1024)
        self.assertEqual(lm._bucket_for(300, 500), 512)
        self.assertEqual(lm._bucket_for(512, 513), 768)
        self.assertEqual(lm._bucket_for(1024, 10), 1024)
        self.assertIsNone(lm._bucket_for(1025, 10))

    def test_oom_in_ort_retries_on_cpu_session(self):
        lm = object.__new__(LaMa)
        lm.session_device = "cuda"
        lm._run_lock = __import__("threading").Semaphore(1)
        lm._onnx_path = "x.onnx"
        lm._cpu_session = None
        gpu = mock.Mock()
        gpu.run.side_effect = RuntimeError("CUDA failure 2: out of memory")
        lm._pool = {"big": gpu}
        lm._pool_lock = __import__("threading").Lock()
        lm.session = gpu
        cpu = mock.Mock()
        cpu.get_inputs.return_value = [mock.Mock(), mock.Mock()]
        cpu.get_inputs.return_value[0].name = "image"
        cpu.get_inputs.return_value[1].name = "mask"
        cpu.run.return_value = ["cpu-result"]
        with mock.patch.object(lama_mod, "make_session", return_value=cpu):
            out = lm._run_onnx({"a": np.zeros(1), "b": np.zeros(1)}, "big")
        self.assertEqual(out, "cpu-result")
        self.assertEqual(list(cpu.run.call_args[0][1]), ["image", "mask"])

    def test_cpu_device_keeps_plain_session_path(self):
        lm = object.__new__(LaMa)
        lm.session_device = "cpu"
        lm.session = mock.Mock()
        lm.session.run.return_value = ["r"]
        self.assertEqual(lm._run_onnx({"a": 1}), "r")


if __name__ == "__main__":
    unittest.main()
