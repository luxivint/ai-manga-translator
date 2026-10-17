"""LaMa CUDA out-of-memory recovery: drop the session, free the GPU, rebuild in arena-shrink mode, retry on the GPU
(full crop, then smaller) and only then fall back to CPU. Fake sessions only: no model, no GPU, no network."""
import gc
import os
import sys
import threading
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.inpainting import lama as lama_mod  # noqa: E402
from modules.inpainting.lama import LaMa  # noqa: E402

OOM = "oom"  # script marker: run() raises a fresh out-of-memory error (no shared traceback keeps sessions alive)


class FakeSession:
    """Stands in for an ORT session. `script` lists per-call outcomes: an exception to raise or 'ok'."""

    def __init__(self, script=(), tag="gpu"):
        self.script = list(script)
        self.tag = tag
        self.calls = []

    def get_inputs(self):
        return [SimpleNamespace(name="image"), SimpleNamespace(name="mask")]

    def get_providers(self):
        return ["CUDAExecutionProvider"]

    def run(self, _out, inputs, run_options=None):
        self.calls.append((dict(inputs), run_options))
        step = self.script.pop(0) if self.script else "ok"
        if step == OOM:
            raise RuntimeError("CUDA failure 2: out of memory")
        if isinstance(step, BaseException):
            raise step
        image = inputs["image"]
        return [np.full((1, 3) + image.shape[-2:], 0.5, np.float32)]


def make_lama(session, *, rebuilds=0):
    lm = object.__new__(LaMa)
    lm.session_device = "cuda"
    lm.session = session
    lm._input_names = ["image", "mask"]
    lm._run_lock = threading.Semaphore(3)
    lm._pool_lock = threading.RLock()
    lm._pool = {"big": session}
    lm._onnx_path = "x.onnx"
    lm._cpu_session = None
    lm._oom_rebuilds = rebuilds
    lm._gpu_gen = 0
    lm._run_options = None
    return lm


def inputs(h=64, w=96):
    return {"image": np.zeros((1, 3, h, w), np.float32), "mask": np.zeros((1, 1, h, w), np.float32)}


class LamaOomRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.release = mock.patch.object(lama_mod, "release_gpu_memory")
        self.release_mock = self.release.start()
        self.addCleanup(self.release.stop)

    def test_oom_rebuilds_shrink_session_and_retries_on_gpu(self):
        old = FakeSession([OOM])
        new = FakeSession()
        lm = make_lama(old)
        old_ref = weakref.ref(old)
        with mock.patch.object(LaMa, "_make_cuda_session", return_value=new) as build:
            out = lm._run_onnx(inputs(), "big")
        build.assert_called_once_with("x.onnx", True)  # shrink-mode providers
        self.assertEqual(out.shape, (1, 3, 64, 96))
        self.assertEqual(len(new.calls), 1)
        self.assertIs(lm.session, new)
        self.assertEqual(lm.session_device, "cuda")
        self.assertGreaterEqual(self.release_mock.call_count, 1)
        del old
        gc.collect()
        self.assertIsNone(old_ref(), "the OOM'd CUDA session must be dropped, not kept alive")

    def test_second_oom_retries_once_with_smaller_crop_then_works(self):
        old = FakeSession([OOM])
        new = FakeSession([OOM])  # full crop OOMs again, the 0.75 crop fits
        lm = make_lama(old)
        with mock.patch.object(LaMa, "_make_cuda_session", return_value=new):
            out = lm._run_onnx(inputs(64, 96), "big")
        self.assertEqual(len(new.calls), 2)
        small = new.calls[1][0]["image"].shape
        self.assertEqual(small[-2:], (48, 72))  # 0.75 scale, multiples of 8
        self.assertEqual(out.shape, (1, 3, 64, 96))  # resized back to the original crop
        self.assertEqual(lm.session_device, "cuda")

    def test_gpu_retries_exhausted_falls_back_to_cpu_for_that_crop(self):
        old = FakeSession([OOM])
        new = FakeSession([OOM, OOM])
        cpu = FakeSession(tag="cpu")
        lm = make_lama(old)
        with mock.patch.object(LaMa, "_make_cuda_session", return_value=new), \
                mock.patch.object(lama_mod, "make_session", return_value=cpu) as ms:
            out = lm._run_onnx(inputs(), "big")
        self.assertEqual(len(cpu.calls), 1)
        self.assertEqual(out.shape, (1, 3, 64, 96))
        self.assertEqual(ms.call_args.kwargs["providers"], ["CPUExecutionProvider"])
        self.assertEqual(lm.session_device, "cuda")  # later crops try the GPU again

    def test_rebuild_cap_goes_straight_to_cpu_and_still_frees_memory(self):
        old = FakeSession([OOM])
        cpu = FakeSession(tag="cpu")
        lm = make_lama(old, rebuilds=2)
        with mock.patch.object(LaMa, "_make_cuda_session") as build, \
                mock.patch.object(lama_mod, "make_session", return_value=cpu):
            lm._run_onnx(inputs(), "big")
        build.assert_not_called()
        self.assertEqual(len(cpu.calls), 1)
        self.assertGreaterEqual(self.release_mock.call_count, 1)

    def test_failed_rebuild_switches_to_cpu(self):
        old = FakeSession([OOM])
        cpu = FakeSession(tag="cpu")
        lm = make_lama(old)
        with mock.patch.object(LaMa, "_make_cuda_session", return_value=None), \
                mock.patch.object(lama_mod, "make_session", return_value=cpu):
            out = lm._run_onnx(inputs(), "big")
        self.assertEqual(lm.session_device, "cpu")
        self.assertIs(lm.session, cpu)
        self.assertEqual(out.shape, (1, 3, 64, 96))

    def test_non_oom_error_is_not_swallowed(self):
        lm = make_lama(FakeSession([ValueError("bad input")]))
        with mock.patch.object(LaMa, "_make_cuda_session") as build:
            with self.assertRaises(ValueError):
                lm._run_onnx(inputs(), "big")
        build.assert_not_called()

    def test_concurrent_oom_rebuilds_once(self):
        old = FakeSession([OOM, OOM, OOM])
        new = FakeSession()
        lm = make_lama(old)
        results = []

        def work():
            results.append(lm._run_onnx(inputs(), "big"))

        with mock.patch.object(LaMa, "_make_cuda_session", return_value=new) as build:
            threads = [threading.Thread(target=work) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(10)
        self.assertEqual(len(results), 3)
        self.assertEqual(build.call_count, 1)

    def test_close_drops_sessions_and_frees_memory(self):
        sess = FakeSession()
        lm = make_lama(sess)
        ref = weakref.ref(sess)
        lm.close()
        del sess
        gc.collect()
        self.assertIsNone(ref())
        self.assertGreaterEqual(self.release_mock.call_count, 1)


class LamaArenaOptionTests(unittest.TestCase):
    def test_shrink_session_uses_same_as_requested_arena(self):
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
        sess = mock.Mock()
        sess.get_providers.return_value = ["CUDAExecutionProvider"]
        with mock.patch.object(lama_mod.compute_device, "cuda_providers", return_value=providers), \
                mock.patch.object(lama_mod, "make_session", return_value=sess) as ms:
            LaMa._make_cuda_session("x.onnx", True)
            LaMa._make_cuda_session("x.onnx")
        shrunk, plain = ms.call_args_list[0].kwargs["providers"], ms.call_args_list[1].kwargs["providers"]
        self.assertEqual(shrunk[0][1]["arena_extend_strategy"], "kSameAsRequested")
        self.assertEqual(shrunk[0][1]["cudnn_conv_algo_search"], "HEURISTIC")
        self.assertNotIn("arena_extend_strategy", plain[0][1])
        self.assertEqual(providers[0][1], {"cudnn_conv_algo_search": "HEURISTIC"})  # caller's list untouched

    def test_arena_shrink_run_option_only_when_enabled(self):
        sess = FakeSession()
        lm = make_lama(sess)
        with mock.patch.dict(os.environ, {"LAMA_ARENA_SHRINK": "false"}):
            lm._ort_run(sess, inputs())
        self.assertIsNone(sess.calls[0][1])
        with mock.patch.dict(os.environ, {"LAMA_ARENA_SHRINK": "true"}):
            lm._ort_run(sess, inputs())
        self.assertIsNotNone(sess.calls[1][1])  # real onnxruntime.RunOptions carrying gpu:0 shrinkage


if __name__ == "__main__":
    unittest.main()
