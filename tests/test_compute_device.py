import unittest
from unittest.mock import patch

from modules.utils import compute_device as cd


class ResolveSettingTests(unittest.TestCase):
    def test_auto_uses_cuda_when_available(self):
        self.assertEqual(cd.resolve_setting("auto", True), "cuda")

    def test_auto_falls_back_to_cpu(self):
        self.assertEqual(cd.resolve_setting("auto", False), "cpu")

    def test_cpu_always_cpu(self):
        self.assertEqual(cd.resolve_setting("cpu", True), "cpu")

    def test_cuda_requested_but_missing_falls_back(self):
        self.assertEqual(cd.resolve_setting("cuda", False), "cpu")

    def test_cuda_requested_and_present(self):
        self.assertEqual(cd.resolve_setting("cuda", True), "cuda")

    def test_garbage_and_empty_mean_auto(self):
        self.assertEqual(cd.resolve_setting("", True), "cuda")
        self.assertEqual(cd.resolve_setting(None, False), "cpu")
        self.assertEqual(cd.resolve_setting(" CUDA ", True), "cuda")
        self.assertEqual(cd.resolve_setting("tpu", True), "cuda")


class AvailabilityTests(unittest.TestCase):
    def test_ort_cuda_provider_detected(self):
        with patch.object(cd.ort, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            self.assertTrue(cd.ort_cuda_available())
        with patch.object(cd.ort, "get_available_providers", return_value=["CPUExecutionProvider"]):
            self.assertFalse(cd.ort_cuda_available())

    def test_ort_error_means_unavailable(self):
        with patch.object(cd.ort, "get_available_providers", side_effect=RuntimeError("x")):
            self.assertFalse(cd.ort_cuda_available())

    def test_env_device_reads_variable(self):
        with patch.dict("os.environ", {"SEG_DEVICE": "cpu"}):
            self.assertEqual(cd.env_device("SEG_DEVICE", True), "cpu")
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(cd.env_device("SEG_DEVICE", True), "cuda")
            self.assertEqual(cd.env_device("SEG_DEVICE", False), "cpu")


class OomTests(unittest.TestCase):
    def test_detects_oom_messages(self):
        self.assertTrue(cd.is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate")))
        self.assertTrue(cd.is_oom_error(RuntimeError("Failed to allocate memory for requested buffer")))
        self.assertTrue(cd.is_oom_error(RuntimeError("cudaMalloc failed: out of memory")))
        self.assertFalse(cd.is_oom_error(ValueError("bad shape")))


class LamaProviderTests(unittest.TestCase):
    def test_cuda_providers_when_available(self):
        with patch.object(cd.ort, "get_available_providers", return_value=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]):
            self.assertEqual(
                cd.cuda_providers(),
                [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"],
            )

    def test_cuda_providers_none_when_missing(self):
        with patch.object(cd.ort, "get_available_providers", return_value=["CPUExecutionProvider"]):
            self.assertIsNone(cd.cuda_providers())


if __name__ == "__main__":
    unittest.main()
