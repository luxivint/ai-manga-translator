import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import container_smoke as smoke

class CpuImageSmokeTests(unittest.TestCase):
    def test_gpu_torch_rejected_before_pipeline(self):
        with mock.patch.dict(sys.modules,{'torch':SimpleNamespace(version=SimpleNamespace(cuda='13')),'onnxruntime':SimpleNamespace(get_available_providers=lambda:['CPUExecutionProvider'])}),mock.patch.object(smoke,'check_pipeline') as check:
            with self.assertRaises(AssertionError):smoke.main()
            check.assert_not_called()

    def test_gpu_ort_rejected(self):
        with mock.patch.dict(sys.modules,{'torch':SimpleNamespace(version=SimpleNamespace(cuda=None)),'onnxruntime':SimpleNamespace(get_available_providers=lambda:['CUDAExecutionProvider','CPUExecutionProvider'])}),mock.patch.object(smoke,'check_pipeline') as check:
            with self.assertRaises(AssertionError):smoke.main()
            check.assert_not_called()

    def test_cpu_image_runs_real_pipeline_gate(self):
        with mock.patch.dict(sys.modules,{'torch':SimpleNamespace(version=SimpleNamespace(cuda=None)),'onnxruntime':SimpleNamespace(get_available_providers=lambda:['CPUExecutionProvider'])}),mock.patch.object(smoke,'check_pipeline') as check:
            smoke.main();check.assert_called_once()

if __name__=='__main__':unittest.main()
