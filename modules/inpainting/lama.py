import numpy as np
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from ..utils.device import get_providers
from ..utils import compute_device

from ..utils.inpainting import (
    norm_img,
    load_jit_model,
)
from ..utils.download import ModelDownloader, ModelID
from modules.utils.onnx import make_session
from modules.utils.torch_autocast import TorchAutocastMixin
from .base import InpaintModel
from .schema import Config

logger = logging.getLogger(__name__)


class LaMa(TorchAutocastMixin, InpaintModel):
    name = "lama"
    pad_mod = 8

    @staticmethod
    def should_use_autocast(device: str) -> bool:
        return str(device).split(":", 1)[0].lower() != "xpu"

    def init_model(self, device, **kwargs):
        self.backend = kwargs.get("backend")
        if self.backend == "onnx":
            ModelDownloader.get(ModelID.LAMA_ONNX)
            onnx_path = ModelDownloader.primary_path(ModelID.LAMA_ONNX)
            self._onnx_path = onnx_path
            self._cpu_session = None
            self._run_lock = threading.Semaphore(max(1, int(os.environ.get("LAMA_GPU_CONCURRENCY", "3"))))
            self.session = None
            if str(device).lower().startswith("cuda"):
                self.session = self._make_cuda_session(onnx_path)
            if self.session is None:
                providers = get_providers("cpu" if str(device).lower().startswith("cuda") else device)
                self.session = make_session(onnx_path, providers=providers)
            self.session_device = (
                "cuda" if self.session.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
            )
            if self.session_device == "cuda":
                # ORT's CUDA Conv path re-plans on every input-shape change (~0.4 s vs
                # 0.1 s); several threads calling one session overlap that cost
                # (LAMA_GPU_CONCURRENCY). Optional speed mode, OFF by default because
                # it changes the fill slightly: LAMA_GPU_BUCKETS=512,768,1024 pads
                # crops (mirror) onto a few square canvases, one warm session each;
                # bigger crops use a shared exact-shape session.
                self._buckets = tuple(sorted(int(v) for v in os.environ.get("LAMA_GPU_BUCKETS", "").split(",") if v.strip()))
                self._pool = {self._buckets[0] if self._buckets else "big": self.session}
                self._pool_lock = threading.Lock()
                # Building a session costs ~20 s (model parse); build the other
                # canvases in parallel background threads.
                rest = self._buckets[1:]
                if rest:
                    self._prewarm_ex = ThreadPoolExecutor(max_workers=len(rest))
                    for b in rest:
                        self._pool[b] = self._prewarm_ex.submit(self._make_cuda_session, onnx_path)
            logger.info("LaMa ONNX session device: %s", self.session_device)
        else:
            import torch
            ModelDownloader.get(ModelID.LAMA_JIT)
            local_path = ModelDownloader.primary_path(ModelID.LAMA_JIT) 
            self.model = load_jit_model(local_path, device)
            self.setup_torch_autocast(torch, device)
            if not self.should_use_autocast(device):
                logger.info(
                    "Disabling LaMa autocast on device '%s' because the TorchScript FFT path requires float32.",
                    device,
                )
                self.use_autocast = False

    @staticmethod
    def _make_cuda_session(onnx_path):
        """CUDA ORT session, or None (caller falls back to CPU) if unavailable."""
        providers = compute_device.cuda_providers()
        if providers is None:
            logger.warning("LaMa: CUDAExecutionProvider not available, using CPU")
            return None
        try:
            import torch  # noqa: F401  (loads the CUDA/cuDNN DLLs ORT needs)
        except Exception:
            pass
        try:
            sess = make_session(onnx_path, providers=providers)
        except Exception as exc:
            logger.warning("LaMa: CUDA session failed (%s), using CPU", exc)
            return None
        if sess.get_providers()[0] != "CUDAExecutionProvider":
            logger.warning("LaMa: CUDA provider not active (%s), using CPU", sess.get_providers())
            return None
        return sess

    def _bucket_for(self, h, w):
        """Smallest configured square canvas that holds (h, w), else None."""
        m = max(h, w)
        for b in getattr(self, "_buckets", ()):
            if m <= b:
                return b
        return None

    def _pick_session(self, key):
        with self._pool_lock:
            sess = self._pool.get(key)
            if sess is None:
                sess = self._make_cuda_session(self._onnx_path) or self.session
                self._pool[key] = sess
            elif isinstance(sess, Future):  # still-building session
                sess = sess.result() or self.session
                self._pool[key] = sess
            return sess

    def _run_onnx(self, ort_inputs, key="big"):
        """Run ORT; on a CUDA out-of-memory error retry this call on a CPU session."""
        if getattr(self, "session_device", "cpu") != "cuda":
            return self.session.run(None, ort_inputs)[0]
        try:
            with self._run_lock:
                return self._pick_session(key).run(None, ort_inputs)[0]
        except Exception as exc:
            if not compute_device.is_oom_error(exc):
                raise
            logger.warning("LaMa: GPU out of memory, retrying this crop on CPU")
            if self._cpu_session is None:
                self._cpu_session = make_session(self._onnx_path, providers=["CPUExecutionProvider"])
            names = [i.name for i in self._cpu_session.get_inputs()[:2]]
            vals = list(ort_inputs.values())
            return self._cpu_session.run(None, dict(zip(names, vals)))[0]

    @staticmethod
    def is_downloaded() -> bool:
        return ModelDownloader.is_downloaded(ModelID.LAMA_JIT)

    def forward(self, image, mask, config: Config):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        orig_hw = image.shape[:2]
        key = "big"
        if getattr(self, "session_device", "cpu") == "cuda":
            bucket = self._bucket_for(*orig_hw)
            if bucket is not None:
                key = bucket
                ph, pw = bucket - orig_hw[0], bucket - orig_hw[1]
                image = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
                if mask.ndim == 2:
                    mask = mask[:, :, np.newaxis]
                mask = np.pad(mask, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
        image_n = norm_img(image)
        mask_n = norm_img(mask)
        # mask_n = (mask_n > 0) * 1
        mask_n = (mask_n > 0).astype('float32')
        backend = getattr(self, 'backend', 'torch')
        if backend == 'onnx':
            image_tensor = image_n[np.newaxis, ...]
            mask_tensor = mask_n[np.newaxis, ...]
            ort_inputs = {self.session.get_inputs()[0].name: image_tensor,
                          self.session.get_inputs()[1].name: mask_tensor}
            inpainted = self._run_onnx(ort_inputs, key)
            cur_res = inpainted[0].transpose(1, 2, 0)
            cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
            # cur_res is already in RGB format
            return cur_res[: orig_hw[0], : orig_hw[1]]
        else:
            import torch  # noqa
            image_t = torch.from_numpy(image_n).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask_n).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                inpainted_image = self.run_with_torch_autocast(
                    torch_module=torch,
                    fn=lambda: self.model(image_t, mask_t),
                    logger=logger,
                    engine_name=self.__class__.__name__,
                )
            cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
            # cur_res is already in RGB format
            return cur_res
    
