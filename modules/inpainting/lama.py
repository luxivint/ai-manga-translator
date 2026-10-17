import numpy as np
import gc
import logging
import os
import sys
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

# Serialises the "drop the CUDA session and rebuild it" recovery after an out-of-memory error.
_OOM_LOCK = threading.Lock()


def _truthy(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def release_gpu_memory() -> None:
    """Collect garbage and hand torch's cached CUDA blocks back to the driver (no-op without torch/CUDA)."""
    gc.collect()
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        pass


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
                self.session = self._make_cuda_session(onnx_path, self._shrink_arena())
            if self.session is None:
                providers = get_providers("cpu" if str(device).lower().startswith("cuda") else device)
                self.session = make_session(onnx_path, providers=providers)
            self.session_device = (
                "cuda" if self.session.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
            )
            self._input_names = [i.name for i in self.session.get_inputs()[:2]]
            self._oom_rebuilds = 0
            self._gpu_gen = 0
            self._run_options = None
            if self.session_device == "cuda":
                # ORT's CUDA Conv path re-plans on every input-shape change (~0.4 s vs
                # 0.1 s); several threads calling one session overlap that cost
                # (LAMA_GPU_CONCURRENCY). Optional speed mode, OFF by default because
                # it changes the fill slightly: LAMA_GPU_BUCKETS=512,768,1024 pads
                # crops (mirror) onto a few square canvases, one warm session each;
                # bigger crops use a shared exact-shape session.
                self._buckets = tuple(sorted(int(v) for v in os.environ.get("LAMA_GPU_BUCKETS", "").split(",") if v.strip()))
                self._pool = {self._buckets[0] if self._buckets else "big": self.session}
                self._pool_lock = threading.RLock()
                # Building a session costs ~20 s (model parse); build the other
                # canvases in parallel background threads.
                rest = self._buckets[1:]
                if rest:
                    self._prewarm_ex = ThreadPoolExecutor(max_workers=len(rest))
                    for b in rest:
                        self._pool[b] = self._prewarm_ex.submit(self._make_cuda_session, onnx_path, self._shrink_arena())
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
    def _shrink_arena() -> bool:
        """LAMA_ARENA_SHRINK=true: allocate exactly what is requested (kSameAsRequested) and give the CUDA arena
        back after every run. Costs a little speed per run; keeps VRAM from creeping. OFF by default."""
        return _truthy("LAMA_ARENA_SHRINK")

    @staticmethod
    def _make_cuda_session(onnx_path, shrink=False):
        """CUDA ORT session, or None (caller falls back to CPU) if unavailable.
        shrink=True builds it with arena_extend_strategy=kSameAsRequested (no power-of-two arena growth)."""
        providers = compute_device.cuda_providers()
        if providers is None:
            logger.warning("LaMa: CUDAExecutionProvider not available, using CPU")
            return None
        if shrink:
            providers = [
                (p[0], {**p[1], "arena_extend_strategy": "kSameAsRequested"})
                if isinstance(p, tuple) and p[0] == "CUDAExecutionProvider" else p
                for p in providers
            ]
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
                sess = self._make_cuda_session(self._onnx_path, getattr(self, "_shrunk", False) or self._shrink_arena()) or self.session
                self._pool[key] = sess
            elif isinstance(sess, Future):  # still-building session
                sess = sess.result() or self.session
                self._pool[key] = sess
            return sess

    def _ort_run(self, sess, ort_inputs):
        """One ORT run. With LAMA_ARENA_SHRINK=true the CUDA arena is shrunk right after the run
        (RunOptions 'memory.enable_memory_arena_shrinkage' = 'gpu:0')."""
        if self._shrink_arena() and getattr(self, "session_device", "cpu") == "cuda":
            if getattr(self, "_run_options", None) is None:
                import onnxruntime as ort
                ro = ort.RunOptions()
                ro.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
                self._run_options = ro
            return sess.run(None, ort_inputs, self._run_options)[0]
        return sess.run(None, ort_inputs)[0]

    def _run_onnx(self, ort_inputs, key="big"):
        """Run ORT. On a CUDA out-of-memory error: free the GPU, rebuild a shrink-mode session and retry on the
        GPU (full crop, then a smaller one); only then run this crop on a CPU session."""
        if getattr(self, "session_device", "cpu") != "cuda":
            return self.session.run(None, ort_inputs)[0]
        gen = getattr(self, "_gpu_gen", 0)
        try:
            with self._run_lock:
                return self._ort_run(self._pick_session(key), ort_inputs)
        except Exception as exc:
            if not compute_device.is_oom_error(exc):
                raise
            logger.warning("LaMa: GPU out of memory, releasing the GPU and retrying")
        result = self._retry_after_oom(ort_inputs, key, gen)
        if result is not None:
            return result
        logger.warning("LaMa: GPU retry failed, running this crop on CPU")
        if self._cpu_session is None:
            self._cpu_session = make_session(self._onnx_path, providers=["CPUExecutionProvider"])
        names = [i.name for i in self._cpu_session.get_inputs()[:2]]
        vals = list(ort_inputs.values())
        return self._cpu_session.run(None, dict(zip(names, vals)))[0]

    def _retry_after_oom(self, ort_inputs, key, gen):
        """Drop the CUDA session(s), collect garbage, empty torch's cache, rebuild one session with arena-shrink
        options and retry on the GPU: first the full crop, then a LAMA_OOM_RETRY_SCALE (0.75) smaller one.
        At most LAMA_OOM_REBUILDS (2) rebuilds per LaMa object, then OOM goes straight to CPU. None = give up."""
        with _OOM_LOCK:
            if getattr(self, "_gpu_gen", 0) == gen:  # nobody rebuilt since our failed run
                limit = int(os.environ.get("LAMA_OOM_REBUILDS", "2") or 0)
                if getattr(self, "_oom_rebuilds", 0) >= limit:
                    release_gpu_memory()
                    return None
                self._oom_rebuilds = getattr(self, "_oom_rebuilds", 0) + 1
                if not self._rebuild_gpu_session():
                    return None
            if getattr(self, "session_device", "cpu") != "cuda":
                return None
        scales = [1.0]
        try:
            small = float(os.environ.get("LAMA_OOM_RETRY_SCALE", "0.75"))
        except ValueError:
            small = 0.75
        if 0.2 <= small < 1.0:
            scales.append(small)
        for scale in scales:
            try:
                with self._run_lock:
                    return self._run_scaled(self._pick_session(key), ort_inputs, scale)
            except Exception as exc:
                if not compute_device.is_oom_error(exc):
                    logger.warning("LaMa: GPU retry (scale %.2f) failed: %s", scale, exc)
                    return None
                logger.warning("LaMa: GPU out of memory again at scale %.2f", scale)
                release_gpu_memory()
        return None

    def _rebuild_gpu_session(self) -> bool:
        """Free every CUDA session, then build a single shrink-mode one. False = no GPU session (now on CPU)."""
        with self._pool_lock:
            self._pool = {}
            self.session = None
            self._shutdown_prewarm()
            release_gpu_memory()
            try:
                new = self._make_cuda_session(self._onnx_path, True)
            except Exception as exc:
                logger.warning("LaMa: CUDA session rebuild failed (%s)", exc)
                new = None
            self._gpu_gen = getattr(self, "_gpu_gen", 0) + 1
            if new is None:
                self.session_device = "cpu"
                if self._cpu_session is None:
                    self._cpu_session = make_session(self._onnx_path, providers=["CPUExecutionProvider"])
                self.session = self._cpu_session
                return False
            self._shrunk = True
            self._pool = {"big": new}
            self.session = new
            logger.info("LaMa: CUDA session rebuilt in arena-shrink mode")
            return True

    def _run_scaled(self, sess, ort_inputs, scale):
        """Run at `scale` (<1: image/mask downsized to multiples of 8, result resized back)."""
        if scale >= 1.0:
            return self._ort_run(sess, ort_inputs)
        import cv2
        names = list(ort_inputs.keys())
        image, mask = (np.asarray(v) for v in ort_inputs.values())
        h, w = image.shape[-2:]
        nh, nw = max(8, int(h * scale) // 8 * 8), max(8, int(w * scale) // 8 * 8)
        small_img = cv2.resize(np.ascontiguousarray(image[0].transpose(1, 2, 0)), (nw, nh), interpolation=cv2.INTER_AREA)
        small_msk = cv2.resize(np.ascontiguousarray(mask[0, 0]), (nw, nh), interpolation=cv2.INTER_NEAREST)
        out = self._ort_run(sess, {
            names[0]: np.ascontiguousarray(small_img.transpose(2, 0, 1)[None], dtype=image.dtype),
            names[1]: np.ascontiguousarray(small_msk[None, None], dtype=mask.dtype),
        })
        big = cv2.resize(np.ascontiguousarray(out[0].transpose(1, 2, 0)), (w, h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(big.transpose(2, 0, 1)[None])

    def _shutdown_prewarm(self):
        ex = getattr(self, "_prewarm_ex", None)
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)
            self._prewarm_ex = None

    def close(self):
        """Release every ORT session (idle unload). The object is unusable afterwards."""
        pool_lock = getattr(self, "_pool_lock", None) or threading.RLock()
        with pool_lock:
            self._shutdown_prewarm()
            self._pool = {}
            self.session = None
            self._cpu_session = None
        release_gpu_memory()

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
            names = getattr(self, "_input_names", None) or [i.name for i in self.session.get_inputs()[:2]]
            ort_inputs = {names[0]: image_tensor, names[1]: mask_tensor}
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
    
