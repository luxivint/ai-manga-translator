"""Device selection helpers for the cleaning stage (YOLO segmenter + LaMa)."""
from __future__ import annotations

import os

import onnxruntime as ort

_VALID = ("auto", "cuda", "cpu")


def resolve_setting(setting: str | None, cuda_available: bool) -> str:
    """auto|cuda|cpu -> 'cuda' or 'cpu'. Unknown/empty values behave like auto;
    an explicit 'cuda' degrades to 'cpu' when CUDA is not available."""
    s = (setting or "auto").strip().lower()
    if s not in _VALID:
        s = "auto"
    if s == "cpu":
        return "cpu"
    return "cuda" if cuda_available else "cpu"


def env_device(var: str, cuda_available: bool) -> str:
    return resolve_setting(os.environ.get(var, "auto"), cuda_available)


def resolve_use_gpu(setting: str | None, cuda_available: bool) -> bool:
    """USE_GPU -> bool. unset/empty/auto = on when CUDA is available; true/1/yes/on = on;
    false/0/no/off = off. Unknown values behave like auto."""
    s = (setting or "auto").strip().lower()
    if s in ("false", "0", "no", "off"):
        return False
    if s in ("true", "1", "yes", "on"):
        return True
    return bool(cuda_available)


def torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def ort_cuda_available() -> bool:
    try:
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def cuda_providers() -> list | None:
    """Explicit ORT provider list for CUDA (no TensorRT), or None if unavailable.
    cudnn_conv_algo_search defaults to HEURISTIC: ORT's EXHAUSTIVE default benchmarks
    every conv algo for each new input shape (tens of seconds for LaMa)."""
    if not ort_cuda_available():
        return None
    search = os.environ.get("ORT_CUDNN_SEARCH", "HEURISTIC").strip().upper() or "HEURISTIC"
    return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": search}), "CPUExecutionProvider"]


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "failed to allocate" in msg or "cudamalloc" in msg
