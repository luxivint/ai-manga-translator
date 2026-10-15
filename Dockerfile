FROM python:3.12-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV QT_QPA_PLATFORM=offscreen
ENV PIP_NO_CACHE_DIR=1
# CPU-only server defaults; container settings can override thread counts.
ENV USE_GPU=false SEG_DEVICE=cpu LAMA_DEVICE=cpu USE_TENSORRT=false
ENV BATCH_WORKERS=1 BATCH_WORKERS_AUTO=false BATCH_WORKERS_MAX=1
ENV BATCH_TRANSLATE_CONCURRENCY=1 OCR_CONCURRENCY=1 TRANSLATION_STALL_SECONDS=1800
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
ENV XDG_DATA_HOME=/app/work/data XDG_CACHE_HOME=/app/work/cache
ENV YOLO_CONFIG_DIR=/app/work/ultralytics

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    libdbus-1-3 \
    libegl1 \
    libfontconfig1 \
    libfreetype6 \
    libgl1 \
    libopengl0 \
    libgomp1 \
    libglib2.0-0 \
    libice6 \
    libsm6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxext6 \
    libxkbcommon0 \
    libxrender1 \
    fonts-dejavu-core \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
  && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
  && python -m pip install -r requirements.txt \
  && python -c "import torch, onnxruntime; assert torch.version.cuda is None; assert 'CUDAExecutionProvider' not in onnxruntime.get_available_providers()"

RUN useradd --create-home --uid 10001 worker
COPY --chown=worker:worker . .
RUN mkdir -p input output work && chown -R worker:worker input output work
USER worker
RUN python scripts/container_smoke.py
STOPSIGNAL SIGTERM
CMD ["python", "scripts/container_entrypoint.py"]
