"""Offline build check: CPU runtime, pipeline imports and Turkish font rendering."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def check_pipeline():
    import local_batch  # Import actual worker dependencies, without starting a job.
    from runtime_config import configure_cleaner
    profile = configure_cleaner()
    assert profile["engine"] == "cv2" and not profile["postprocess"]
    assert not hasattr(local_batch, "_get_lama")
    assert not (ROOT / "modules/inpainting/lama.py").exists()
    import numpy as np
    image=np.full((64,80,3),180,np.uint8)
    mask=np.zeros((64,80),np.uint8);mask[20:30,25:45]=255
    result=local_batch.clean_selected(image,mask)
    assert np.array_equal(result[mask==0],image[mask==0])
    assert local_batch.loaded_models() == []
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFontDatabase
    app = QApplication.instance() or QApplication([])
    font = ROOT / "assets/Comic Geek.ttf"
    assert font.is_file(), "Bundled font is missing"
    assert QFontDatabase.addApplicationFont(str(font)) >= 0, "Qt cannot load the font"
    assert app is not None and local_batch is not None
    print("CPU pipeline imports and offscreen Qt font check passed")


def main():
    import torch
    import onnxruntime as ort
    assert torch.version.cuda is None, "The CPU image must use CPU-only PyTorch"
    assert "CUDAExecutionProvider" not in ort.get_available_providers()
    check_pipeline()


if __name__ == "__main__":
    main()
