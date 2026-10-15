"""Offline build check: CPU runtime, pipeline imports and Turkish font rendering."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main():
    import torch
    import onnxruntime as ort
    assert torch.version.cuda is None, "The CPU image must use CPU-only PyTorch"
    assert "CUDAExecutionProvider" not in ort.get_available_providers()
    import local_batch  # Import actual worker dependencies, without starting a job.
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFontDatabase
    app = QApplication.instance() or QApplication([])
    font = ROOT / "assets/Comic Geek.ttf"
    assert font.is_file(), "Bundled font is missing"
    assert QFontDatabase.addApplicationFont(str(font)) >= 0, "Qt cannot load the font"
    assert app is not None and local_batch is not None
    print("CPU pipeline imports and offscreen Qt font check passed")


if __name__ == "__main__":
    main()
