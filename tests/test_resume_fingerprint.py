"""Source, code and settings fingerprinted resume tests."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import local_batch as lb


def args():
    return SimpleNamespace(source_lang="English", target_lang="Turkish",
        translator="GPT-5.4-mini", ocr="9Router Vision Grid OCR",
        output_format="webp", suffix="_translated", detector="RT-DETR-v4-s-int8")


class ResumeFingerprintTests(unittest.TestCase):
    def test_existing_output_without_marker_is_reprocessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "page.png"
            src.write_bytes(b"original")
            (root / "page_translated.webp").write_bytes(b"output")
            with mock.patch.dict(os.environ, {"RESUME_VERIFY_FINGERPRINT": "true"}):
                self.assertFalse(lb._already_done(src, root, args()))

    def test_fingerprint_invalidates_source_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "page.png"
            src.write_bytes(b"original")
            out = root / "page_translated.webp"
            out.write_bytes(b"rendered")
            a = args()
            with mock.patch.dict(os.environ, {"RESUME_VERIFY_FINGERPRINT": "true", "RENDER_DEBUG": "false"}):
                lb._mark_output_complete(src, out, a)
                self.assertTrue(lb._already_done(src, root, a))
                src.write_bytes(b"modified")
                self.assertFalse(lb._already_done(src, root, a))
                src.write_bytes(b"original")
                self.assertTrue(lb._already_done(src, root, a))
                os.environ["RENDER_DEBUG"] = "true"
                self.assertFalse(lb._already_done(src, root, a))

    def test_zero_byte_output_never_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "page.png"
            src.write_bytes(b"original")
            (root / "page_translated.webp").write_bytes(b"")
            self.assertFalse(lb._already_done(src, root, args()))


if __name__ == "__main__":
    unittest.main()
