import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r2_storage", ROOT / "scripts/r2_storage.py")
r2 = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = r2
spec.loader.exec_module(r2)

class FolderBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.keys = ["raw/test/bolum-2/001.webp", "raw/test/bolum-20/001.webp", "raw/test/bolum-21/001.webp"]
        self.s3 = Mock()
        self.s3.get_paginator.return_value.paginate.side_effect = lambda **kw: [{"Contents": [{"Key": k} for k in self.keys if k.startswith(kw["Prefix"])]}]
        self.config = Mock(bucket="test")

    def test_delete_does_not_include_number_neighbours(self):
        with patch.object(r2, "client", return_value=self.s3):
            self.assertEqual(r2.delete_prefix("raw/test/bolum-2", self.config), 1)
        objects = self.s3.delete_objects.call_args.kwargs["Delete"]["Objects"]
        self.assertEqual(objects, [{"Key": self.keys[0]}])

    def test_empty_delete_prefix_rejected(self):
        with patch.object(r2, "client", return_value=self.s3), self.assertRaises(ValueError):
            r2.delete_prefix("/", self.config)
        self.s3.delete_objects.assert_not_called()

    @unittest.skipUnless(hasattr(r2, "download_prefix"), "translation downloader only")
    def test_download_only_requested_chapter(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(r2, "client", return_value=self.s3):
            files = r2.download_prefix("raw/test/bolum-2", Path(folder), self.config)
        self.assertEqual(len(files), 1)
        self.assertEqual(self.s3.download_file.call_args.args[1], self.keys[0])

if __name__ == "__main__":
    unittest.main()
