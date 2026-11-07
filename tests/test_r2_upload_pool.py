"""Upload parallelism must preserve page order and withhold failed manifests."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from scripts import r2_storage as storage


class UploadPoolTests(unittest.TestCase):
    def test_one_pool_bounded_parallelism_and_ordered_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(8):
                (root / f'{i:03d}.png').write_bytes(b'image')
            active = peak = 0
            lock = threading.Lock()
            def upload(path, bucket, key, **kwargs):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(.01 * (8 - int(Path(key).stem)))
                with lock:
                    active -= 1
            s3 = Mock()
            s3.upload_file.side_effect = upload
            config = Mock(bucket='test', public_base_url='https://test.invalid')
            with patch.object(storage, 'client', return_value=s3) as client, \
                 patch.dict(os.environ, {'R2_UPLOAD_WORKERS': '2'}):
                manifest = storage.upload_directory(root, 'translated/manga/chapter', config)
            client.assert_called_once_with(config)
            self.assertEqual(peak, 2)
            self.assertEqual([Path(item['key']).name for item in manifest['items']],
                             [f'{i:03d}.png' for i in range(8)])
            payload = json.loads(s3.put_object.call_args.kwargs['Body'])
            self.assertEqual(payload['count'], 8)

    def test_failed_image_never_publishes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '001.png').write_bytes(b'image')
            s3 = Mock()
            s3.upload_file.side_effect = OSError('temporary upload failure')
            with patch.object(storage, 'client', return_value=s3), \
                 patch.dict(os.environ, {'R2_UPLOAD_WORKERS': '2'}):
                with self.assertRaises(OSError):
                    storage.upload_directory(root, 'translated/manga/chapter', Mock(bucket='test'))
            s3.put_object.assert_not_called()
            s3.delete_objects.assert_not_called()


if __name__ == '__main__':
    unittest.main()
