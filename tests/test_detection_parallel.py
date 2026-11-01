import json,os,sys,tempfile,threading,time,unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts.safe_recut import prepare,RecutError
from modules.utils import gateway
from modules.detection.api import APIDetection

class RecutParallelTests(unittest.TestCase):
    def source(self,folder):
        p=folder/'source.png';a=np.full((16000,40,3),255,np.uint8)
        for y in range(16000):a[y,0]=[y%251,(y//251)%251,0]
        Image.fromarray(a).save(p);return p

    def test_parallel_is_bounded_and_matches_serial_pixels_cuts_and_regions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.source(root);state={'active':0,'peak':0};lock=threading.Lock()
            def detect(a):
                with lock:state['active']+=1;state['peak']=max(state['peak'],state['active'])
                try:
                    time.sleep(.015+(int(a[0,0,0])%3)*.002)
                    return [[5,100,30,200]]
                finally:
                    with lock:state['active']-=1
            serial=prepare([p],root/'serial',detect,workers=1)
            state['peak']=0;parallel=prepare([p],root/'parallel',detect,workers=4)
            self.assertGreater(state['peak'],1);self.assertLessEqual(state['peak'],4)
            self.assertEqual(serial.cuts,parallel.cuts);self.assertEqual(serial.digest,parallel.digest)
            s=json.loads((root/'serial/recut-manifest.json').read_text());q=json.loads((root/'parallel/recut-manifest.json').read_text())
            for key in ['protected_regions','source_sha256','cuts','pixel_exact']:self.assertEqual(s[key],q[key])
            for a,b in zip(serial.images,parallel.images):self.assertEqual(Image.open(a).tobytes(),Image.open(b).tobytes())

    def test_failed_request_does_not_create_or_publish_cut_pages(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.source(root)
            def fail(a):raise RuntimeError('API unavailable')
            with self.assertRaisesRegex(RuntimeError,'API unavailable'):prepare([p],root/'bad',fail,workers=4)
            self.assertFalse(list((root/'bad').glob('*.png')));self.assertTrue(p.exists())

    def test_bad_concurrency_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.source(root)
            for n in [0,9]:
                with self.assertRaises(RecutError):prepare([p],root/str(n),lambda a:[],workers=n)

class DetectionLimiterTests(unittest.TestCase):
    def tearDown(self):gateway.reset_process_state()
    def test_detection_does_not_share_ocr_single_slot(self):
        with patch.dict(os.environ,{'DETECTION_API_CONCURRENCY':'4','OCR_CONCURRENCY':'1'}):
            self.assertEqual(gateway.detection_limiter().max_n,4);self.assertEqual(gateway.ocr_limiter().max_n,1)
            self.assertIsNot(gateway.detection_limiter(),gateway.ocr_limiter())

    def test_limit_is_process_wide_and_resets(self):
        with patch.dict(os.environ,{'DETECTION_API_CONCURRENCY':'3'}):
            first=gateway.detection_limiter();self.assertIs(first,gateway.detection_limiter())
            gateway.reset_process_state();self.assertIsNot(first,gateway.detection_limiter())

    def test_global_limit_handles_nested_senders(self):
        state={'active':0,'peak':0};lock=threading.Lock()
        with patch.dict(os.environ,{'DETECTION_API_CONCURRENCY':'4'}):
            limiter=gateway.detection_limiter()
            def request(i):
                with limiter.slot():
                    with lock:state['active']+=1;state['peak']=max(state['peak'],state['active'])
                    time.sleep(.015)
                    with lock:state['active']-=1
            with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(request,range(24)))
            self.assertEqual(state['peak'],4)

class TileTests(unittest.TestCase):
    def test_parallel_tiles_have_original_offsets_and_same_order(self):
        engine=APIDetection()
        with patch.dict(os.environ,{'DETECTION_API_KEY':'test','DETECTION_API_BASE_URL':'https://test.invalid/v1','DETECTION_API_CONCURRENCY':'4'}):
            engine.initialize();im=np.zeros((3800,800,3),np.uint8);im[:,:,0]=np.arange(3800,dtype=np.uint16)[:,None]%251
            state={'active':0,'peak':0};lock=threading.Lock()
            def fake(tile):
                with lock:state['active']+=1;state['peak']=max(state['peak'],state['active'])
                time.sleep(.02)
                with lock:state['active']-=1
                return [],[[10,20,40,50]]
            with patch.object(engine,'_request_tile',side_effect=fake):b,t=engine._detect_single_image(im)
            self.assertGreater(state['peak'],1);self.assertLessEqual(state['peak'],4)
            self.assertEqual(t.tolist(),[[10,20,40,50],[10,1249,40,1279],[10,2284,40,2314]])

if __name__=='__main__':unittest.main()
