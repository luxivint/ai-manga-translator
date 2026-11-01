import hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from modules.detection.reuse import window_observations,merge_observations,page_record,lookup,pixel_hash
from scripts.safe_recut import prepare

class GeometryTests(unittest.TestCase):
    def test_typed_boxes_keep_classes_and_absolute_offset(self):
        raw,obs=window_observations({'bubbles':[[0,50,80,150]],'texts':[[10,80,60,130]]},1000,400,3000,100)
        self.assertEqual(obs[0]['box'],[0,1050,80,1150]);self.assertEqual(obs[1]['kind'],'text')
        self.assertEqual(raw,[[0,50,80,150],[10,80,60,130]])

    def test_cross_window_duplicate_prefers_complete_and_keeps_neighbours(self):
        obs=[{'kind':'text','box':[10,300,80,450],'window':0,'complete':True},
             {'kind':'text','box':[10,350,80,450],'window':350,'complete':False},
             {'kind':'text','box':[12,500,78,550],'window':0,'complete':True}]
        groups=merge_observations(obs);self.assertEqual(len(groups),2);self.assertTrue(all(g['complete'] for g in groups))

    def test_same_window_nested_targets_not_collapsed(self):
        obs=[{'kind':'text','box':b,'window':0,'complete':True} for b in [[0,10,100,100],[5,20,95,90]]]
        self.assertEqual(len(merge_observations(obs)),2)

    def test_unresolved_edge_and_cross_cut_require_fallback(self):
        group={'kind':'text','box':[10,90,60,150],'complete':False}
        record=page_record([group],100,200,100,'x')
        self.assertFalse(record['reusable']);self.assertIn('clipped_scan_window',record['reason']);self.assertIn('region_crosses_cut',record['reason'])

class LookupTests(unittest.TestCase):
    def setUp(self):
        self.path=Path('recut.png');self.im=np.full((300,200,3),123,np.uint8)
        group=[{'kind':'text','box':[20,1040,80,1080],'complete':True}, {'kind':'bubble','box':[10,1020,100,1120],'complete':True}]
        self.record=page_record(group,1000,1300,200,pixel_hash(self.im));self.context={'identity':'same','records':{str(self.path.resolve()):self.record}}

    def test_new_page_offset_and_crop_offset(self):
        boxes,reason=lookup(self.context,self.path,self.im,'same',(0,30,200,280))
        self.assertEqual(reason,'reused');self.assertEqual(boxes[0].tolist(),[[20,10,80,50]]);self.assertEqual(boxes[1].tolist(),[[10,0,100,90]])

    def test_missing_stale_and_wrong_model_never_reused(self):
        self.assertIsNone(lookup(self.context,self.path,self.im,'other')[0])
        changed=self.im.copy();changed[0,0]=0;self.assertEqual(lookup(self.context,self.path,changed,'same')[1],'pixels_changed')
        self.assertIsNone(lookup(self.context,Path('other.png'),self.im,'same')[0])

    def test_crop_splitting_text_is_explicit_fallback(self):
        self.assertEqual(lookup(self.context,self.path,self.im,'same',(0,60,200,280))[1],'crop_splits_text')

    def test_empty_success_is_reused_without_detection(self):
        self.record.update(texts=[],bubbles=[])
        boxes,reason=lookup(self.context,self.path,self.im,'same');self.assertEqual(reason,'reused');self.assertEqual(boxes[0].shape,(0,4))

class RecutIntegrationTests(unittest.TestCase):
    def test_same_source_pixels_and_typed_page_records(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'source.png';Image.new('RGB',(100,4000),'white').save(p)
            calls=[]
            def detect(a):calls.append(len(a));return {'bubbles':[[10,100,90,200]],'texts':[[20,120,80,180]]}
            plan=prepare([p],root/'recut',detect,workers=4)
            self.assertEqual(len(calls),4);self.assertIsNotNone(plan.detections)
            self.assertEqual(len(plan.detections),len(plan.images))
            records=json.loads((root/'recut/recut-detection.json').read_text())
            for path in plan.images:
                im=np.array(Image.open(path).convert('RGB'));record=plan.detections[path.name]
                self.assertEqual(record['pixel_sha256'],pixel_hash(im));self.assertEqual(record,records[path.name])

    def test_untyped_callback_never_claims_complete_cached_detection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'source.png';Image.new('RGB',(100,500),'white').save(p)
            plan=prepare([p],root/'recut',lambda a:[])
            self.assertIsNone(plan.detections)

class PipelineTests(unittest.TestCase):
    def test_cached_page_reaches_ocr_without_detect_request(self):
        import local_batch as lb
        image=np.full((200,200,3),240,np.uint8);path=Path('recut.png')
        record=page_record([{'kind':'text','box':[20,40,100,80],'complete':True}],0,200,200,pixel_hash(image))
        args=SimpleNamespace(source_lang='English',target_lang='Turkish',recut_detection_context={'identity':'same','records':{str(path.resolve()):record}})
        block=SimpleNamespace(text_class='text_free',text='SOURCE TEXT',translation='',xyxy=[20,40,100,80])
        fake=Mock();fake.reuse_identity.return_value='same';fake.detect.side_effect=AssertionError('Second detection forbidden');fake.detect_from_regions.return_value=[block]
        with patch.object(lb,'TextBlockDetector',return_value=fake),patch.object(lb,'LocalSettings'),patch.object(lb,'LocalMain'),patch.object(lb.imk,'read_image',return_value=image),patch.object(lb,'OCRProcessor') as ocr,patch.object(lb,'split_merged_blocks_ocr',side_effect=lambda im,b,o:b),patch.object(lb,'recover_empty_bubble_text_with_gpt'),patch.object(lb,'drop_non_english_ocr_blocks',side_effect=lambda b:b),patch.object(lb,'prune_ocr_blocks',side_effect=lambda b,im:b),patch.object(lb,'strip_translation_garbage_letters'),patch.object(lb,'sort_blk_list',side_effect=lambda b,**k:b):
            result=lb._stage_detect_ocr((0,path,args,None))
        fake.detect.assert_not_called();fake.detect_from_regions.assert_called_once();ocr.return_value.process.assert_called_once();self.assertEqual(result[3],[block])

class ReuseFallbackTests(unittest.TestCase):
    def test_uncertain_cache_only_falls_back_once(self):
        import local_batch as lb
        path=Path('uncertain.png');image=np.zeros((100,100,3),np.uint8)
        record=page_record([{'kind':'text','box':[10,10,50,40],'complete':False}],0,100,100,pixel_hash(image))
        args=SimpleNamespace(recut_detection_context={'identity':'same','records':{str(path.resolve()):record}})
        detector=Mock();detector.reuse_identity.return_value='same';detector.detect.return_value=['fresh']
        with patch.object(lb,'TextBlockDetector',return_value=detector):result=lb.detect_page_blocks(path,image,None,args)
        self.assertEqual(result,['fresh']);detector.detect.assert_called_once();detector.detect_from_regions.assert_not_called()

    def test_banner_crop_reuses_geometry_without_another_detection(self):
        import local_batch as lb
        image=np.full((200,200,3),240,np.uint8);path=Path('banner.png')
        record=page_record([{'kind':'text','box':[20,40,100,80],'complete':True}],0,200,200,pixel_hash(image))
        args=SimpleNamespace(source_lang='English',target_lang='Turkish',recut_detection_context={'identity':'same','records':{str(path.resolve()):record}})
        block=SimpleNamespace(text_class='text_free',text='SOURCE TEXT',translation='',xyxy=[20,40,100,80])
        fake=Mock();fake.reuse_identity.return_value='same';fake.detect.side_effect=AssertionError('Second detection forbidden');fake.detect_from_regions.return_value=[block]
        with patch.object(lb,'TextBlockDetector',return_value=fake),patch.object(lb,'LocalSettings'),patch.object(lb,'LocalMain'),patch.object(lb.imk,'read_image',return_value=image),patch.object(lb,'OCRProcessor'),patch.object(lb,'split_merged_blocks_ocr',side_effect=lambda im,b,o:b),patch.object(lb,'recover_empty_bubble_text_with_gpt'),patch.object(lb,'drop_non_english_ocr_blocks',side_effect=lambda b:b),patch.object(lb,'prune_ocr_blocks',side_effect=lambda b,im:b),patch.object(lb,'strip_translation_garbage_letters'),patch.object(lb,'sort_blk_list',side_effect=lambda b,**k:b):
            result=lb._stage_detect_ocr((0,path,args,(0,0,10,0)))
        fake.detect.assert_not_called();self.assertEqual(fake.detect_from_regions.call_count,2)
        self.assertEqual(fake.detect_from_regions.call_args.args[1].tolist(),[[20,30,100,70]])
        self.assertEqual(result[2].shape[:2],(190,200))

if __name__=='__main__':unittest.main()
