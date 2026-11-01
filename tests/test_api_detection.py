import json
import os
import unittest
from unittest.mock import Mock,patch
import numpy as np
from modules.detection.api import APIDetection,DetectionAPIError,tile_starts
from modules.detection.factory import DetectionEngineFactory


def response(regions=None,has_text=None,finish='stop'):
    regions=[] if regions is None else regions
    if has_text is None:has_text=any(r['kind']=='text' for r in regions)
    return {'choices':[{'finish_reason':finish,'message':{'content':json.dumps({'has_text':has_text,'regions':regions})}}]}


class APIDetectionTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{'DETECTION_API_KEY':'test','DETECTION_API_BASE_URL':'https://test.invalid/v1','DETECTION_API_COORDINATES':'pixels','DETECTION_API_TEXT_PADDING':'0.05','DETECTION_API_VALIDATION_RETRIES':'1'})
        self.env.start();self.addCleanup(self.env.stop)
        self.engine=APIDetection();self.engine.initialize()

    def test_pixel_coordinates_rescale_and_padding_stays_inside_image(self):
        b,t=self.engine._parse(response([{'kind':'text','box':[0,5,100,50]}]),(100,50),(200,100))
        self.assertEqual(t,[[0,5,200,100]])

    def test_normalized_coordinates_have_explicit_adapter(self):
        self.engine.config.update(coordinates='normalized',padding=0)
        b,t=self.engine._parse(response([{'kind':'text','box':[100,200,500,800]}]),(100,50),(200,100))
        self.assertEqual(t,[[20,20,100,80]])

    def test_text_near_source_edge_keeps_clipped_glyph_fragment(self):
        b,t=self.engine._parse(response([{'kind':'text','box':[200,1450,650,1540]}]),(800,1560),(800,1560))
        self.assertEqual(t[0][3],1560)

    def test_truncated_response_is_not_empty_success(self):
        with self.assertRaises(DetectionAPIError):self.engine._parse(response(finish='length'),(100,50),(100,50))

    def test_invalid_boxes_fail_entire_page_instead_of_dropping_text(self):
        for box in ([20,2,10,5],[-1,0,30,40],[0,0,101,40],[0,0,float('nan'),40],[0,0,True,40]):
            with self.subTest(box=box),self.assertRaises(DetectionAPIError):
                self.engine._parse(response([{'kind':'text','box':box}]),(100,50),(100,50))

    def test_inconsistent_empty_answer_rejected(self):
        with self.assertRaises(DetectionAPIError):self.engine._parse(response([],True),(100,50),(100,50))

    def test_explicit_empty_and_empty_balloon_are_valid(self):
        self.assertEqual(self.engine._parse(response(),(100,50),(100,50)),([],[]))
        b,t=self.engine._parse(response([{'kind':'bubble','box':[1,2,30,40]}],False),(100,50),(100,50))
        self.assertEqual(len(b),1);self.assertFalse(t)

    def test_missing_explicit_empty_indicator_rejected(self):
        body={'choices':[{'finish_reason':'stop','message':{'content':'{"regions":[]}'}}]}
        with self.assertRaises(DetectionAPIError):self.engine._parse(body,(100,50),(100,50))

    def test_tiling_covers_bottom_and_offsets_boxes(self):
        self.assertEqual(tile_starts(4000,1536,307)[-1],2464)
        self.engine.config['padding']=0
        with patch.object(self.engine,'_request_tile',return_value=([],[[1,2,20,30]])) as req:
            b,t=self.engine._detect_single_image(np.zeros((4000,800,3),dtype=np.uint8))
        self.assertEqual(req.call_count,3)
        self.assertEqual(int(t[:,3].max()),2494)
        self.assertEqual(req.call_args.args[0].shape[0],1536)

    def test_no_local_fallback_on_transport_failure(self):
        from modules.utils import gateway
        with patch('modules.detection.api.gateway.resilient_post',side_effect=gateway.GatewayError('offline')):
            with self.assertRaises(gateway.GatewayError):self.engine._request_tile(np.zeros((100,80,3),dtype=np.uint8))

    def test_invalid_answer_is_discarded_and_retry_can_recover(self):
        bad=Mock(status_code=200);bad.json.return_value=response(finish='length')
        good=Mock(status_code=200);good.json.return_value=response()
        with patch('modules.detection.api.gateway.resilient_post',side_effect=[bad,good]) as post,patch('modules.detection.api.gateway.cache_discard') as discard,patch('modules.detection.api.gateway.cache_commit') as commit:
            self.assertEqual(self.engine._request_tile(np.zeros((100,80,3),dtype=np.uint8)),([],[]))
        self.assertEqual(post.call_count,2);discard.assert_called_once();commit.assert_called_once()

    def test_factory_api_does_not_construct_local_detector(self):
        settings=Mock();settings.is_gpu_enabled.return_value=False
        with patch.dict(os.environ,{'DETECTION_BACKEND':'api'}),patch.object(DetectionEngineFactory,'_create_rtdetr_v4s_int8') as local:
            engine=DetectionEngineFactory.create_engine(settings,'RT-DETR-v4-s-int8')
        self.assertIsInstance(engine,APIDetection);local.assert_not_called()

    def test_factory_default_stays_onnx_and_explicit_backend_wins(self):
        with patch.dict(os.environ,{'DETECTION_BACKEND':''}):self.assertEqual(DetectionEngineFactory._resolve_backend(),'onnx')
        with patch.dict(os.environ,{'DETECTION_BACKEND':'api'}):self.assertEqual(DetectionEngineFactory._resolve_backend('onnx'),'onnx')

    def test_invalid_backend_does_not_silently_run_local(self):
        with patch.dict(os.environ,{'DETECTION_BACKEND':'typo'}),self.assertRaises(ValueError):DetectionEngineFactory._resolve_backend()


if __name__=='__main__':unittest.main()
