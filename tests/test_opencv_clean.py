import importlib.util,os,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from modules.inpainting.opencv_clean import clean
from modules.inpainting.glyph_mask import refine_free_text
from runtime_config import configure_cleaner

class CleanerTests(unittest.TestCase):
    def test_colour_gradient_and_strict_mask(self):
        yy,xx=np.mgrid[:120,:180];im=np.stack((40+xx*.5,90+yy*.3,160-xx*.2),-1).astype(np.uint8)
        m=np.zeros((120,180),np.uint8);m[40:70,60:110]=255;damaged=im.copy();damaged[m>0]=0
        result=clean(damaged,m);self.assertLess(np.abs(result.astype(float)-im)[m>0].mean(),.5)
        for mode in ['adaptive','telea','ns']:
            a=clean(damaged,m,mode);np.testing.assert_array_equal(a[m==0],im[m==0]);np.testing.assert_array_equal(a,clean(damaged,m,mode))

    def test_reject_removed_algorithms_and_bad_masks(self):
        im=np.zeros((25,30,3),np.uint8);m=np.ones((25,30),np.uint8)
        for mode in ['lama','hybrid','patch','coarse']:
            with self.assertRaises(ValueError):clean(im,m,mode)
        with self.assertRaises(ValueError):clean(im,m[:10])
        np.testing.assert_array_equal(clean(im,np.zeros_like(m)),im)

    def test_no_legacy_motor_or_post_scan_in_runtime(self):
        import local_batch as lb
        self.assertFalse(hasattr(lb,'_get_lama'));self.assertFalse(hasattr(lb,'block_inpaint'));self.assertFalse(hasattr(lb,'clean_postprocess'))
        im=np.full((40,80,3),180,np.uint8);m=np.zeros((40,80),np.uint8);m[15:25,25:50]=255
        with mock.patch.dict(os.environ,{'CLEAN_ENGINE':'cv2','CLEAN_POSTPROCESS':'true'}):
            self.assertEqual(lb.clean_selected(im,m).shape,im.shape)
        self.assertNotIn('modules.inpainting.lama',sys.modules)

class RuntimeTests(unittest.TestCase):
    def test_old_engine_fails_before_claiming_job(self):
        for old in ['lama','hybrid','typo']:
            with mock.patch.dict(os.environ,{'CLEAN_ENGINE':old}):
                with self.assertRaises(ValueError):configure_cleaner()

    def test_shared_pc_container_contract(self):
        with mock.patch.dict(os.environ,{},clear=True):
            state=configure_cleaner();self.assertEqual(state['engine'],'cv2');self.assertFalse(state['postprocess']);self.assertEqual(cv2.getNumThreads(),2)
            self.assertEqual(os.environ['CLEAN_POSTPROCESS'],'false')

class GlyphTests(unittest.TestCase):
    def scene(self,light=False):
        im=np.full((150,420,3),(135,170,160),np.uint8);m=np.zeros((150,420),np.uint8);m[40:100,20:395]=255
        if light:
            cv2.putText(im,'A NEW CHAPTER',(30,84),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,0),thickness=8,lineType=cv2.LINE_AA)
            cv2.putText(im,'A NEW CHAPTER',(30,84),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(250,250,250),thickness=3,lineType=cv2.LINE_AA)
        else:cv2.putText(im,'A NEW CHAPTER',(30,84),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,0),thickness=3,lineType=cv2.LINE_AA)
        block=SimpleNamespace(xyxy=(20,40,395,100),text_class='text_free',translation='Yeni bölüm',text='A NEW CHAPTER',is_ad=False)
        return im,m,block

    def test_free_text_refinement_keeps_mask_inside_original(self):
        for light in [False,True]:
            im,m,b=self.scene(light);new,count=refine_free_text(im,m,[b]);self.assertEqual(count,1);self.assertLess(new.sum(),m.sum());self.assertFalse((new[m==0]>0).any())

    def test_balloons_ads_and_blank_boxes_unchanged(self):
        im,m,b=self.scene();b.text_class='text_bubble';new,n=refine_free_text(im,m,[b]);np.testing.assert_array_equal(new,m);self.assertEqual(n,0)
        b.text_class='text_free';b.is_ad=True;new,n=refine_free_text(im,m,[b]);self.assertEqual(n,0)
        im[:]=170;b.is_ad=False;new,n=refine_free_text(im,m,[b]);np.testing.assert_array_equal(new,m)

    def test_stray_art_mask_removed_but_neighbour_text_preserved(self):
        im,m,b=self.scene(False)
        m[22:32,35:50]=255
        neighbour=SimpleNamespace(xyxy=(320,15,370,35),text_class='text_bubble',translation='Diğer yazı',text='Other',is_ad=False)
        m[18:30,325:365]=255
        refined,count=refine_free_text(im,m,[b,neighbour])
        self.assertEqual(count,1)
        self.assertFalse(refined[22:32,35:50].any())
        np.testing.assert_array_equal(refined[18:30,325:365],m[18:30,325:365])

    def test_no_source_text_no_refinement(self):
        im,m,b=self.scene();b.text=b.translation='';_,n=refine_free_text(im,m,[b]);self.assertEqual(n,0)

if __name__=='__main__':unittest.main()
