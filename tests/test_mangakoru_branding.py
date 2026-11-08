import argparse
import os
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import local_batch as batch
import worker
from modules.utils.watermark import match_watermark


class MangaKoruBrandingTests(unittest.TestCase):
    def test_image_alt_reads_settings_after_module_import_and_keeps_chapter_tens(self):
        with mock.patch.dict(os.environ, {'SEO_SITE_NAME':'MangaKoru','SEO_SITE_URL':'https://mangakoru.lol',
             'SEO_TITLE_TEMPLATE':'{site_name} {slug} bölüm {number} sayfa {page} {site_url}'}):
            for number, expected in [(10,'10'), (100,'100'), (Decimal('10.00'),'10'), (Decimal('10.50'),'10.5')]:
                self.assertEqual(worker.seo_alt('test-seri',number,2),f'MangaKoru test-seri bölüm {expected} sayfa 2 https://mangakoru.lol')

    def test_default_upload_filename_uses_new_brand(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(batch.seo_upload_name('translated/test-seri/bolum-10',2,'.WEBP'),
                             'mangakoru-turkce-manga-oku-test-seri-bolum-10-sayfa-2.webp')

    def test_current_and_legacy_own_brands_survive_cleanup(self):
        for text in ('MangaKoru','mangakoru.lol','https://mangakoru.lol','Manga Koru .lol','TrendManga.com'):
            self.assertIsNone(match_watermark(text),text)
        self.assertIsNotNone(match_watermark('https://otherscans.com'))

    def test_approved_asset_has_alpha_and_composites_without_background_plate(self):
        stamp_path=ROOT/'assets/mangakoru-stamp.png'
        with Image.open(stamp_path) as stamp:
            self.assertIn('A',stamp.getbands())
            self.assertEqual(stamp.getchannel('A').getextrema(),(0,255))
            self.assertEqual(stamp.getpixel((0,0))[3],0)
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'5.png'
            Image.new('RGB',(800,1200),'white').save(output)
            args=argparse.Namespace(stamp_enabled='true',stamp_every_pages=5,stamp_path=str(stamp_path),
                stamp_width_ratio=.2,stamp_opacity=1.,stamp_black_background='false',output_format='png')
            self.assertTrue(batch.apply_page_stamp(output,Path('5.png'),args))
            with Image.open(output) as result:
                self.assertEqual(result.size,(800,1200))
                self.assertEqual(result.convert('RGB').getpixel((799,1199)),(255,255,255))

if __name__=='__main__':unittest.main()
