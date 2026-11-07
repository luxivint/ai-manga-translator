"""Real-page render checks for joined balloons, airy wrapping and dark captions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from modules.utils.textblock import TextBlock
from modules.rendering import render_v2
from modules.inpainting.white_bubble import clean_white_balloons
from modules.inpainting.flat_caption import clean_flat_dark_captions

FIXTURES = Path(__file__).parent / 'fixtures/layout_review'


def block(box, bubble, source, translation, style='normal'):
    b = TextBlock(text_bbox=np.array(box), text_class='text_bubble' if bubble else 'text_free')
    b.bubble_xyxy = np.array(bubble) if bubble else None
    b.text, b.translation, b.style = source, translation, style
    return b


def render(number, blocks):
    image = np.array(Image.open(FIXTURES / f'page-{number:03d}.png').convert('RGB'))
    cleaned, _ = clean_white_balloons(image, image.copy(), blocks)
    settings = SimpleNamespace(line_spacing=1.12, upper_case=True, font_family='Arial',
                               min_font_size=9, max_font_size=128, direction=Qt.LayoutDirection.LeftToRight)
    return render_v2.build_items(blocks, image, cleaned, [b.xyxy.copy() for b in blocks],
                                 settings, Qt.AlignmentFlag.AlignCenter, 'tr')


class ReportedLayoutRound2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_spiky_balloon_uses_three_rows_instead_of_a_word_column(self):
        items = render(23, [block([193,321,604,506], [48,91,752,738],
            "HE'S A FIRST YEAR LIKE ME...?", 'Benim gibi birinci sınıf mı...?', 'thought')])
        self.assertEqual(len(items), 1)
        self.assertLessEqual(items[0]['text'].count('\n'), 2)
        self.assertGreater(items[0]['width'], items[0]['height'])

    def test_spiky_narration_keeps_a_broad_paragraph(self):
        items = render(85, [block([150,433,655,607], [10,148,800,895],
            'STARTING WITH THE U-17 WORLD CUP, HE BEGAN DRAWING GLOBAL ATTENTION.',
            "17 Yaş Altı Dünya Kupası'yla birlikte dünya çapında ilgi görmeye başladı.", 'narration')])
        self.assertLessEqual(items[0]['text'].count('\n'), 4)
        self.assertGreater(items[0]['width'], items[0]['height'])

    def test_unequal_joined_lobes_keep_title_above_explanation(self):
        bubble = [30,175,645,921]
        items = render(48, [block([381,250,553,395], bubble, 'U-17 WORLD CUP...', 'U-17 Dünya Kupası...'),
            block([85,518,578,708], bubble, 'AN INTERNATIONAL TOURNAMENT HOSTED BY PIFA FOR PLAYERS UNDER 17.',
                  "PIFA'nın 17 yaş altı oyuncular için düzenlediği uluslararası turnuva.", 'narration')])
        self.assertEqual(len(items), 2)
        top, bottom = sorted(items, key=lambda i: i['position'][1])
        self.assertIn('U-17', top['text'])
        self.assertLess(top['position'][1], 400)
        self.assertGreater(bottom['position'][1], 450)
        self.assertLess(top['position'][1]+top['height'], bottom['position'][1])

    def test_wide_joined_caption_is_not_cut_off_by_peak_bisector(self):
        bubble = [48,157,800,889]
        items = render(21, [block([258,255,703,514], bubble,
            'NUMBER 16, SON HYEONG MIN, HEXAGONAL STATS AND KILLER FINISHING...!',
            '16 numara, Son Hyeong Min. Altıgen istatistikler ve ölümcül bitiricilik...!'),
            block([123,520,431,711], bubble, "HE'S THE REAL DEAL, COACH!", 'Gerçekten harika, koç!')])
        self.assertEqual(len(items), 2)
        self.assertGreaterEqual(min(i['font_size'] for i in items), 30)
        self.assertLess(max(i['font_size'] for i in items)/min(i['font_size'] for i in items), 1.3)
        top, bottom = sorted(items, key=lambda i: i['position'][1])
        self.assertLess(top['position'][1]+top['height'], bottom['position'][1])

    def test_black_caption_has_no_source_glyphs_after_cleanup(self):
        original = np.array(Image.open(FIXTURES/'dark-caption.png').convert('RGB'))
        b = block([59,99,645,397], None, 'EVERY MORNING, AND, EVEN AFTER TEAM PRACTICE.',
                  'Her sabah, takım antrenmanından sonra bile.', 'narration')
        cleaned = clean_flat_dark_captions(original, original.copy(), [b])
        self.assertEqual(int(cleaned[99:397,59:645].max()), 0)
        np.testing.assert_array_equal(cleaned[:80], original[:80])

    def test_long_joined_dialogue_keeps_opening_pronoun_with_its_word(self):
        bubble = [40,1278,783,2164]
        upper = block([331,1522,700,1729], bubble,
            'THAT STAMINA AND TENACITY OF YOURS... THAT WAS TALENT, PLAIN AND SIMPLE.',
            'O dayanıklılığın, azmin... Bunlar yetenekti, düpedüz.')
        lower = block([103,1847,534,2011], bubble,
            "IF YOU'D KEPT GOING, YOU COULD'VE MADE IT AS A STARTER IN THE K-LEAGUE.",
            "Devam etseydin K-League'de ilk 11'e girebilirdin.")
        upper.split_from_double_balloon = lower.split_from_double_balloon = True
        items = render(95, [upper, lower])
        top, bottom = sorted(items, key=lambda i: i['position'][1])
        self.assertNotEqual(top['text'].split('\n')[0], 'O')
        self.assertLess(top['position'][1]+top['height'], bottom['position'][1])

    def test_nonuniform_art_or_preserved_graphic_is_not_flattened(self):
        original = np.array(Image.open(FIXTURES/'dark-caption.png').convert('RGB'))
        b = block([59,99,645,397], None, 'CAPTION', 'Yazı', 'narration')
        b.preserve_original = True
        np.testing.assert_array_equal(clean_flat_dark_captions(original, original, [b]), original)
        b.preserve_original = False
        original[90:96,80:160] = [255, 0, 0]
        np.testing.assert_array_equal(clean_flat_dark_captions(original, original, [b]), original)


if __name__ == '__main__':
    unittest.main()
