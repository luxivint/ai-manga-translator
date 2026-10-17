import os
import unittest

from modules.utils import block_cleanup as bcl
from modules.rendering import typeset as ts
from scripts import replay_chapter as rc


class CoverageMisses(unittest.TestCase):
    def src(self, text, tr, box):
        return dict(text=text, translation=tr, box=box)

    def test_rendered_block_is_covered(self):
        s = [self.src("HELLO THERE", "Merhaba orada", [0, 0, 100, 50])]
        self.assertEqual(bcl.coverage_misses(s, [dict(text="Merhaba orada", box=[0, 0, 100, 50])]), [])

    def test_unrendered_block_is_missed(self):
        s = [self.src("HELLO THERE", "Merhaba orada", [0, 0, 100, 50]),
             self.src("SECOND LINE", "Ikinci satir burada", [0, 900, 100, 950])]
        items = [dict(text="Merhaba orada", box=[0, 0, 100, 50])]
        self.assertEqual(bcl.coverage_misses(s, items), [1])

    def test_folded_duplicate_is_covered_by_text(self):
        s = [self.src("A B C", "bir iki uc dort", [0, 0, 100, 50]),
             self.src("A B", "bir iki", [10, 60, 90, 90])]
        items = [dict(text="bir iki uc dort bes", box=[0, 0, 100, 50])]
        self.assertEqual(bcl.coverage_misses(s, items), [])

    def test_deliberate_skip_and_no_translation_are_not_missed(self):
        s = [self.src("LOGO", "LOGO", [0, 0, 50, 20]), self.src("TEXT HERE", "", [0, 200, 50, 220])]
        self.assertEqual(bcl.coverage_misses(s, [], deliberate=[[0, 0, 60, 30]]), [])


class SingleWord(unittest.TestCase):
    def test_single_word_not_hyphenated_when_it_fits_smaller(self):
        def measure(size):
            return (lambda t: 0.6 * size * len(t)), 1.2 * size
        out = ts.fit_text("MERHABA.", measure, 400, 300, max_size=80, min_size=8)
        self.assertNotIn("-", out[0])


class Crops(unittest.TestCase):
    def test_parse_and_rows(self):
        lines = ["  [vision-crop] [1] recut-00001.png: üst%=42 alt%=0 kırpıldı",
                 "  [banner-crop] [1] recut-00001.png: üst=1522px alt=0px",
                 "  [vision-crop] [2] recut-00002.png: üst%=2 alt%=0 REDDEDİLDİ (x)"]
        c = rc.parse_crops(lines)
        self.assertEqual(rc.crop_rows(4049, c["recut-00001.png"]), (1700, 4049))
        self.assertEqual(rc.crop_rows(1000, c["recut-00002.png"]), (0, 1000))


if __name__ == "__main__":
    unittest.main()
