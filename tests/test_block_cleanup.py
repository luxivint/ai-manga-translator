"""Containment / duplicate pruning of blocks and duplicate-translation clean-up (modules/utils/block_cleanup.py).

Fixtures: tests/fixtures/ch29_blocks.json holds REAL block dumps (box, OCR text, translation) of the job
raw/the-real-tycoon-all-by-myself/bolum-29. Overlapping variants are derived from those real entries (the live chapter
had no overlapping blocks left after the old prune, so the overlap itself is constructed)."""
import json
import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.utils import block_cleanup as bc  # noqa: E402

_FIXTURE = ROOT / "tests" / "fixtures" / "ch29_blocks.json"
if not _FIXTURE.exists():  # local-only fixture (real chapter text is not committed)
    raise unittest.SkipTest(f"missing local fixture {_FIXTURE.name}")
REAL = json.loads(_FIXTURE.read_text(encoding="utf-8"))


def real(page, idx):
    return next(e for e in REAL if e["page"] == page and e["idx"] == idx)


def blk(box, text, trans="", cls="text_bubble", bubble=None):
    return types.SimpleNamespace(xyxy=list(box), text=text, translation=trans, text_class=cls, bubble_xyxy=bubble)


class ContainmentTests(unittest.TestCase):
    def test_line_echo_inside_paragraph_block_is_dropped(self):
        big = real("004.webp", 0)  # DEPARTMENT HEAD CHOI HEE KEEPS DELIVERING ...
        x1, y1, x2, y2 = big["box"]
        echo = blk((x1 + 20, y1 + 10, x2 - 20, y1 + 50), "ONE PIECE OF GOOD NEWS AFTER ANOTHER.")
        kept, dropped = bc.prune_contained_blocks([blk(big["box"], big["text"]), echo])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].text, big["text"])
        self.assertEqual(dropped[0][:2], (1, 0))

    def test_echo_listed_before_the_container_is_still_dropped(self):
        big = real("004.webp", 2)
        x1, y1, x2, y2 = big["box"]
        echo = blk((x1 + 5, y1 + 5, x2 - 5, y1 + 45), "KNOW YOU'RE NOT GOING TO GIVE ME")
        kept, _ = bc.prune_contained_blocks([echo, blk(big["box"], big["text"])])
        self.assertEqual([b.text for b in kept], [big["text"]])

    def test_contained_block_with_different_text_is_kept(self):
        big = real("009.webp", 1)
        x1, y1, x2, y2 = big["box"]
        other = blk((x1 + 10, y1 + 10, x1 + 150, y1 + 40), "MEANWHILE AT THE HARBOUR")
        kept, dropped = bc.prune_contained_blocks([blk(big["box"], big["text"]), other])
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_two_separate_real_balloons_are_both_kept(self):
        a, b = real("037.webp", 0), real("037.webp", 1)  # near-identical sentences, different balloons
        kept, _ = bc.prune_contained_blocks([blk(a["box"], a["text"]), blk(b["box"], b["text"])])
        self.assertEqual(len(kept), 2)

    def test_partly_overlapping_boxes_below_ratio_are_kept(self):
        a = blk((0, 0, 100, 100), "WH... WHAT?! W-W-WHO LIKES WHOM?!")
        b = blk((60, 0, 160, 100), "WH... WHAT?!")  # only 40 percent inside
        kept, _ = bc.prune_contained_blocks([a, b])
        self.assertEqual(len(kept), 2)

    def test_ratio_is_configurable(self):
        a = blk((0, 0, 100, 100), "WH... WHAT?! W-W-WHO LIKES WHOM?!")
        b = blk((60, 0, 160, 100), "WH... WHAT?!")
        os.environ["BLOCK_CONTAINMENT_RATIO"] = "0.3"
        try:
            kept, _ = bc.prune_contained_blocks([a, b])
        finally:
            del os.environ["BLOCK_CONTAINMENT_RATIO"]
        self.assertEqual(len(kept), 1)

    def test_every_dropped_word_survives_in_the_container(self):
        big = real("041.webp", 1)  # WH... WHAT?! W-W-WHO LIKES WHOM?!
        x1, y1, x2, y2 = big["box"]
        echo = blk((x1 + 10, y1 + 5, x2 - 10, y1 + 40), "WH... WHAT?!")
        kept, dropped = bc.prune_contained_blocks([blk(big["box"], big["text"]), echo])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(bc.covered_fraction(echo.text, kept[0].text), 1.0)

    def test_empty_text_blocks_are_ignored(self):
        kept, dropped = bc.prune_contained_blocks([blk((0, 0, 100, 100), "HELLO THERE FRIEND"), blk((10, 10, 50, 50), "")])
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_chain_collapses_onto_largest(self):
        t = "KIM HONG-SU IS MUCH MORE GENEROUS THAN I THOUGHT"
        kept, _ = bc.prune_contained_blocks([
            blk((10, 10, 90, 40), "KIM HONG-SU IS MUCH"), blk((0, 0, 100, 100), t), blk((5, 5, 95, 60), "KIM HONG-SU IS MUCH MORE GENEROUS")])
        self.assertEqual([b.text for b in kept], [t])


class TranslationDedupeTests(unittest.TestCase):
    def test_substring_translation_is_dropped(self):
        big = real("004.webp", 1)["trans"]
        blocks = [blk((0, 0, 400, 130), "", big), blk((20, 5, 380, 40), "", "BU SEFER BEN BİLE ŞAŞIRDIM.")]
        drop, _ = bc.drop_duplicate_translations(blocks, [b.xyxy for b in blocks], [[0, 1]])
        self.assertEqual(drop, [1])

    def test_distinct_translations_both_stay(self):
        blocks = [blk((0, 0, 400, 130), "", "İLK KEZ KİM HONG-SU BANA TEŞEKKÜR ETTİ."), blk((20, 5, 380, 40), "", "ÇOK ÇALIŞMAMIZ GEREKİYOR.")]
        drop, _ = bc.drop_duplicate_translations(blocks, [b.xyxy for b in blocks], [[0, 1]])
        self.assertEqual(drop, [])

    def test_repeated_phrase_is_collapsed(self):
        self.assertEqual(bc.collapse_repeats("KARŞILIK OLARAK KARŞILIK OLARAK KARŞILIK OLARAK SANA"), "KARŞILIK OLARAK SANA")

    def test_single_repeated_word_is_kept(self):
        self.assertEqual(bc.collapse_repeats("ÇOK ÇOK GÜZEL"), "ÇOK ÇOK GÜZEL")

    def test_text_without_repeat_is_unchanged(self):
        t = real("072.webp", 1)["trans"]
        self.assertEqual(bc.collapse_repeats(t), t)

    def test_near_duplicate_translation_dropped_keeps_longer(self):
        a = blk((0, 0, 400, 130), "", "EŞİ BENZERİ GÖRÜLMEMİŞ BİR OLAY")
        b = blk((10, 10, 300, 60), "", "Eşi benzeri görülmemiş")
        drop, _ = bc.drop_duplicate_translations([a, b], [a.xyxy, b.xyxy], [[0, 1]])
        self.assertEqual(drop, [1])

    def test_overlap_groups_pairs_only_overlapping_boxes(self):
        groups = bc.overlap_groups([(0, 0, 100, 100), (10, 10, 60, 60), (500, 500, 600, 600)])
        self.assertEqual(groups, [[0, 1]])


if __name__ == "__main__":
    unittest.main()
