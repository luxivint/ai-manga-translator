"""Grouping robustness, narration merge, stray-fragment absorption, duplicate-translation safety net and the replayable
block dump. Real-data fixtures: tests/fixtures/ch29_blocks.json (real dump entries of bolum-29); overlapping / stacked
variants are derived from real boxes and text."""
import contextlib
import io
import json
import os
import sys
import types
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from modules.rendering import balloon_layout as bl  # noqa: E402
from test_balloon_layout import _balloon_page, _block, _settings  # noqa: E402

_FIXTURE = ROOT / "tests" / "fixtures" / "ch29_blocks.json"
if not _FIXTURE.exists():  # local-only fixture (real chapter text is not committed)
    raise unittest.SkipTest(f"missing local fixture {_FIXTURE.name}")
REAL = json.loads(_FIXTURE.read_text(encoding="utf-8"))
OLD_LOG_LINE = ("2026-10-12 21:43:25,140 01 cls=text_bubble bubble=True box=[106, 330, 517, 463] "
                "text='SAMPLE ENGLISH LINE FOR THE PARSER TEST.' trans='ORNEK TURKCE SATIR.'")


class env:
    """Set env vars for a with-block."""
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kw}
        os.environ.update(self.kw)

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class RobustGroupingTests(unittest.TestCase):
    def test_differently_sized_balloon_boxes_of_one_balloon_are_grouped(self):
        # two blocks of one balloon whose detector balloon boxes differ (IoU ~0.5) but each holds the other's text
        bubbles = [(100, 100, 600, 500), (150, 150, 560, 560)]
        boxes = [(200, 150, 500, 220), (220, 380, 480, 450)]
        self.assertEqual(bl.group_by_balloon(bubbles, boxes), [[0, 1]])

    def test_flag_off_restores_strict_grouping(self):
        bubbles = [(100, 100, 600, 500), (150, 150, 560, 560)]
        boxes = [(200, 150, 500, 220), (220, 380, 480, 450)]
        with env(RENDER_GROUP_ROBUST="false"):
            self.assertEqual(bl.group_by_balloon(bubbles, boxes), [])

    def test_text_box_inside_another_text_box_is_one_group_even_without_balloons(self):
        big = REAL[0]["box"] if REAL[0]["cls"] == "text_bubble" else REAL[1]["box"]
        small = (big[0] + 10, big[1] + 5, big[2] - 10, big[1] + 40)
        self.assertEqual([sorted(g) for g in bl.group_by_balloon([None, None], [big, small])], [[0, 1]])

    def test_two_real_balloons_of_one_page_stay_apart(self):
        page = [e for e in REAL if e["page"] == "004.webp"]
        g = bl.group_by_balloon([None] * len(page), [e["box"] for e in page])
        self.assertEqual(g, [])


class NarrationTests(unittest.TestCase):
    BOXES = [(300, 100, 520, 130), (320, 138, 500, 166), (330, 172, 490, 200), (260, 206, 560, 238)]

    def test_stacked_narration_lines_form_one_group(self):
        self.assertEqual(bl.narration_groups(self.BOXES, ["narration"] * 4), [[0, 1, 2, 3]])

    def test_far_apart_or_side_by_side_boxes_do_not_group(self):
        boxes = [(100, 100, 300, 130), (100, 600, 300, 630), (500, 100, 700, 130)]
        self.assertEqual(bl.narration_groups(boxes, ["narration"] * 3), [])

    def test_only_narration_style_is_grouped(self):
        self.assertEqual(bl.narration_groups(self.BOXES, ["normal", "narration", "narration", "sign"]), [[1, 2]])

    def test_narration_texts_are_mergeable_but_not_mixed_with_speech(self):
        texts = ["İLK KEZ,", "KİM HONG-SU", "BANA", "“TEŞEKKÜR EDERİM” DEDİ."]
        self.assertTrue(bl.mergeable(texts, ["narration"] * 4, [8, 6, 6], [24.0, 24.0, 24.0, 24.0]))
        self.assertFalse(bl.mergeable(texts[:2], ["narration", "normal"], [8], [24.0, 24.0]))

    def test_narration_merge_switch(self):
        with env(RENDER_MERGE_NARRATION="false"):
            self.assertFalse(bl.mergeable(["A B", "C D"], ["narration"] * 2, [5], [20.0, 20.0]))


class StrayTests(unittest.TestCase):
    def test_tiny_fragment_in_balloon_is_absorbed(self):
        tr = ["...BÜTÜN BUNLARIN SENİN TALİMATINLA YAPILDIĞINI.", "YÖNÜNÜ."]
        self.assertEqual(bl.stray_fragments(tr, ["...ALL OF THIS WAS DONE", "DIRECTION."], [30.0, 14.0], [True, True]), [1])

    def test_same_size_fragment_is_kept(self):
        tr = ["BÜTÜN BUNLAR SENİN TALİMATINLA OLDU", "YÖNÜNÜ."]
        self.assertEqual(bl.stray_fragments(tr, ["A B C D", "E"], [30.0, 29.0], [True, True]), [])

    def test_fragment_outside_balloon_is_kept(self):
        tr = ["BÜTÜN BUNLAR SENİN TALİMATINLA OLDU", "YÖNÜNÜ."]
        self.assertEqual(bl.stray_fragments(tr, ["A B C D", "E"], [30.0, 12.0], [True, False]), [])

    def test_short_main_text_absorbs_nothing(self):
        self.assertEqual(bl.stray_fragments(["HAYIR.", "NE?"], ["NO", "WHAT"], [30.0, 10.0], [True, True]), [])


class RenderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        cls.align = Qt.AlignmentFlag.AlignCenter

    def _items(self, blocks, orig, cleaned):
        from modules.rendering import render_v2

        return render_v2.build_items(blocks, orig, cleaned, [np.array(b.xyxy) for b in blocks], _settings(), self.align, "tr")

    def test_echo_block_with_substring_translation_is_not_drawn_twice(self):
        rows = [("THE PRESS CONFERENCE WILL BE HELD AT", 250, 0.8), ("THE AIRPORT TOMORROW MORNING", 300, 0.8)]
        orig, cleaned, mask = _balloon_page(rows)
        bub = (50, 100, 650, 600)
        full = "BASIN TOPLANTISI YARIN SABAH HAVALİMANINDA DÜZENLENECEK."
        blocks = [_block((130, 220, 570, 320), "THE PRESS CONFERENCE WILL BE HELD AT THE AIRPORT TOMORROW MORNING", full, bubble=bub),
                  _block((140, 225, 400, 260), "THE PRESS CONFERENCE", "BASIN TOPLANTISI", bubble=bub)]
        items = self._items(blocks, orig, cleaned)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"].replace("\n", " ").count("TOPLANTISI"), 1)

    def test_dedupe_switch_restores_old_behaviour(self):
        rows = [("THE PRESS CONFERENCE WILL BE HELD AT", 250, 0.8)]
        orig, cleaned, mask = _balloon_page(rows)
        bub = (50, 100, 650, 600)
        blocks = [_block((130, 220, 570, 320), "THE PRESS CONFERENCE WILL BE HELD", "BASIN TOPLANTISI YARIN SABAH DÜZENLENECEK.", bubble=bub),
                  _block((140, 225, 400, 260), "THE PRESS CONFERENCE", "BASIN TOPLANTISI", bubble=bub)]
        with env(RENDER_DEDUPE_TRANSLATIONS="false", RENDER_MERGE_BALLOON="false", RENDER_ABSORB_STRAYS="false"):
            items = self._items(blocks, orig, cleaned)
        self.assertEqual(len(items), 2)

    def test_stray_tiny_block_joins_the_balloon_text(self):
        rows = [("ALL OF THIS WAS DONE", 240, 1.0), ("AT YOUR ORDER", 290, 1.0), ("DIRECTION", 420, 0.45)]
        orig, cleaned, mask = _balloon_page(rows)
        bub = (50, 100, 650, 600)
        blocks = [_block((160, 205, 540, 305), "ALL OF THIS WAS DONE AT YOUR ORDER", "...BÜTÜN BUNLARIN SENİN TALİMATINLA YAPILDIĞINI.", bubble=bub),
                  _block((290, 405, 410, 425), "DIRECTION", "YÖNÜNÜ.", bubble=bub)]
        items = self._items(blocks, orig, cleaned)
        self.assertEqual(len(items), 1)
        self.assertIn("YÖNÜNÜ", items[0]["text"])

    def test_four_stacked_narration_blocks_become_one_paragraph(self):
        page = np.full((500, 700, 3), 235, np.uint8)
        orig = page.copy()
        import cv2

        lines = [("FOR THE FIRST TIME,", 100), ("KIM HONG-SU", 140), ("SAID", 180), ("THANK YOU TO ME.", 220)]
        for t, y in lines:
            cv2.putText(orig, t, (200, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (10, 10, 10), 2, cv2.LINE_AA)
        trs = ["İLK KEZ,", "KİM HONG-SU", "BANA", "TEŞEKKÜR EDERİM DEDİ."]
        blocks = []
        for (t, y), tr in zip(lines, trs):
            b = _block((195, y - 26, 480, y + 8), t, tr, free=True)
            b.style = "narration"
            blocks.append(b)
        items = self._items(blocks, orig, page.copy())
        self.assertEqual(len(items), 1)
        self.assertEqual(" ".join(items[0]["text"].split()), "İLK KEZ, KİM HONG-SU BANA TEŞEKKÜR EDERİM DEDİ.")

    def test_no_words_are_lost_by_dedupe_and_merge(self):
        rows = [("ALL OF THIS WAS DONE", 240, 1.0), ("DIRECTION", 420, 0.45)]
        orig, cleaned, mask = _balloon_page(rows)
        bub = (50, 100, 650, 600)
        blocks = [_block((160, 205, 540, 305), "ALL OF THIS WAS DONE AT YOUR ORDER", "...BÜTÜN BUNLARIN SENİN TALİMATINLA YAPILDIĞINI.", bubble=bub),
                  _block((290, 405, 410, 425), "DIRECTION", "YÖNÜNÜ.", bubble=bub)]
        items = self._items(blocks, orig, cleaned)
        drawn = " ".join(d["text"] for d in items).replace("\n", " ").upper()
        for w in ("BÜTÜN", "BUNLARIN", "SENİN", "TALİMATINLA", "YAPILDIĞINI", "YÖNÜNÜ"):
            self.assertIn(w.replace("İ", "İ"), drawn)


class DumpFormatTests(unittest.TestCase):
    def test_old_log_line_still_parses(self):
        from scripts import replay_chapter as rc

        ents = rc.parse_dump_lines(["--- DEBUG BLOCKS after-clean-translate ---", OLD_LOG_LINE])
        self.assertEqual(len(ents), 1)
        e = ents[0]
        self.assertEqual((e["idx"], e["cls"], e["bubble"], e["box"]), (1, "text_bubble", True, [106, 330, 517, 463]))
        self.assertEqual(e["trans"], "ORNEK TURKCE SATIR.")
        self.assertNotIn("page", e)

    def test_new_dump_carries_page_bubble_and_style_and_round_trips(self):
        sys.argv = ["x"]
        from scripts import local_batch as lb
        from scripts import replay_chapter as rc

        b = types.SimpleNamespace(xyxy=np.array([1, 2, 30, 40]), text_class="text_bubble", bubble_xyxy=np.array([0, 0, 90, 99]),
                                  text="HI THERE", translation="MERHABA 'X'", style="shout")
        buf = io.StringIO()
        with env(DEBUG_TRANSLATION_BLOCKS="true"), contextlib.redirect_stdout(buf):
            lb.debug_dump_blocks("after-clean-translate", [b], page="007.webp")
        ents = rc.parse_dump_lines(buf.getvalue().splitlines())
        self.assertEqual(len(ents), 1)
        self.assertEqual((ents[0]["page"], ents[0]["bubble_xyxy"], ents[0]["style"]), ("007.webp", [0, 0, 90, 99], "shout"))
        self.assertEqual(ents[0]["trans"], "MERHABA 'X'")

    def test_dump_is_silent_without_the_debug_flag(self):
        sys.argv = ["x"]
        from scripts import local_batch as lb

        buf = io.StringIO()
        with env(DEBUG_TRANSLATION_BLOCKS="false"), contextlib.redirect_stdout(buf):
            lb.debug_dump_blocks("x", [], page="1.webp")
        self.assertEqual(buf.getvalue(), "")


class PruneIntegrationTests(unittest.TestCase):
    def test_prune_ocr_blocks_drops_contained_echo(self):
        sys.argv = ["x"]
        from scripts import local_batch as lb
        from modules.utils.textblock import TextBlock

        e = next(x for x in REAL if x["page"] == "004.webp" and x["idx"] == 0)
        x1, y1, x2, y2 = e["box"]
        big = TextBlock(text_bbox=np.array(e["box"]), bubble_bbox=np.array([x1 - 20, y1, x2 + 20, y2 + 20]), text_class="text_bubble", text=e["text"])
        echo = TextBlock(text_bbox=np.array([x1 + 20, y1 + 10, x2 - 20, y1 + 50]), bubble_bbox=None, text_class="text_free",
                         text="ONE PIECE OF GOOD NEWS AFTER ANOTHER.")
        with contextlib.redirect_stdout(io.StringIO()):
            kept = lb.prune_ocr_blocks([big, echo])
            kept_off = None
            with env(BLOCK_CONTAINMENT_PRUNE="false"):
                kept_off = lb.prune_ocr_blocks([big, echo])
        self.assertEqual([b.text for b in kept], [e["text"]])
        self.assertEqual(len(kept_off), 2)


if __name__ == "__main__":
    unittest.main()
