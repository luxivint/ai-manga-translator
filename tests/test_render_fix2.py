"""Tests for the render_fix2 work: shared batch/process translation prompt, line-count enforcement,
merged-block groups, original-mask colour analysis, slanted-speech font choice."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.rendering import font_profiles as fp  # noqa: E402
from modules.rendering import orig_mask  # noqa: E402
from modules.rendering import text_style as tst  # noqa: E402
from modules.translation.base import LLMTranslation  # noqa: E402
from modules.translation.llm.base import BaseLLMTranslation  # noqa: E402
from modules.utils.merged_blocks import split_merged_ocr_blocks  # noqa: E402
from modules.utils.textblock import TextBlock  # noqa: E402
from modules.utils.translator_utils import (  # noqa: E402
    apply_translation_json, parse_translation_value, redistribute_lines, translation_temperature)


def _page(w=500, h=400, bg=(200, 200, 200)):
    return np.full((h, w, 3), bg, np.uint8)


def _put_rows(img, rows, x=40, y0=30, pitch=50, scale=1.6, color=(0, 0, 0)):
    for i, t in enumerate(rows):
        cv2.putText(img, t, (x, y0 + 30 + i * pitch), cv2.FONT_HERSHEY_DUPLEX, scale, color, 3, cv2.LINE_AA)
    return img


class SharedPromptTests(unittest.TestCase):
    class Fake(BaseLLMTranslation):
        supports_multi_image = True

        def __init__(self, replies):
            super().__init__()
            self.source_lang, self.target_lang = "English", "Turkish"
            self.max_tokens, self.replies, self.calls = 5000, list(replies), []

        def _perform_translation(self, user_prompt, system_prompt, image):
            self.calls.append((user_prompt, system_prompt, image))
            return self.replies.pop(0)

    def setUp(self):
        os.environ["RENDER_STYLE_TAGS"] = "true"

    def test_batch_prompt_is_the_process_prompt_plus_ad_rule(self):
        e = self.Fake([])
        base = e.get_system_prompt("English", "Turkish")
        batch = e.get_system_prompt("English", "Turkish", ad_detection=True, own_brand="TrendManga")
        self.assertTrue(batch.startswith(base.rstrip()[:200]))
        self.assertIn("BLOCK RULES", batch)
        self.assertIn("STYLE TAGS", batch)
        self.assertIn("[[AD]]", batch)
        self.assertNotIn("[[AD]]", base)

    def test_translate_keyed_uses_same_temperature_and_parses_styles(self):
        e = self.Fake(['{"p0_b0": {"text": "MERHABA\\nDUNYA", "style": "shout"}, "p0_b1": "x"}'])
        e.temperature = translation_temperature()
        out = e.translate_keyed({"p0_b0": "HELLO WORLD", "p0_b1": "x"}, images=[np.zeros((4, 4, 3), np.uint8)])
        self.assertEqual(parse_translation_value(out["p0_b0"]), ("MERHABA\nDUNYA", "shout"))
        self.assertEqual(parse_translation_value(out["p0_b1"]), ("x", ""))
        self.assertAlmostEqual(e.temperature, 0.3)
        self.assertEqual(e.max_tokens, 5000)  # restored after the call

    def test_apply_json_sets_style_and_newlines(self):
        b = TextBlock(text_bbox=np.array([0, 0, 10, 10]), text="A B")
        apply_translation_json({"k": b}, '{"k": {"text": "BIR\\nIKI", "style": "phone"}}')
        self.assertEqual(b.translation, "BIR\nIKI")
        self.assertEqual(b.style, "phone")

    def test_redistribute_lines(self):
        self.assertEqual(len(redistribute_lines(["a b", "c d", "e f", "g h"], 2)), 2)
        self.assertEqual(redistribute_lines(["alpha beta gamma delta"], 2), ["alpha beta", "gamma delta"])
        self.assertEqual(redistribute_lines(["x", "y"], 5), ["x", "y"])


class LineCountTests(unittest.TestCase):
    class Fake(SharedPromptTests.Fake):
        pass

    def _list_block(self):
        img = _put_rows(_page(), ["LEVEL - 1", "HP - 71", "MP - 0", "STATS - 0"], pitch=50)
        b = TextBlock(text_bbox=np.array([20, 20, 480, 260]), text="LEVEL - 1 HP - 71 MP - 0 STATS - 0")
        b.text_class, b.style = "text_free", "phone"
        return img, b

    def test_source_row_count_and_row_split(self):
        img, b = self._list_block()
        self.assertEqual(tst.source_line_count(img, b.xyxy), 4)
        self.assertEqual(tst.source_rows(img, b.xyxy, b.text), ["LEVEL - 1", "HP - 71", "MP - 0", "STATS - 0"])

    def test_wrong_line_count_is_retried_then_correct(self):
        img, b = self._list_block()
        b.translation = "SEVIYE - 1\nCAN - 71\nMP - 0\nOZELLIK\n0"  # 5 lines, image shows 4
        e = self.Fake(['{"block_0": {"text": "SEVIYE - 1\\nCAN - 71\\nMP - 0\\nOZELLIK - 0", "style": "phone"}}'])
        self.assertEqual(e.fix_line_counts([b], img), 1)
        self.assertEqual(b.translation.count("\n"), 3)
        self.assertEqual(len(e.calls), 1)

    def test_failed_retry_falls_back_to_exact_line_count(self):
        img, b = self._list_block()
        b.translation = "A B\nC D E F"

        class Boom(self.Fake):
            def _perform_translation(self, *a):
                raise RuntimeError("net down")

        e = Boom([])
        e.fix_line_counts([b], img)
        self.assertEqual(len([ln for ln in b.translation.split("\n") if ln]), 4)

    def test_matching_count_makes_no_call(self):
        img, b = self._list_block()
        b.translation = "1\n2\n3\n4"
        e = self.Fake([])
        self.assertEqual(e.fix_line_counts([b], img), 0)
        self.assertEqual(e.calls, [])

    def test_speech_bubbles_are_never_forced(self):
        img, b = self._list_block()
        b.text_class, b.translation = "text_bubble", "tek satir"
        self.assertEqual(BaseLLMTranslation.expected_lines(b, img), 0)

    def test_row_hint_lists_rows(self):
        img, b = self._list_block()
        hint = BaseLLMTranslation.row_hint_context({"block_0": b}, img)
        self.assertIn("4 text rows", hint)
        self.assertIn("LEVEL - 1 | HP - 71", hint)


class MergedGroupsTests(unittest.TestCase):
    def _board(self):
        img = _page(700, 520, (235, 240, 245))
        img[150:230] = (170, 70, 240)  # purple menu bar
        cv2.putText(img, "BIG TITLE", (30, 70), cv2.FONT_HERSHEY_DUPLEX, 2.4, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, "MENU ITEMS HERE", (30, 205), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "| LIVE BOARD", (30, 290), cv2.FONT_HERSHEY_DUPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, "ANYONE KNOW THIS", (30, 390), cv2.FONT_HERSHEY_DUPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, "WILD NAME", (30, 480), cv2.FONT_HERSHEY_DUPLEX, 1.0, (110, 110, 110), 2, cv2.LINE_AA)
        return img

    def test_groups_found_on_original_with_coloured_bar(self):
        groups = tst.text_groups(self._board(), (0, 0, 700, 520))
        self.assertEqual(len(groups), 5)
        ys = [g["box"][1] for g in groups]
        self.assertEqual(ys, sorted(ys))

    def test_paragraph_or_list_with_regular_pitch_is_not_merged(self):
        img = _put_rows(_page(500, 400), ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"], pitch=62, scale=1.2)
        self.assertFalse(tst.looks_merged(tst.text_groups(img, (0, 0, 500, 400))))

    def test_board_with_varied_rows_is_merged(self):
        self.assertTrue(tst.looks_merged(tst.text_groups(self._board(), (0, 0, 700, 520))))

    def test_ocr_echo_keeps_block_whole(self):
        img = self._board()
        blk = TextBlock(text_bbox=np.array([0, 0, 700, 520]), text="x y z")

        class Echo:
            def process(self, image, blocks):
                for b in blocks:
                    b.text = "DENSE FOREST"
                return blocks

        self.assertEqual(len(split_merged_ocr_blocks([blk], img, Echo())), 1)

    def test_merged_block_is_split_and_reread_per_group(self):
        img = self._board()
        blk = TextBlock(text_bbox=np.array([0, 0, 700, 520]), text="BIG TITLE MENU ITEMS HERE LIVE BOARD ANYONE KNOW THIS WILD NAME")
        blk.text_class = "text_bubble"

        class Ocr:
            def process(self, image, blocks):
                for i, b in enumerate(blocks):
                    b.text = f"group{i}"
                return blocks

        out = split_merged_ocr_blocks([blk], img, Ocr())
        self.assertEqual([b.text for b in out], [f"group{i}" for i in range(5)])
        self.assertTrue(all(b.text_class == "text_free" for b in out))

    def test_skip_predicate_keeps_block(self):
        img = self._board()
        blk = TextBlock(text_bbox=np.array([0, 0, 700, 520]), text="x")
        out = split_merged_ocr_blocks([blk], img, object(), skip=lambda b: True)
        self.assertEqual(len(out), 1)


class ColourTests(unittest.TestCase):
    def test_outlined_gradient_text_uses_fill_not_outline(self):
        # orange->red vertical gradient letters with a thick cream outline on a pink backdrop;
        # the mask is the outline ring only (what a cleaner removes), the fill sits in its holes
        h, w = 140, 300
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (205, 60, 110)  # pink backdrop
        glyph = np.zeros((h, w), np.uint8)
        cv2.putText(glyph, "HEP", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 3.4, 255, 18, cv2.LINE_AA)
        outline = cv2.dilate(glyph, np.ones((9, 9), np.uint8))
        img[outline > 0] = (255, 249, 247)
        for y in range(h):
            img[y][glyph[y] > 0] = (225, 120 - int(y * 0.6), 20)
        ring = (outline > 0) & (glyph == 0)
        res = tst.analyze_colors(img, ring, (0, 0), bg_rgb=(205, 60, 110))
        fill = np.array(res["fill"])
        self.assertGreater(fill[0], 200)
        self.assertLess(fill[2], 120)  # orange/red, not cream
        self.assertIsNotNone(res["outline"])
        self.assertGreater(min(res["outline"]), 200)

    def test_gradient_fit_ignores_outliers(self):
        n = 400
        coord = np.linspace(0, 1, n)
        colors = np.stack([200 + 0 * coord, 150 - 120 * coord, 20 + 0 * coord], axis=1)
        colors[::40] = (0, 255, 0)  # specks
        ends, r2 = tst._fit_linear(coord, colors)
        self.assertAlmostEqual(ends[0][1], 150, delta=12)
        self.assertAlmostEqual(ends[1][1], 30, delta=12)

    def test_orig_mask_registry_follows_array_lifetime(self):
        a = np.zeros((10, 10, 3), np.uint8)
        orig_mask.remember(a, np.ones((10, 10), np.uint8) * 255)
        self.assertIsNotNone(orig_mask.lookup(a))
        self.assertIsNone(orig_mask.lookup(np.zeros((10, 10, 3), np.uint8)))
        self.assertIsNotNone(orig_mask.glyph_mask(a, (0, 0, 10, 10)))


class SlantFontTests(unittest.TestCase):
    def test_slanted_normal_speech_is_not_a_heavy_shout_font(self):
        src = {"stem": 0.11, "slant": 0.22, "wid": 0.6, "height": 40.0, "caps": True}
        rel = fp.match_font("normal", src)
        self.assertLess(fp.traits_of(rel)["stem"], 0.17)
        self.assertNotIn("CCShoutOut", rel)
        self.assertTrue(fp.wants_synthetic_italic("normal", src, rel))

    def test_upright_or_shout_gets_no_synthetic_italic(self):
        up = {"stem": 0.11, "slant": 0.02, "wid": 0.6, "height": 40.0}
        sl = {"stem": 0.11, "slant": 0.22, "wid": 0.6, "height": 40.0}
        self.assertFalse(fp.wants_synthetic_italic("normal", up))
        self.assertFalse(fp.wants_synthetic_italic("shout", sl))
        self.assertFalse(fp.wants_synthetic_italic("sfx", sl))

    def test_slanted_shout_still_gets_slanted_font(self):
        sl = {"stem": 0.2, "slant": 0.25, "wid": 0.6, "height": 40.0, "caps": True}
        self.assertGreater(fp.traits_of(fp.match_font("shout", sl)).get("slant", 0), 0.1)


class SpeechPoolTests(unittest.TestCase):
    K = "FONT KLASÖR/"

    def test_pool_is_konusma_folder_without_thin_fonts(self):
        pool = fp.speech_pool()
        if not (ROOT / "FONT KLASÖR" / "konuşma").is_dir():
            self.skipTest("font folder not present (gitignored)")
        self.assertTrue(pool)
        self.assertTrue(all(r.startswith(fp.SPEECH_FOLDER) for r in pool))
        self.assertIn(self.K + "konuşma/mercyorj.otf", pool)
        self.assertEqual(fp.ranked_candidates("normal"), pool)
        for r in pool:
            self.assertGreaterEqual(fp.traits_of(r)["stem"], fp.SPEECH_MIN_STEM)
        self.assertNotIn(self.K + "Kare/Chinacat.ttf", pool)

    def test_thin_font_is_rejected_by_threshold(self):
        if not (ROOT / "FONT KLASÖR" / "konuşma").is_dir():
            self.skipTest("font folder not present (gitignored)")
        os.environ["FONT_SPEECH_MIN_STEM"] = "0.12"
        try:
            pool = fp.speech_pool()
        finally:
            del os.environ["FONT_SPEECH_MIN_STEM"]
        self.assertNotIn(self.K + "konuşma/CCVictorySpeech.otf", pool)  # stem 0.103
        self.assertIn(self.K + "konuşma/mercyorj.otf", pool)

    def test_neighbouring_bubbles_share_one_font(self):
        if not fp.speech_pool():
            self.skipTest("font folder not present (gitignored)")
        picks = {fp.match_font("normal", {"stem": s, "slant": 0.0, "wid": w, "height": 40.0})
                 for s in (0.06, 0.08, 0.11, 0.14, 0.18) for w in (0.55, 0.65, 0.7, 0.9)}
        self.assertEqual(len(picks), 1)
        self.assertEqual(picks.pop(), fp.speech_pool()[0])

    def test_clearly_different_traits_may_switch_font(self):
        pool = fp.speech_pool()
        if len(pool) < 2:
            self.skipTest("font folder not present (gitignored)")
        wide = {"stem": 0.11, "slant": 0.0, "wid": 1.0, "height": 40.0}
        self.assertIn(fp.match_font("normal", wide), pool)
        self.assertNotEqual(fp.match_font("normal", wide), pool[0])

    def test_fresh_clone_without_fonts_falls_back_without_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_root = fp.ROOT
            fp.ROOT = Path(tmp)
            try:
                self.assertEqual(fp.speech_pool(), [])
                # old list (by committed traits) is used; the renderer then falls back to Comic Geek when a file
                # is missing. Nothing may crash.
                rel = fp.match_font("normal", {"stem": 0.1, "slant": 0, "wid": 0.7, "height": 40.0})
                self.assertIsInstance(rel, str)
            finally:
                fp.ROOT = old_root

    def test_env_font_map_file_overrides_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "map.json"
            f.write_text('{"normal": ["FONT KLASÖR/Kare/CC_VictorySpeechLower.otf"]}', encoding="utf-8")
            os.environ["FONT_MAP_FILE"] = str(f)
            try:
                self.assertEqual(fp.ranked_candidates("normal"), [self.K + "Kare/CC_VictorySpeechLower.otf"])
                self.assertTrue(fp.ranked_candidates("whisper"))  # other styles untouched
            finally:
                del os.environ["FONT_MAP_FILE"]
        os.environ["FONT_MAP_FILE"] = str(ROOT / "does_not_exist.json")
        try:
            self.assertEqual(fp.ranked_candidates("normal"), fp.speech_pool() or fp.ranked_candidates("normal"))
        finally:
            del os.environ["FONT_MAP_FILE"]


if __name__ == "__main__":
    unittest.main()
