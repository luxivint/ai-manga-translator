"""Translation misassignment regression + consistency guard (see modules/translation/consistency.py).

Root cause covered here: a batch answer that misses a key is a sign the model shifted/merged blocks; the old retry
kept the other (misaligned) keys and only re-asked for the missing one."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import local_batch as lb  # noqa: E402
from modules.translation import consistency as cc  # noqa: E402


def blk(text, translation="", is_ad=False):
    return SimpleNamespace(text=text, translation=translation, style="", is_ad=is_ad,
                           xyxy=(0, 0, 10, 10), text_class="text_bubble")


SRC = ["Much income they made.",
       "For Kyung-sook's sake, it was natural to cooperate faithfully with anything.",
       "I'm twenty-one this year.",
       "I lacked courtesy. I should have told you first."]
GOOD = ["Ne kadar kazandıklarını bile sorardı.",
        "Kyung-sook'un hatırı için her şeye içtenlikle uymak doğaldı.",
        "Bu yıl yirmi bir yaşındayım.",
        "Nezaketsizlik ettim. Önce söylemeliydim."]


class CheckPageTests(unittest.TestCase):
    def reasons(self, src, tr, lang="Turkish"):
        return cc.check_page([blk(s, t) for s, t in zip(src, tr)], lang)

    def test_clean_page_has_no_flags(self):
        self.assertEqual(self.reasons(SRC, GOOD), [])

    def test_shifted_by_one_is_flagged(self):
        shifted = GOOD[1:] + [GOOD[-1]]
        r = self.reasons(SRC, shifted)
        self.assertTrue(r)
        self.assertTrue(any("identical" in x for x in r))
        self.assertTrue(any("twenty-one" in x or "Kyung" in x for x in r))

    def test_shift_without_duplicate_is_flagged_by_token_check(self):
        shifted = GOOD[1:] + ["Hiç nezaket göstermedim. Önce sana söylemeliydim."]
        r = self.reasons(SRC, shifted)
        self.assertTrue(any("missing" in x for x in r), r)

    def test_identical_translation_for_different_sources(self):
        r = self.reasons(["Hello there friend", "Where are you going"], ["Merhaba dostum", "Merhaba dostum"])
        self.assertTrue(any("identical" in x for x in r))

    def test_same_source_twice_may_share_translation(self):
        self.assertEqual(self.reasons(["Okay!", "Okay!"], ["Tamam!", "Tamam!"]), [])

    def test_empty_translation_flagged(self):
        r = self.reasons(["Where are you going now"], [""])
        self.assertTrue(any("empty" in x for x in r))

    def test_empty_translation_ok_for_ad_and_garbage_source(self):
        page = [blk("DISCORD.GG/abc", "", is_ad=True), blk("T", ""), blk("", "")]
        self.assertEqual(cc.check_page(page, "Turkish"), [])

    def test_untranslated_flagged(self):
        r = self.reasons(["I should have told you first."], ["I should have told you first."])
        self.assertTrue(any("unchanged" in x for x in r))

    def test_short_name_or_sfx_may_stay_unchanged(self):
        self.assertEqual(self.reasons(["Kyung-sook", "BOOM!"], ["Kyung-sook", "BOOM!"]), [])

    def test_length_outlier_flagged(self):
        r = self.reasons(["Where are you going now, big brother?"], ["Ne."])
        self.assertTrue(any("length" in x for x in r))
        r = self.reasons(["Where are you going?"], ["Nereye gidiyorsun? " * 12])
        self.assertTrue(any("length" in x for x in r))

    def test_number_word_and_digits(self):
        self.assertEqual(self.reasons(["I'm twenty-one this year."], ["Bu yıl 21 yaşındayım."]), [])
        self.assertEqual(self.reasons(["I'm twenty-one this year."], ["Bu yıl yirmi bir yaşındayım."]), [])
        r = self.reasons(["I'm twenty-one this year."], ["Önce söylemeliydim sana."])
        self.assertTrue(any("twenty-one" in x for x in r))
        r = self.reasons(["Room 305 is on the left."], ["Oda solda."])
        self.assertTrue(any("305" in x for x in r))

    def test_proper_noun_missing(self):
        r = self.reasons(["Tell Kyung-sook I said hello."], ["Selam söylediğimi ilet."])
        self.assertTrue(any("Kyung-sook" in x for x in r))
        # sentence-initial capitalised words are not names
        self.assertEqual(self.reasons(["Tomorrow we leave early today."], ["Yarın erken çıkıyoruz."]), [])

    def test_translated_place_names_are_not_flagged(self):
        self.assertEqual(self.reasons(["Since it's coming from the United States, it might take a little more time."],
                                      ["ABD'den geldiği için biraz daha zaman alabilir."]), [])

    def test_other_target_language_digits_only(self):
        self.assertEqual(self.reasons(["I'm twenty-one this year."], ["Ich bin dieses Jahr einundzwanzig."], "German"), [])

    def test_disabled_by_env(self):
        with mock.patch.dict(os.environ, {"TRANSLATE_CHECK": "false"}):
            self.assertFalse(cc.enabled())
        with mock.patch.dict(os.environ, {"TRANSLATE_CHECK": "true"}):
            self.assertTrue(cc.enabled())
        env = {k: v for k, v in os.environ.items() if k != "TRANSLATE_CHECK"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(cc.enabled())


class GuardPageTests(unittest.TestCase):
    def test_no_retry_when_clean(self):
        page = [blk(s, t) for s, t in zip(SRC, GOOD)]
        retry = mock.Mock()
        self.assertEqual(cc.guard_page("1", page, retry, "Turkish"), [])
        retry.assert_not_called()

    def test_retry_fixes_page(self):
        page = [blk(s, t) for s, t in zip(SRC, GOOD[1:] + GOOD[-1:])]

        def retry():
            for b, t in zip(page, GOOD):
                b.translation = t
        self.assertEqual(cc.guard_page("25", page, retry, "Turkish"), [])
        self.assertEqual([b.translation for b in page], GOOD)

    def test_worse_retry_is_rolled_back_and_warns(self):
        bad = GOOD[1:] + GOOD[-1:]
        page = [blk(s, t) for s, t in zip(SRC, bad)]

        def retry():
            for b in page:
                b.translation = ""
        with mock.patch("builtins.print") as pr:
            reasons = cc.guard_page("25", page, retry, "Turkish")
        self.assertTrue(reasons)
        self.assertEqual([b.translation for b in page], bad)
        self.assertTrue(any("[translate-check] page 25 suspicious:" in str(c.args[0]) for c in pr.call_args_list))

    def test_retry_exception_keeps_original(self):
        page = [blk("Where are you going now", "")]
        with mock.patch("builtins.print"):
            reasons = cc.guard_page("3", page, mock.Mock(side_effect=RuntimeError("net")), "Turkish")
        self.assertTrue(reasons)
        self.assertEqual(page[0].translation, "")


class FakeEngine:
    max_tokens = 100

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def translate_keyed(self, raw, images=None, extra_context="", ad_detection=False, own_brand=""):
        self.calls.append((dict(raw), extra_context))
        ans = self.answers.pop(0)
        if callable(ans):
            ans = ans(raw)
        if isinstance(ans, Exception):
            raise ans
        return ans

    def row_hint_context(self, keyed_blocks, image, crops=None):
        return ""

    def fix_line_counts(self, blocks, image, extra_context=""):
        return 0


def run_batch(engine, pages, **env):
    args = SimpleNamespace(source_lang="English", target_lang="Turkish", extra_context="")
    translator = SimpleNamespace(engine=engine)
    full_env = {"BATCH_TRANSLATE_RETRY_DELAY_SECONDS": "0", "BATCH_TRANSLATE_CONCURRENCY": "1", **env}
    with mock.patch.dict(os.environ, full_env), \
            mock.patch.object(lb, "LocalMain"), mock.patch.object(lb, "LocalSettings"), \
            mock.patch("modules.translation.processor.Translator", return_value=translator):
        lb._batch_translate_all(pages, args)


def make_pages():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    p0 = [blk("IN THOSE DAYS, SCHOOLS EVEN SURVEYED WHAT PARENTS DID AND HOW")]
    p1 = [blk(s) for s in SRC]
    p2 = [blk("TWENTY-ONE...?")]
    return [(Path("a.webp"), img, p0), (Path("b.webp"), img, p1), (Path("c.webp"), img, p2)]


P0 = "O zamanlar okullar velilerin ne iş yaptığını ve"
P2 = "Yirmi bir...?"


def keyed(values):
    keys = ["p0_b0", "p1_b0", "p1_b1", "p1_b2", "p1_b3", "p2_b0"]
    return {k: v for k, v in zip(keys, values)}


class BatchRetryRegressionTests(unittest.TestCase):
    def test_partial_shifted_answer_is_not_kept(self):
        # first answer: the model merged page-0 sentence + 'Much income', shifted everything up and dropped the last key
        shifted = keyed([P0 + " ne kadar kazandıklarını bile sorardı.", GOOD[1], GOOD[2], GOOD[3],
                         "Hiç nezaket göstermedim. Önce sana söylemeliydim."])
        full = keyed([P0, *GOOD, P2])
        engine = FakeEngine([shifted, full])
        pages = make_pages()
        run_batch(engine, pages)
        self.assertEqual([b.translation for b in pages[1][2]], GOOD)
        self.assertEqual(pages[2][2][0].translation, P2)
        # the retry asked for the WHOLE batch again, not just the missing key
        self.assertEqual(len(engine.calls[1][0]), 6)
        self.assertIn("p1_b0", engine.calls[1][0])

    def test_all_attempts_partial_keeps_page_clean_via_guard(self):
        shifted = keyed([P0, GOOD[1], GOOD[2], GOOD[3], GOOD[3]])  # misses p2_b0 and shifted
        per_page = lambda raw: {k: GOOD[int(k.split("_b")[1])] for k in raw}  # noqa: E731
        engine = FakeEngine([shifted, shifted, shifted, per_page, lambda raw: {k: P2 for k in raw}])
        pages = make_pages()
        with mock.patch("builtins.print"):
            run_batch(engine, pages, BATCH_TRANSLATE_RETRY_ATTEMPTS="3")
        self.assertEqual([b.translation for b in pages[1][2]], GOOD)

    def test_complete_but_shifted_answer_is_fixed_by_guard(self):
        shifted = keyed([P0, GOOD[1], GOOD[2], GOOD[3], GOOD[3], P2])
        engine = FakeEngine([shifted, lambda raw: {k: GOOD[int(k.split("_b")[1])] for k in raw}])
        pages = make_pages()
        with mock.patch("builtins.print"):
            run_batch(engine, pages)
        self.assertEqual([b.translation for b in pages[1][2]], GOOD)
        guard_call = engine.calls[1][0]
        self.assertEqual(sorted(guard_call), ["p1_b0", "p1_b1", "p1_b2", "p1_b3"])  # page alone, all its blocks
        self.assertIn("order", engine.calls[1][1].lower())

    def test_clean_answer_costs_no_extra_call(self):
        engine = FakeEngine([keyed([P0, *GOOD, P2])])
        pages = make_pages()
        run_batch(engine, pages)
        self.assertEqual(len(engine.calls), 1)

    def test_guard_can_be_disabled(self):
        shifted = keyed([P0, GOOD[1], GOOD[2], GOOD[3], GOOD[3], P2])
        engine = FakeEngine([shifted])
        pages = make_pages()
        run_batch(engine, pages, TRANSLATE_CHECK="false")
        self.assertEqual(len(engine.calls), 1)

    def test_translation_stays_with_block_object_after_reorder(self):
        engine = FakeEngine([keyed([P0, *GOOD, P2])])
        pages = make_pages()
        run_batch(engine, pages)
        blocks = pages[1][2]
        blocks.reverse()  # a later stage re-sorting / pruning must not shift translations
        self.assertEqual({b.text: b.translation for b in blocks}, dict(zip(SRC, GOOD)))


if __name__ == "__main__":
    unittest.main()
