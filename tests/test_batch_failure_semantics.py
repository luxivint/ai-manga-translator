"""Page-level failure semantics of the batch pipeline and the worker: a failed OCR is never an empty page,
a failed/suspicious batch is never published and never deletes the raw pages."""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import local_batch as lb  # noqa: E402
import worker  # noqa: E402


def row(idx, name, failed=False, detected=0, empty=0, kept=0, blocks=None):
    info = {"ocr_failed": failed, "detected": detected, "empty": empty, "kept": kept}
    return (idx, Path(name), object(), blocks or [], info)


class FakeBlock:
    def __init__(self, text="", translation="", is_ad=False):
        self.text, self.translation, self.is_ad = text, translation, is_ad


def env(**kw):
    return mock.patch.dict(os.environ, kw)


def captured():
    return contextlib.redirect_stdout(io.StringIO())


class RerunFailedOcrTests(unittest.TestCase):
    def test_nothing_failed_no_wait(self):
        sleeps = []
        phase1 = [row(0, "1.jpg"), row(1, "2.jpg")]
        with captured():
            out, still = lb.rerun_failed_ocr(phase1, [], run=lambda t: None, sleep=sleeps.append)
        self.assertEqual(still, [])
        self.assertEqual(sleeps, [])
        self.assertEqual(out, phase1)

    def test_failed_pages_rerun_one_by_one_after_cooldown(self):
        order, sleeps = [], []
        phase1 = [row(0, "1.jpg", failed=True), row(1, "2.jpg"), row(2, "3.jpg", failed=True)]
        tasks = [(i, Path(f"{i + 1}.jpg"), None, None) for i in range(3)]

        def run(task):
            order.append(task[0])
            # page 0 recovers, page 2 stays failed
            return row(task[0], f"{task[0] + 1}.jpg", failed=task[0] == 2)

        with captured():
            out, still = lb.rerun_failed_ocr(phase1, tasks, run=run, cooldown=30, sleep=sleeps.append)
        self.assertEqual(sleeps, [30])
        self.assertEqual(order, [0, 2])
        self.assertEqual(still, ["3.jpg"])
        self.assertEqual([r[0] for r in out], [0, 1, 2])
        self.assertFalse(out[0][4]["ocr_failed"])


class EnforceTests(unittest.TestCase):
    def test_any_failure_fails_by_default(self):
        with env(OCR_MAX_FAIL_RATIO="0"), captured():
            with self.assertRaises(lb.BatchValidationError) as ctx:
                lb.enforce_page_failures("OCR failed", ["a.jpg"], 100)
        self.assertEqual(str(ctx.exception), "OCR failed on 1 pages: ['a.jpg']")

    def test_no_failures_is_fine(self):
        lb.enforce_page_failures("OCR failed", [], 10)

    def test_ratio_allows_small_loss_but_not_more(self):
        with env(OCR_MAX_FAIL_RATIO="0.02"), captured():
            lb.enforce_page_failures("OCR failed", ["a"], 100)  # 1% ok
            with self.assertRaises(lb.BatchValidationError):
                lb.enforce_page_failures("OCR failed", ["a", "b", "c"], 100)  # 3%


class PassthroughSanityTests(unittest.TestCase):
    def test_incident_shape_is_rejected(self):
        # 83 of 113 pages: detector found bubbles, OCR came back empty, 0 blocks kept
        rows = [row(i, f"{i}.jpg", detected=2, empty=1, kept=0) for i in range(83)]
        rows += [row(100 + i, f"ok{i}.jpg", detected=3, empty=0, kept=3) for i in range(30)]
        with env(PASSTHROUGH_SUSPECT_RATIO="0.5"), captured():
            with self.assertRaises(lb.BatchValidationError) as ctx:
                lb.check_passthrough_sanity(rows)
        self.assertIn("suspicious passthrough", str(ctx.exception))

    def test_text_free_pages_are_unchanged(self):
        rows = [row(i, f"{i}.jpg", detected=0, empty=0, kept=0) for i in range(50)]
        rows += [row(99, "t.jpg", detected=2, empty=0, kept=2)]
        lb.check_passthrough_sanity(rows)  # must not raise

    def test_non_english_filtered_pages_are_not_suspect(self):
        # detected blocks but OCR was not empty (Korean text read and dropped by the filter)
        rows = [row(i, f"{i}.jpg", detected=2, empty=0, kept=0) for i in range(20)]
        lb.check_passthrough_sanity(rows)

    def test_below_ratio_only_warns(self):
        rows = [row(i, f"{i}.jpg", detected=2, empty=1, kept=0) for i in range(3)]
        rows += [row(10 + i, f"ok{i}.jpg", detected=2, empty=0, kept=2) for i in range(20)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lb.check_passthrough_sanity(rows)
        self.assertIn("WARNING passthrough", out.getvalue())

    def test_ratio_env_is_respected(self):
        rows = [row(i, f"{i}.jpg", detected=2, empty=1, kept=0) for i in range(6)]
        rows += [row(10 + i, f"ok{i}.jpg", detected=2, empty=0, kept=2) for i in range(4)]
        with env(PASSTHROUGH_SUSPECT_RATIO="0.9"), captured():
            lb.check_passthrough_sanity(rows)  # 60% < 90%
        with env(PASSTHROUGH_SUSPECT_RATIO="0.5"), captured():
            with self.assertRaises(lb.BatchValidationError):
                lb.check_passthrough_sanity(rows)

    def test_failed_ocr_pages_are_not_counted_as_passthrough(self):
        rows = [row(i, f"{i}.jpg", failed=True) for i in range(10)]
        self.assertEqual(lb.passthrough_suspects(rows), ([], 0))


class UnresolvedTranslationTests(unittest.TestCase):
    def pages(self, *blocks_per_page):
        return [(Path(f"p{i}.jpg"), None, list(b)) for i, b in enumerate(blocks_per_page)]

    def test_untranslated_text_block_is_unresolved(self):
        pages = self.pages([FakeBlock("I WILL FIND YOU", "")], [FakeBlock("HELLO THERE", "MERHABA")])
        self.assertEqual(lb.unresolved_translation_pages(pages, "English"), ["p0.jpg"])

    def test_ads_and_empty_text_are_not_unresolved(self):
        pages = self.pages([FakeBlock("READ AT SITE.COM", "", is_ad=True), FakeBlock("", "")])
        self.assertEqual(lb.unresolved_translation_pages(pages, "English"), [])


class ProcessAllBatchedFailureTests(unittest.TestCase):
    def run_batch(self, phase1_results, tmp):
        images = [Path(tmp) / f"{i + 1}.jpg" for i in range(len(phase1_results))]
        for p in images:
            p.write_bytes(b"x")
        out_dir = Path(tmp) / "out"
        args = SimpleNamespace(output_format="webp", suffix="", source_lang="English", target_lang="Turkish")
        by_idx = {r[0]: r for r in phase1_results}
        calls = {"translate": 0, "clean": 0, "render": 0}

        def fake_detect(task):
            r = by_idx[task[0]]
            return (r[0], images[r[0]], r[2], r[3], r[4])

        patches = [
            mock.patch.object(lb, "LocalSettings", mock.MagicMock()),
            mock.patch.object(lb, "LocalMain", mock.MagicMock()),
            mock.patch.object(lb, "_prewarm_cleaners", lambda: None),
            mock.patch.object(lb, "TextBlockDetector", mock.MagicMock()),
            mock.patch.object(lb.imk, "read_image", lambda p: None),
            mock.patch.object(lb, "detect_ad_regions", lambda pending, a: {}),
            mock.patch.object(lb, "_dynamic_workers", lambda n: 2),
            mock.patch.object(lb, "_stage_detect_ocr", fake_detect),
            mock.patch.object(lb, "_batch_translate_all", lambda *a, **k: calls.__setitem__("translate", calls["translate"] + 1)),
            mock.patch.object(lb, "_stage_clean_safe", lambda t: calls.__setitem__("clean", calls["clean"] + 1)),
            mock.patch.object(lb, "_do_render_and_save", lambda *a, **k: calls.__setitem__("render", calls["render"] + 1)),
            mock.patch.object(lb.time, "sleep", lambda s: None),
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with captured():
                try:
                    result = lb.process_all_batched(images, out_dir, args)
                    return result, calls, args, None
                except lb.BatchValidationError as exc:
                    return None, calls, args, exc

    def test_persistent_ocr_failure_aborts_before_translate_clean_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [row(0, "1.jpg"), row(1, "2.jpg", failed=True), row(2, "3.jpg")]
            with env(OCR_MAX_FAIL_RATIO="0", OCR_RERUN_COOLDOWN_SECONDS="0"):
                _res, calls, _args, exc = self.run_batch(rows, tmp)
            self.assertIsNotNone(exc)
            self.assertIn("OCR failed on 1 pages", str(exc))
            self.assertEqual(calls, {"translate": 0, "clean": 0, "render": 0})
            self.assertEqual(list((Path(tmp) / "out").glob("*")), [])


class MainPublishGateTests(unittest.TestCase):
    """main(): on a failed batch nothing is uploaded and no raw page is deleted; exit code != 0, last line FAILED."""

    def run_main(self, batch_side_effect, degraded=None):
        tmp = tempfile.mkdtemp()
        args = SimpleNamespace(
            env=".env", mask_refiner="fast", gpu=False, r2_input_prefix="raw/m/c", r2_output_prefix="translated/m/c",
            r2_workdir=tmp, r2_keep_workdir=False, r2_delete_input=True, input="input", output="output",
            zip_output=False, source_lang="English", target_lang="Turkish", safe_recut=False)
        fake_img = Path(tmp) / "x.jpg"

        def fake_batch(images, out_dir, a):
            a.degraded_pages = degraded or []
            if isinstance(batch_side_effect, Exception):
                raise batch_side_effect
            return batch_side_effect

        m = {
            "delete_prefix": mock.MagicMock(return_value=3),
            "upload_directory": mock.MagicMock(return_value={"count": 1}),
        }
        patches = [
            mock.patch.object(lb, "parse_args", lambda: args),
            mock.patch.object(lb, "load_env", lambda p: None),
            mock.patch.object(lb, "resolve_font", lambda a: "font"),
            mock.patch.object(lb, "download_prefix", lambda prefix, d: [fake_img]),
            mock.patch.object(lb, "iter_images", lambda d: [fake_img]),
            mock.patch.object(lb, "process_all_batched", fake_batch),
            mock.patch.object(lb, "prepare_seo_upload_dir", lambda *a: Path(tmp)),
            mock.patch.object(lb, "delete_prefix", m["delete_prefix"]),
            mock.patch.object(lb, "upload_directory", m["upload_directory"]),
            mock.patch.object(lb, "list_objects", lambda prefix: {}),
            # upload verification has its own tests (test_upload_verify.py)
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k", "BATCH_PIPELINE": "true", "R2_VERIFY_UPLOAD": "false"}),
        ]
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with contextlib.redirect_stdout(out):
                code = lb.main()
        return code, m, out.getvalue().strip().splitlines(), tmp

    def test_validation_failure_exits_nonzero_and_publishes_nothing(self):
        code, m, lines, tmp = self.run_main(lb.BatchValidationError("OCR failed on 2 pages: ['a.jpg', 'b.jpg']"))
        self.assertEqual(code, lb.EXIT_VALIDATION_FAILED)
        self.assertNotEqual(code, 0)
        m["upload_directory"].assert_not_called()
        m["delete_prefix"].assert_not_called()
        self.assertTrue(lines[-1].startswith("FAILED: OCR failed on 2 pages"), lines[-1])
        self.assertTrue(Path(tmp).exists())  # local workdir is kept

    def test_success_uploads_and_deletes_raw(self):
        code, m, _lines, _tmp = self.run_main(0)
        self.assertEqual(code, 0)
        m["upload_directory"].assert_called_once()
        # the published translated folder is overwritten in place, never emptied before the upload
        self.assertEqual([c.args[0] for c in m["delete_prefix"].call_args_list], ["raw/m/c"])

    def test_degraded_but_tolerated_batch_uploads_but_keeps_raw(self):
        code, m, lines, _tmp = self.run_main(0, degraded=["7.jpg"])
        self.assertEqual(code, 0)
        m["upload_directory"].assert_called_once()
        self.assertNotIn("raw/m/c", [c.args[0] for c in m["delete_prefix"].call_args_list])
        self.assertTrue(any("R2 raw KEPT" in l for l in lines))


class WorkerFailurePathTests(unittest.TestCase):
    chapter = {"id": "ch1", "manga_id": "m1", "number": 26, "slug": "bolum-26", "manga_slug": "s",
               "raw_key": "raw/s/bolum-26/1.jpg", "auto_delete_raw_after_translate": True}

    def test_nonzero_exit_raises_and_nothing_is_recorded(self):
        err = subprocess.CalledProcessError(worker.LOCAL_BATCH_VALIDATION_EXIT, ["x"])
        with mock.patch.object(worker, "run_batch_process", side_effect=err), \
                mock.patch.object(worker, "single_chapter", return_value=self.chapter), \
                mock.patch.object(worker, "stop_if_cancelled"), mock.patch.object(worker, "heartbeat"), \
                mock.patch.object(worker, "record_translated_assets") as record, \
                mock.patch.object(worker, "delete_prefix") as deleted, \
                mock.patch.object(worker, "set_job") as set_job, \
                captured():
            with self.assertRaises(RuntimeError) as ctx:
                worker.process_translate_job(object(), {"id": "j1", "job_type": "TRANSLATE_CHAPTER",
                                                       "chapter_id": "ch1", "manga_id": "m1"})
        self.assertIn("raw pages kept", str(ctx.exception))
        record.assert_not_called()
        deleted.assert_not_called()
        set_job.assert_not_called()

    def test_run_once_requeues_via_retry_logic_and_never_marks_done(self):
        job = {"id": "j1", "job_type": "TRANSLATE_CHAPTER", "chapter_id": "ch1", "manga_id": "m1"}
        out = io.StringIO()
        with mock.patch.object(worker, "claim_job", return_value=job), \
                mock.patch.object(worker, "process_translate_job", side_effect=RuntimeError("OCR failed")), \
                mock.patch.object(worker, "mark_failed_or_retry") as retry, \
                mock.patch.object(worker, "set_job") as set_job, \
                contextlib.redirect_stdout(out):
            self.assertTrue(worker.run_once(mock.MagicMock()))
        retry.assert_called_once()
        self.assertEqual(retry.call_args.args[1], "j1")
        set_job.assert_not_called()
        self.assertNotIn("translated=", out.getvalue())

    def test_crash_exit_code_is_reported_as_crash(self):
        err = subprocess.CalledProcessError(1, ["x"])
        with mock.patch.object(worker, "run_batch_process", side_effect=err), captured():
            with self.assertRaises(RuntimeError) as ctx:
                worker.translate_chapter(self.chapter, True, "j1")
        self.assertIn("crashed", str(ctx.exception))

    def test_cache_invalidate_sends_user_agent(self):
        seen = {}

        def fake_urlopen(req, timeout=0):
            seen.update({k.lower(): v for k, v in req.header_items()})
            seen["url"] = req.full_url
            resp = mock.MagicMock()
            resp.__enter__.return_value.status = 200
            return resp

        with env(WEB_INTERNAL_URL="https://example.test", INTERNAL_API_KEY="k"), \
                mock.patch.object(worker.urllib.request, "urlopen", fake_urlopen), captured():
            worker.invalidate_web_cache("slug")
        self.assertEqual(seen["url"], "https://example.test/api/internal/revalidate")
        self.assertIn("user-agent", seen)
        self.assertNotIn("python-urllib", seen["user-agent"].lower())
        self.assertIn("x-internal-key", seen)


if __name__ == "__main__":
    unittest.main()
