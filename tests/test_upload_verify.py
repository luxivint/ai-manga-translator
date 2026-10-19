"""Upload verification: a chapter is only published / raw only deleted when every page really is in storage."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import upload_verify as uv  # noqa: E402
import worker  # noqa: E402

PREFIX = "translated/s/bolum-4"


class FakeBucket:
    """In-memory R2: drop_once keys are lost on the first upload (partial upload), stuck keys never land."""

    def __init__(self, objects=None, drop_once=(), stuck=()):
        self.objects = dict(objects or {})
        self.drop_once, self.stuck, self.uploads = set(drop_once), set(stuck), []

    def list_objects(self, prefix):
        return {k: v for k, v in self.objects.items() if k.startswith(prefix.rstrip("/") + "/")}

    def upload_file(self, path, key):
        self.uploads.append(key)
        if key in self.stuck:
            raise OSError("connection reset")
        if key in self.drop_once:
            self.drop_once.discard(key)
            return
        self.objects[key] = Path(path).stat().st_size

    def upload_directory(self, folder, prefix):
        for path in sorted(Path(folder).iterdir()):
            self.upload_file(path, f"{prefix}/{path.name}")


def pages(folder, count, size=2000):
    for i in range(1, count + 1):
        (Path(folder) / f"p-sayfa-{i}.webp").write_bytes(b"x" * size)


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class VerifyAndRepairTests(unittest.TestCase):
    def test_complete_upload_is_verified_without_reupload(self):
        with tempfile.TemporaryDirectory() as folder:
            pages(folder, 3)
            bucket = FakeBucket()
            bucket.upload_directory(folder, PREFIX)
            missing = uv.verify_and_repair(Path(folder), PREFIX, bucket.list_objects, bucket.upload_file, log=lambda *a, **k: None)
        self.assertEqual(missing, [])
        self.assertEqual(len(bucket.uploads), 3)

    def test_missing_page_is_reuploaded_once(self):
        with tempfile.TemporaryDirectory() as folder:
            pages(folder, 3)
            bucket = FakeBucket(drop_once={f"{PREFIX}/p-sayfa-2.webp"})
            bucket.upload_directory(folder, PREFIX)
            missing = uv.verify_and_repair(Path(folder), PREFIX, bucket.list_objects, bucket.upload_file, log=lambda *a, **k: None)
        self.assertEqual(missing, [])
        self.assertEqual(bucket.uploads.count(f"{PREFIX}/p-sayfa-2.webp"), 2)

    def test_page_that_never_lands_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            pages(folder, 2)
            stuck = f"{PREFIX}/p-sayfa-1.webp"
            bucket = FakeBucket(stuck={stuck})
            with contextlib.suppress(OSError):
                bucket.upload_directory(folder, PREFIX)
            missing = uv.verify_and_repair(Path(folder), PREFIX, bucket.list_objects, bucket.upload_file, log=lambda *a, **k: None)
        self.assertEqual(missing, [stuck])

    def test_size_mismatch_counts_as_missing(self):
        expected = {f"{PREFIX}/a.webp": 2000}
        self.assertEqual(uv.missing_objects(expected, {f"{PREFIX}/a.webp": 10}), [f"{PREFIX}/a.webp"])

    def test_empty_upload_folder_never_verifies(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = uv.verify_and_repair(Path(folder), PREFIX, FakeBucket().list_objects, FakeBucket().upload_file)
        self.assertEqual(len(missing), 1)

    def test_stale_objects_are_old_pages_only(self):
        expected = {f"{PREFIX}/p-sayfa-1.webp": 1}
        listed = {f"{PREFIX}/p-sayfa-1.webp": 1, f"{PREFIX}/p-sayfa-2.webp": 1, f"{PREFIX}/manifest.json": 1}
        self.assertEqual(uv.stale_objects(expected, listed), [f"{PREFIX}/p-sayfa-2.webp"])
        self.assertEqual(uv.stale_objects({}, listed), [])


class ClassifyTests(unittest.TestCase):
    def test_classification(self):
        self.assertIsNone(uv.classify(200, "image/webp", 7212))
        self.assertEqual(uv.classify(404, "text/plain", None), "http_404")
        self.assertEqual(uv.classify(200, "text/html", 5000), "not_image")
        self.assertEqual(uv.classify(200, "image/webp", 0), "too_small")
        self.assertEqual(uv.classify(0, "", None), "network")
        self.assertEqual(uv.classify(503, "", None), "network")

    def test_network_errors_are_retried_once(self):
        calls = []

        def head(url):
            calls.append(url)
            return (0, "", None) if calls.count(url) == 1 else (200, "image/webp", 5000)

        self.assertEqual(uv.check_urls(["https://cdn/x.webp"], head=head), [])
        self.assertEqual(len(calls), 2)


class LocalBatchPublishTests(unittest.TestCase):
    """main(): verify after upload, re-upload once, keep raw on failure, delete only stale translated pages."""

    def run_main(self, bucket, page_count=3, verify="true"):
        import local_batch as lb
        from types import SimpleNamespace
        tmp = tempfile.mkdtemp()
        upload = Path(tmp) / "seo-upload"
        upload.mkdir()
        pages(upload, page_count)
        args = SimpleNamespace(
            env=".env", mask_refiner="fast", gpu=False, r2_input_prefix="raw/s/bolum-4", r2_output_prefix=PREFIX,
            r2_workdir=tmp, r2_keep_workdir=True, r2_delete_input=True, input="input", output="output",
            zip_output=False, source_lang="English", target_lang="Turkish", safe_recut=False)
        fake_img = Path(tmp) / "x.jpg"
        deleted_prefixes = []

        def fake_batch(images, out_dir, a):
            a.degraded_pages = []
            return 0

        def delete_keys(keys):
            for key in keys:
                bucket.objects.pop(key, None)
            return len(keys)

        patches = [
            mock.patch.object(lb, "parse_args", lambda: args),
            mock.patch.object(lb, "load_env", lambda p: None),
            mock.patch.object(lb, "resolve_font", lambda a: "font"),
            mock.patch.object(lb, "download_prefix", lambda prefix, d: [fake_img]),
            mock.patch.object(lb, "iter_images", lambda d: [fake_img]),
            mock.patch.object(lb, "process_all_batched", fake_batch),
            mock.patch.object(lb, "prepare_seo_upload_dir", lambda *a: upload),
            mock.patch.object(lb, "upload_directory",
                              lambda folder, prefix: (bucket.upload_directory(folder, prefix), {"count": page_count})[1]),
            mock.patch.object(lb, "upload_file", bucket.upload_file),
            mock.patch.object(lb, "list_objects", bucket.list_objects),
            mock.patch.object(lb, "delete_keys", delete_keys),
            mock.patch.object(lb, "delete_prefix", lambda prefix: deleted_prefixes.append(prefix) or 1),
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k", "BATCH_PIPELINE": "true", "R2_VERIFY_UPLOAD": verify}),
        ]
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with contextlib.redirect_stdout(out):
                code = lb.main()
        return code, deleted_prefixes, out.getvalue().strip().splitlines()

    def test_partial_upload_is_repaired_and_raw_deleted(self):
        bucket = FakeBucket(drop_once={f"{PREFIX}/p-sayfa-3.webp"})
        code, deleted, lines = self.run_main(bucket)
        self.assertEqual(code, 0)
        self.assertEqual(deleted, ["raw/s/bolum-4"])  # translated folder is never wiped up front
        self.assertIn(f"{PREFIX}/p-sayfa-3.webp", bucket.objects)
        self.assertTrue(any("R2 verified" in l for l in lines))

    def test_failed_verification_keeps_raw_and_fails(self):
        import local_batch as lb
        bucket = FakeBucket(stuck={f"{PREFIX}/p-sayfa-2.webp"})
        with mock.patch.object(FakeBucket, "upload_directory",
                               lambda self, folder, prefix: [self.upload_file(p, f"{prefix}/{p.name}")
                                                             for p in sorted(Path(folder).iterdir())
                                                             if f"{prefix}/{p.name}" not in self.stuck]):
            code, deleted, lines = self.run_main(bucket)
        self.assertEqual(code, lb.EXIT_VALIDATION_FAILED)
        self.assertEqual(deleted, [])
        self.assertTrue(lines[-1].startswith("FAILED: R2 upload verification failed"), lines[-1])

    def test_old_extra_pages_are_removed_after_upload(self):
        old = {f"{PREFIX}/p-sayfa-{i}.webp": 1 for i in range(1, 6)}
        old[f"{PREFIX}/001.webp"] = 1
        bucket = FakeBucket(objects=old)
        code, _deleted, _lines = self.run_main(bucket, page_count=3)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(bucket.objects), [f"{PREFIX}/p-sayfa-{i}.webp" for i in range(1, 4)])


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params):
        self.sql = sql

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return FakeCursor(self.rows)


class WorkerPublishVerifyTests(unittest.TestCase):
    chapter = {"id": "ch1"}
    rows = [{"r2Key": f"{PREFIX}/p-sayfa-1.webp", "imageUrl": ""}, {"r2Key": f"{PREFIX}/p-sayfa-2.webp", "imageUrl": ""}]

    def env(self, **extra):
        return mock.patch.dict(os.environ, {"R2_PUBLIC_BASE_URL": "https://cdn.example", "TRANSLATION_VERIFY_PUBLISHED": "true", **extra})

    def test_urls_follow_web_reader_rule(self):
        with self.env():
            urls = worker.published_page_urls(FakeConn(self.rows), "ch1")
        self.assertEqual(urls[0], f"https://cdn.example/{PREFIX}/p-sayfa-1.webp")

    def test_broken_page_raises_needs_attention(self):
        check = lambda urls: [{"url": urls[1], "status": 404, "reason": "http_404"}]
        with self.env(), quiet(), self.assertRaises(worker.PublishVerificationError) as ctx:
            worker.verify_published_pages(FakeConn(self.rows), self.chapter, check=check)
        self.assertIn("1/2", str(ctx.exception))

    def test_network_only_problem_does_not_fail(self):
        check = lambda urls: [{"url": urls[0], "status": 0, "reason": "network"}]
        with self.env(), quiet():
            worker.verify_published_pages(FakeConn(self.rows), self.chapter, check=check)

    def test_knob_off_skips(self):
        with self.env(TRANSLATION_VERIFY_PUBLISHED="false"):
            worker.verify_published_pages(FakeConn([]), self.chapter, check=lambda u: 1 / 0)

    def test_run_once_marks_failed_without_retry(self):
        job = {"id": "j1", "job_type": "TRANSLATE_CHAPTER", "chapter_id": "ch1", "manga_id": "m1"}
        with mock.patch.object(worker, "claim_job", return_value=job), \
                mock.patch.object(worker, "process_translate_job", side_effect=worker.PublishVerificationError("x")), \
                mock.patch.object(worker, "mark_failed_or_retry") as retry, \
                mock.patch.object(worker, "set_job") as set_job, quiet():
            worker.run_once(mock.MagicMock())
        retry.assert_not_called()
        self.assertEqual(set_job.call_args.kwargs["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
