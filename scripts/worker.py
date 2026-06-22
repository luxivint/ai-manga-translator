from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import psycopg2
import psycopg2.extras
import urllib.request
import urllib.error
import json

from r2_storage import R2Config, client, delete_prefix, load_env, normalize_key, public_url


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
JOB_TYPES = ("TRANSLATE_PROJECT", "RETRANSLATE_CHAPTER", "TRANSLATE_CHAPTER", "CLEANUP_RAW_CHAPTER")


class CancelledJob(RuntimeError):
    pass


def make_id(prefix: str) -> str:
    return f"{prefix}{int(time.time() * 1000):x}{os.urandom(4).hex()}"


def db_url() -> str:
    value = os.environ.get("DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is required for the translation worker")
    return value


def connect():
    return psycopg2.connect(db_url(), cursor_factory=psycopg2.extras.RealDictCursor)


def execute(conn, sql: str, params: tuple | list = ()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur


def claim_job(conn):
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE pipeline_jobs
                SET status = 'RUNNING', started_at = now(), updated_at = now(), error_message = NULL
                WHERE id = (
                  SELECT id
                  FROM pipeline_jobs
                  WHERE status = 'QUEUED'
                    AND job_type = ANY(%s)
                  ORDER BY priority ASC, created_at ASC
                  FOR UPDATE SKIP LOCKED
                  LIMIT 1
                )
                RETURNING *
                """,
                (list(JOB_TYPES),),
            )
            return cur.fetchone()


def set_job(conn, job_id: str, status: str | None = None, progress: int | None = None, error: str | None = None):
    fields = ["updated_at = now()"]
    params: list = []
    if status:
        fields.append("status = %s")
        params.append(status)
        if status in {"DONE", "FAILED"}:
            fields.append("finished_at = now()")
    if progress is not None:
        fields.append("progress = %s")
        params.append(max(0, min(100, int(progress))))
    if error is not None:
        fields.append("error_message = %s")
        params.append(error[:2000])
    params.append(job_id)
    with conn:
        execute(conn, f"UPDATE pipeline_jobs SET {', '.join(fields)} WHERE id = %s", params)


def cancel_requested(conn, job_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM pipeline_jobs WHERE id = %s", [job_id])
        row = cur.fetchone()
        return bool(row and row["status"] == "CANCEL_REQUESTED")


def stop_if_cancelled(conn, job_id: str):
    if cancel_requested(conn, job_id):
        raise CancelledJob("Job cancelled by admin")


def heartbeat(conn, job_id: str):
    with conn:
        execute(conn, "UPDATE pipeline_jobs SET updated_at = now() WHERE id = %s", [job_id])


def startup_cleanup(conn):
    """Worker başlarken kendi job türlerindeki zombi RUNNING jobları FAILED yap."""
    with conn:
        cur = execute(conn, """
            UPDATE pipeline_jobs
            SET status = 'FAILED',
                error_message = 'Translator worker yeniden başlatıldı, iş yarıda kesildi.',
                finished_at = now(), updated_at = now()
            WHERE status = 'RUNNING'
              AND job_type = ANY(%s)
        """, [list(JOB_TYPES)])
        if cur.rowcount:
            print(f"startup_cleanup: {cur.rowcount} orphan RUNNING → FAILED", flush=True)
    with conn:
        cur = execute(conn, """
            UPDATE pipeline_jobs
            SET status = 'CANCELLED',
                error_message = 'Translator worker yeniden başlatıldı, kuyruktan kaldırıldı.',
                finished_at = now(), updated_at = now()
            WHERE status = 'CANCEL_REQUESTED'
              AND job_type = ANY(%s)
        """, [list(JOB_TYPES)])
        if cur.rowcount:
            print(f"startup_cleanup: {cur.rowcount} CANCEL_REQUESTED → CANCELLED", flush=True)


def reap_dead_jobs(conn):
    """20 dakika güncellenmeyen çeviri joblarını FAILED yap (çeviri uzun sürer)."""
    with conn:
        execute(conn, """
            UPDATE pipeline_jobs
            SET status = 'FAILED',
                error_message = 'Translator worker 20 dakika yanıt vermedi, otomatik iptal.',
                finished_at = now(), updated_at = now()
            WHERE status = 'RUNNING'
              AND job_type = ANY(%s)
              AND updated_at < now() - interval '20 minutes'
        """, [list(JOB_TYPES)])
    with conn:
        execute(conn, """
            UPDATE pipeline_jobs
            SET status = 'CANCELLED',
                error_message = 'Durdurma talebi zaman aşımına uğradı, otomatik iptal.',
                finished_at = now(), updated_at = now()
            WHERE status = 'CANCEL_REQUESTED'
              AND job_type = ANY(%s)
              AND updated_at < now() - interval '2 minutes'
        """, [list(JOB_TYPES)])


def cleanup_old_jobs(conn):
    """Eski terminal jobları sil."""
    with conn:
        execute(conn, """
            DELETE FROM pipeline_jobs
            WHERE status IN ('DONE', 'CANCELLED')
              AND job_type = ANY(%s)
              AND updated_at < now() - interval '3 days'
        """, [list(JOB_TYPES)])
    with conn:
        execute(conn, """
            DELETE FROM pipeline_jobs
            WHERE status = 'FAILED'
              AND job_type = ANY(%s)
              AND updated_at < now() - interval '14 days'
        """, [list(JOB_TYPES)])


def update_worker_heartbeat(conn, jobs_done: int = 0):
    with conn:
        execute(conn, """
            INSERT INTO worker_heartbeats (worker_id, last_seen_at, jobs_processed)
            VALUES ('translator', now(), %s)
            ON CONFLICT (worker_id) DO UPDATE
              SET last_seen_at = now(),
                  jobs_processed = worker_heartbeats.jobs_processed + EXCLUDED.jobs_processed
        """, [jobs_done])


def prefix_from_key(key: str) -> str:
    key = normalize_key(key)
    return key.rsplit("/", 1)[0] if "/" in key else key


def translated_prefix(raw_prefix: str) -> str:
    raw_prefix = normalize_key(raw_prefix)
    raw_root = normalize_key(os.environ.get("R2_RAW_PREFIX", "raw"))
    translated_root = normalize_key(os.environ.get("R2_TRANSLATED_PREFIX", "translated"))
    if raw_prefix == raw_root:
        return translated_root
    if raw_prefix.startswith(raw_root + "/"):
        return normalize_key(translated_root, raw_prefix[len(raw_root):].lstrip("/"))
    return normalize_key(translated_root, raw_prefix)


def list_images(prefix: str) -> list[dict]:
    config = R2Config.from_env()
    s3 = client(config)
    prefix = normalize_key(prefix)
    items: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=config.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/") or key.endswith("manifest.json"):
                continue
            if Path(key).suffix.lower() not in IMAGE_EXTS:
                continue
            items.append(
                {
                    "key": key,
                    "url": public_url(config, key),
                    "bytes": int(item.get("Size") or 0),
                }
            )
    def page_sort_key(item: dict):
        key = str(item.get("key") or "")
        stem = Path(key).stem.lower()
        match = re.search(r"(?:sayfa|page|p)[-_ ]*(\d+)$|(\d+)$", stem)
        if match:
            number = next((group for group in match.groups() if group), "0")
            return (0, int(number), key)
        return (1, key)

    return sorted(items, key=page_sort_key)


def project_chapters(conn, manga_id: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ch.id, ch."mangaId" AS manga_id, ch.number, ch.slug, m.slug AS manga_slug,
                   MIN(ca.object_key) AS raw_key,
                   COUNT(ca.id) FILTER (WHERE ca.asset_type = 'RAW') AS raw_assets,
                   COUNT(translated.id) AS translated_assets,
                   pa.auto_delete_raw_after_translate
            FROM "Chapter" ch
            JOIN "Manga" m ON m.id = ch."mangaId"
            LEFT JOIN chapter_assets ca ON ca.chapter_id = ch.id AND ca.asset_type = 'RAW'
            LEFT JOIN chapter_assets translated ON translated.chapter_id = ch.id AND translated.asset_type = 'TRANSLATED'
            LEFT JOIN project_automations pa ON pa.manga_id = ch."mangaId"
            WHERE ch."mangaId" = %s
            GROUP BY ch.id, m.slug, pa.auto_delete_raw_after_translate
            HAVING COUNT(ca.id) FILTER (WHERE ca.asset_type = 'RAW') > 0
               AND COUNT(translated.id) = 0
            ORDER BY ch.number ASC
            """,
            [manga_id],
        )
        return cur.fetchall()


def single_chapter(conn, chapter_id: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ch.id, ch."mangaId" AS manga_id, ch.number, ch.slug, m.slug AS manga_slug,
                   MIN(ca.object_key) AS raw_key,
                   COUNT(ca.id) FILTER (WHERE ca.asset_type = 'RAW') AS raw_assets,
                   pa.auto_delete_raw_after_translate
            FROM "Chapter" ch
            JOIN "Manga" m ON m.id = ch."mangaId"
            LEFT JOIN chapter_assets ca ON ca.chapter_id = ch.id AND ca.asset_type = 'RAW'
            LEFT JOIN project_automations pa ON pa.manga_id = ch."mangaId"
            WHERE ch.id = %s
            GROUP BY ch.id, m.slug, pa.auto_delete_raw_after_translate
            """,
            [chapter_id],
        )
        row = cur.fetchone()
        if not row or not row.get("raw_key"):
            raise RuntimeError("Chapter has no raw R2 assets")
        return row


def cleanup_raw(conn, chapter_id: str):
    chapter = single_chapter(conn, chapter_id)
    prefix = prefix_from_key(chapter["raw_key"])
    deleted = delete_prefix(prefix)
    with conn:
        execute(conn, "DELETE FROM chapter_assets WHERE chapter_id = %s AND asset_type = 'RAW'", [chapter_id])
    return deleted


SEO_TITLE_TEMPLATE = os.environ.get(
    "SEO_TITLE_TEMPLATE",
    "{site_name} {language_phrase} {slug} bölüm {number} sayfa {page}",
)
SEO_SITE_NAME = os.environ.get("SEO_SITE_NAME", "TrendManga")
SEO_LANGUAGE_PHRASE = os.environ.get("SEO_LANGUAGE_PHRASE", "Türkçe manga oku")


def seo_alt(manga_slug: str, number, page_index: int) -> str:
    """Build the image alt/title text used for SEO. Fully driven by env vars
    (SEO_SITE_NAME, SEO_LANGUAGE_PHRASE, SEO_TITLE_TEMPLATE) so deployments can
    rebrand without touching code."""
    number_text = str(number).rstrip("0").rstrip(".")
    return SEO_TITLE_TEMPLATE.format(
        site_name=SEO_SITE_NAME,
        language_phrase=SEO_LANGUAGE_PHRASE,
        slug=manga_slug,
        number=number_text,
        page=page_index,
    )


def record_translated_assets(conn, chapter: dict, output_prefix: str, raw_prefix: str, delete_raw: bool):
    images = list_images(output_prefix)
    if not images:
        raise RuntimeError("No translated images found in R2 output prefix")
    with conn:
        execute(conn, 'DELETE FROM chapter_assets WHERE chapter_id = %s AND asset_type = %s', [chapter["id"], "TRANSLATED"])
        execute(conn, 'DELETE FROM "ChapterPage" WHERE "chapterId" = %s', [chapter["id"]])
        for page_index, item in enumerate(images, start=1):
            alt = seo_alt(chapter["manga_slug"], chapter["number"], page_index)
            execute(
                conn,
                """
                INSERT INTO chapter_assets (
                  id, chapter_id, page_index, asset_type, storage_provider,
                  bucket, object_key, public_url, mime_type, size_bytes, created_at, updated_at
                )
                VALUES (%s, %s, %s, 'TRANSLATED', 'R2', %s, %s, %s, 'image/webp', %s, now(), now())
                ON CONFLICT (chapter_id, page_index, asset_type) DO UPDATE SET
                  object_key = EXCLUDED.object_key,
                  public_url = EXCLUDED.public_url,
                  size_bytes = EXCLUDED.size_bytes,
                  updated_at = now()
                """,
                [
                    make_id("asset_"),
                    chapter["id"],
                    page_index,
                    os.environ.get("R2_BUCKET", "manga"),
                    item["key"],
                    item.get("url") or "",
                    item.get("bytes") or 0,
                ],
            )
            execute(
                conn,
                """
                INSERT INTO "ChapterPage" (
                  id, "chapterId", "pageIndex", "imageUrl", "altText", "titleText", "r2Key", "createdAt"
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT ("chapterId", "pageIndex") DO UPDATE SET
                  "imageUrl" = EXCLUDED."imageUrl",
                  "altText" = EXCLUDED."altText",
                  "titleText" = EXCLUDED."titleText",
                  "r2Key" = EXCLUDED."r2Key"
                """,
                [make_id("page_"), chapter["id"], page_index, item.get("url") or "", alt, alt, item["key"]],
            )
        execute(
            conn,
            """
            UPDATE "Chapter"
            SET "publishStatus" = 'PUBLISHED',
                "publishedAt" = COALESCE("publishedAt", now()),
                "updatedAt" = now()
            WHERE id = %s
            """,
            [chapter["id"]],
        )
        execute(
            conn,
            """
            UPDATE "Manga"
            SET "publishStatus" = 'PUBLISHED',
                "publishedAt" = COALESCE("publishedAt", now()),
                "latestChapterNo" = GREATEST(COALESCE("latestChapterNo", 0), %s),
                "updatedAt" = now()
            WHERE id = %s
            """,
            [chapter["number"], chapter["manga_id"]],
        )
        if delete_raw:
            execute(conn, "DELETE FROM chapter_assets WHERE chapter_id = %s AND asset_type = 'RAW'", [chapter["id"]])

    invalidate_web_cache(chapter.get("manga_slug") or "")


def invalidate_web_cache(manga_slug: str):
    url = (os.environ.get("WEB_INTERNAL_URL") or "").rstrip("/")
    key = os.environ.get("INTERNAL_API_KEY") or ""
    if not url or not key or not manga_slug:
        return
    try:
        payload = json.dumps({"slug": manga_slug}).encode()
        req = urllib.request.Request(
            f"{url}/api/internal/revalidate",
            data=payload,
            headers={"Content-Type": "application/json", "x-internal-key": key},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"cache_invalidated: {manga_slug} status={resp.status}", flush=True)
    except Exception as exc:
        print(f"cache_invalidate_warn: {exc}", flush=True)


def translate_chapter(chapter: dict, delete_raw: bool, job_id: str):
    raw_prefix = prefix_from_key(chapter["raw_key"])
    output_prefix = translated_prefix(raw_prefix)
    env = os.environ.copy()
    env.update(
        {
            "PIPELINE_JOB_ID": job_id,
            "TRANSLATION_MANGA_ID": str(chapter.get("manga_id") or ""),
            "TRANSLATION_CHAPTER_ID": str(chapter.get("id") or ""),
            "TRANSLATION_CHAPTER_NUMBER": str(chapter.get("number") or ""),
            "TRANSLATION_MANGA_SLUG": str(chapter.get("manga_slug") or ""),
            "TRANSLATION_RAW_PREFIX": raw_prefix,
            "TRANSLATION_OUTPUT_PREFIX": output_prefix,
            "OPENAI_USAGE_OPERATION": "translation",
        }
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "local_batch.py"),
        "--r2-input-prefix",
        raw_prefix,
        "--r2-output-prefix",
        output_prefix,
        "--output-format",
        "webp",
    ]
    if delete_raw:
        command.append("--r2-delete-input")
    else:
        command.append("--no-r2-delete-input")
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=env)
    return raw_prefix, output_prefix


def enqueue_chapter_jobs(conn, chapters: list, priority: int, payload: dict) -> int:
    """Her chapter için ayrı TRANSLATE_CHAPTER job'u oluştur, zaten kuyrukta olanları atla."""
    created = 0
    for chapter in chapters:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM pipeline_jobs
                WHERE chapter_id = %s
                  AND job_type = 'TRANSLATE_CHAPTER'
                  AND status IN ('QUEUED', 'RUNNING')
                LIMIT 1
                """,
                [chapter["id"]],
            )
            if cur.fetchone():
                print(f"enqueue_chapter_jobs: chapter {chapter['id']} already queued, skipping", flush=True)
                continue
        job_id = make_id("job_")
        with conn:
            execute(
                conn,
                """
                INSERT INTO pipeline_jobs
                  (id, manga_id, chapter_id, job_type, status, priority, payload, created_at, updated_at)
                VALUES (%s, %s, %s, 'TRANSLATE_CHAPTER', 'QUEUED', %s, %s::jsonb, now(), now())
                """,
                [job_id, chapter["manga_id"], chapter["id"], priority, json.dumps(payload)],
            )
        created += 1
    return created


def process_translate_job(conn, job):
    # TRANSLATE_PROJECT → chapter'ları ayrı job'lara böl, worker'lar paralel alsın
    if job["job_type"] == "TRANSLATE_PROJECT":
        chapters = project_chapters(conn, job["manga_id"])
        if not chapters:
            print("no chapters waiting for translation", flush=True)
            return 0
        priority = int(job.get("priority") or 100)
        payload = dict(job.get("payload") or {})
        created = enqueue_chapter_jobs(conn, chapters, priority, payload)
        print(f"TRANSLATE_PROJECT → {created} TRANSLATE_CHAPTER jobs enqueued", flush=True)
        return 0

    # TRANSLATE_CHAPTER / RETRANSLATE_CHAPTER → tek chapter işle
    if job["job_type"] in {"RETRANSLATE_CHAPTER", "TRANSLATE_CHAPTER"}:
        if not job.get("chapter_id"):
            raise RuntimeError(f"{job['job_type']} requires chapter_id")
        chapters = [single_chapter(conn, job["chapter_id"])]
    else:
        chapters = project_chapters(conn, job["manga_id"])

    if not chapters:
        print("no chapters waiting for translation", flush=True)
        return 0

    total = len(chapters)
    for index, chapter in enumerate(chapters, start=1):
        stop_if_cancelled(conn, job["id"])
        heartbeat(conn, job["id"])
        delete_raw = bool(chapter.get("auto_delete_raw_after_translate", True))
        raw_prefix, output_prefix = translate_chapter(chapter, delete_raw, job["id"])
        record_translated_assets(conn, chapter, output_prefix, raw_prefix, delete_raw)
        heartbeat(conn, job["id"])
        set_job(conn, job["id"], progress=int(index / total * 95))
    return total


def run_once(conn) -> bool:
    job = claim_job(conn)
    if not job:
        return False
    print(f"claimed {job['id']} {job['job_type']}", flush=True)
    try:
        if job["job_type"] in {"TRANSLATE_PROJECT", "RETRANSLATE_CHAPTER", "TRANSLATE_CHAPTER"}:
            count = process_translate_job(conn, job)
            print(f"{job['job_type']} translated={count}", flush=True)
        elif job["job_type"] == "CLEANUP_RAW_CHAPTER":
            if not job.get("chapter_id"):
                raise RuntimeError("CLEANUP_RAW_CHAPTER requires chapter_id")
            stop_if_cancelled(conn, job["id"])
            deleted = cleanup_raw(conn, job["chapter_id"])
            print(f"CLEANUP_RAW_CHAPTER deleted={deleted}", flush=True)
        set_job(conn, job["id"], status="DONE", progress=100)
    except CancelledJob as exc:
        print(f"job cancelled {job['id']}: {exc}", flush=True)
        set_job(conn, job["id"], status="CANCELLED", error=str(exc))
    except Exception as exc:
        print(f"job failed {job['id']}: {exc}", flush=True)
        set_job(conn, job["id"], status="FAILED", error=str(exc))
    return True


def main() -> int:
    load_env(ROOT / ".env")
    command = os.environ.get("WORKER_STARTUP_COMMAND", "").strip()
    if command:
        print(f"Running WORKER_STARTUP_COMMAND: {command}", flush=True)
        return subprocess.call(shlex.split(command))

    interval = int(os.environ.get("WORKER_IDLE_SECONDS", "15"))
    conn = connect()
    print("Translation worker connected to pipeline_jobs.", flush=True)
    startup_cleanup(conn)
    update_worker_heartbeat(conn)
    cleanup_tick = 0
    while True:
        try:
            reap_dead_jobs(conn)
            had_job = run_once(conn)
            update_worker_heartbeat(conn, jobs_done=1 if had_job else 0)
            cleanup_tick += 1
            if cleanup_tick >= 20:
                cleanup_old_jobs(conn)
                cleanup_tick = 0
        except psycopg2.Error:
            conn.close()
            time.sleep(2)
            conn = connect()
            continue
        if not had_job:
            time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
