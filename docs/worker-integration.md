# Worker integration

The queue worker is a TrendManga integration, not a generic database installer.
The admin/downloader creates jobs; the worker claims queued jobs and updates progress.
Adapt the schema and queries before using it with another application.

## Publication

Once a chapter's pages are translated and uploaded to R2, the worker:

- Inserts one `chapter_assets` row per page (`asset_type = 'TRANSLATED'`)
  with the R2 object key, public URL, and size.
- Inserts one `"ChapterPage"` row per page with the image URL plus
  SEO `altText`/`titleText` built from `SEO_TITLE_TEMPLATE` (see
  [environment template](../.env.example)).
- Sets `"Chapter"."publishStatus"` and `"Manga"."publishStatus"` to
  `PUBLISHED`, and bumps `"Manga"."latestChapterNo"`.
- Optionally deletes raw assets after successful publication and public URL verification,
  when the chapter/manga is configured to auto-delete them.
- Optionally calls a webhook (`WEB_INTERNAL_URL` + `INTERNAL_API_KEY`) so
  your frontend can invalidate its cache for that manga.

## Database contract

| Table | Purpose | Key columns `worker.py` reads/writes |
|---|---|---|
| `pipeline_jobs` | The queue itself | `id`, `manga_id`, `chapter_id`, `job_type` (`TRANSLATE_PROJECT`/`TRANSLATE_CHAPTER`/`RETRANSLATE_CHAPTER`/`CLEANUP_RAW_CHAPTER`), `status` (`QUEUED`/`RUNNING`/`DONE`/`FAILED`/`CANCEL_REQUESTED`/`CANCELLED`), `priority`, `payload` (jsonb), `progress`, `error_message`, `created_at`, `updated_at`, `started_at`, `finished_at` |
| `"Manga"` / `"Chapter"` | Your existing content tables | manga: `id`, `slug`, `publishStatus`, `publishedAt`, `latestChapterNo`; chapter: `id`, `mangaId`, `number`, `slug`, `publishStatus`, `publishedAt` |
| `chapter_assets` | Tracks uploaded image batches per chapter | `id`, `chapter_id`, `page_index`, `asset_type` (`RAW`/`TRANSLATED`), `storage_provider`, `bucket`, `object_key`, `public_url`, `mime_type`, `size_bytes` |
| `"ChapterPage"` | Per-page rows your frontend reads to render a chapter | `id`, `"chapterId"`, `"pageIndex"`, `"imageUrl"`, `"altText"`, `"titleText"`, `"r2Key"` |
| `worker_heartbeats` | Lets you monitor that a worker process is alive | `worker_id`, `last_seen_at`, `jobs_processed` |
| `project_automations` | Per-manga automation settings | `manga_id`, `auto_delete_raw_after_translate` — the table is queried directly; create it even if there are no per-manga overrides |

Read through `scripts/worker.py` top to bottom — every SQL statement names
the exact columns it expects — and adjust the queries to match your schema
before pointing `DATABASE_URL` at a real database. The piece you still need
to build yourself is whatever inserts `TRANSLATE_CHAPTER`/`TRANSLATE_PROJECT`
rows into `pipeline_jobs` when you actually want something translated.

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATABASE_URL, storage backend, etc.

python scripts/worker.py
```


A completed process is not sufficient: inspect job status, page records and public image URLs.
