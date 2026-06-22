# manga-translation

🇬🇧 English | [🇹🇷 Türkçe](README.tr.md)

A headless, queue-driven worker that translates manga/comic chapters end to
end: download raw pages → detect text → OCR → clean bubbles → translate →
render translated text → upload the result. No GUI, no desktop dependencies
beyond what's needed to render text offscreen.

## Speed & cost

Real numbers from translating a 67-page chapter end to end with the
default engines (RT-DETR v4-s int8 detector, Qwen3-VL-Flash Grid OCR,
GPT-5.4-mini translation), `BATCH_PIPELINE=true`, 3 parallel workers on a
plain CPU machine:

| | OCR (Qwen3-VL-Flash) | Translation (GPT-5.4-mini) | Total |
|---|---|---|---|
| Tokens (in / out) | 16,234 / 1,723 | 1,629 / 2,000 | 21,586 |
| Price per 1M tokens (in / out) | $0.10 / $0.40 | $0.75 / $4.50 | — |
| **Cost for this chapter** | $0.0023 | $0.0102 | **~$0.0125** |

Detection and rendering run locally and cost nothing — the only spend is
OCR + translation API calls. At roughly $0.0125/chapter, **1,000 chapters
of this size cost about $12–13 in API usage**. Total wall-clock time for
this chapter (detect → OCR → clean → translate → render → save) was
**~90–115 seconds**.

Your own numbers will vary with dialogue density per page and current
provider pricing. Set `QWEN_GRID_OCR_USAGE_PATH` / `BATCH_TRANSLATE_USAGE_PATH`
(see `.env.example`) to dump real per-call token usage to a file, or
`LOG_OPENAI_USAGE=true` with a `DATABASE_URL` for persistent cost tracking
(see [Configuration](#configuration)).

## Absolute beginner walkthrough (no prior knowledge needed)

If you've never run a Python project before, do exactly this, in order.

1. **Install Python.** Go to https://www.python.org/downloads/, download
   the latest "Python 3.12" installer for your OS, and run it. On Windows,
   make sure you tick **"Add python.exe to PATH"** during install.
2. **Download this project.** Click the green "Code" button on GitHub →
   "Download ZIP", then unzip it anywhere (e.g. your Desktop).
3. **Open a terminal in that folder.**
   - Windows: open the unzipped folder in File Explorer, click the address
     bar, type `cmd`, press Enter.
   - Mac/Linux: right-click the folder → "Open Terminal here" (or `cd` into
     it manually).
4. **Install the project's dependencies** by typing this and pressing
   Enter:
   ```
   pip install -r requirements.txt
   ```
   This downloads everything the project needs. It can take a few minutes.
5. **Create your settings file.** Copy `.env.example` and rename the copy
   to `.env` (in File Explorer: copy/paste the file, then rename it; on
   Mac/Linux: `cp .env.example .env`).
6. **Get a translation API key.** Go to https://platform.openai.com/api-keys,
   create an account if you don't have one, and create a new API key (it
   looks like `sk-...`). Open `.env` in any text editor (Notepad is fine),
   find the line `OPENAI_API_KEY=`, and paste your key right after the `=`
   with no spaces, e.g. `OPENAI_API_KEY=sk-abc123...`. Save the file.
   (OpenAI charges a small amount per page translated — a few cents per
   chapter is typical.)
7. **Put your images in.** There's an `input` folder in the project — drag
   and drop your manga page images (jpg/png/webp) into it.
8. **Run it.** Back in the terminal, type:
   ```
   python scripts/local_batch.py
   ```
   and press Enter. You'll see progress printed for each page
   (`detect` → `ocr` → `clean` → `translate` → `render` → `save`).
9. **Get your result.** When it finishes, open the `output` folder — your
   translated pages are there. Want everything as a single .zip file
   instead of loose images? Open `.env`, change `ZIP_OUTPUT=false` to
   `ZIP_OUTPUT=true`, save, and run step 8 again — you'll get an
   `output.zip` next to the `output` folder.

That's it — no database, no server, no account other than the translation
API key. Everything below this point is reference documentation for more
advanced setups (changing the target language, running on a server,
hooking it into a website, etc.).

It can run two ways:

- **CLI mode** (`scripts/local_batch.py`) — the main, batteries-included
  way to use this project. Drop images in `input/`, run one command, get
  translated images in `output/`. No database, no cloud account, no
  integration work required. This is what most people want.
- **Worker mode** (`scripts/worker.py`) — an advanced/optional mode that
  polls a Postgres job queue and writes results back into a website's
  database. This is how the original author runs it in production,
  integrated with their own site's schema. It is **not** a drop-in
  component — see [Worker mode / database integration](#worker-mode--database-integration)
  before attempting to use it.

Storage and branding are environment-driven, not hardcoded, so the same
code works for any site name, target language, font, or storage backend.

## Pipeline stages

```
raw pages (R2 bucket or local disk)
        │  download
        ▼
 [1] detect       — find text/bubble regions (RT-DETR ONNX)
        ▼
 [2] ocr          — read text out of each region
        ▼
 [3] clean         — remove source text, inpaint/clean the bubble
        ▼
 [4] translate     — translate each text block (GPT, Gemini, etc.)
        ▼
 [5] render        — draw translated text back onto the page
        ▼
 [6] save          — write the translated image (webp/png/jpg)
        │  upload
        ▼
translated pages (R2 bucket or local disk)
        │
        ▼ (worker mode only)
   write chapter_assets / page rows + SEO alt text to DATABASE_URL
```

## What each stage actually looks like

Still not sure what "detect → ocr → clean → translate → render" means in
practice? Here's the same page, photographed at every step. (Example page is
in English; the target language here is Turkish — yours can be anything via
`TARGET_LANG`.)

| | |
|---|---|
| <img src="assets/examples/1-raw.jpg" width="260"><br>**1. Raw page** — exactly what you dropped into `input/`. Untouched. | <img src="assets/examples/2-detect.jpg" width="260"><br>**2. Detect** — the model finds every speech bubble / text area (green boxes). It doesn't know what the text says yet, just *where* it is. |
| <img src="assets/examples/3-ocr.jpg" width="260"><br>**3. OCR** — each boxed area gets read. The orange text above each box is what the OCR engine thinks it says (here: "GREAT, IT'S A NEW BEGINN...", "NOW THAT"). | <img src="assets/examples/4-cleaned.jpg" width="260"><br>**4. Clean** — the original English text is erased and the bubble is patched back to a plain background, ready for new text. |
| <img src="assets/examples/5-rendered.jpg" width="260"><br>**5. Translate + render** — the OCR'd text is translated ("Great, it's a new beginning!" → "Harika, yeni bir başlangıç!") and drawn back into the cleaned bubble, matching size/position. | This is the image that lands in `output/`. Steps 4 and 5 happen so fast you'll only ever see the raw page and the final result in normal use — these in-between shots exist purely to show what the pipeline does under the hood. |

## Requirements

- Python 3.12
- An API key for whichever translator/OCR engine you select (e.g.
  `OPENAI_API_KEY` for GPT, `DASHSCOPE_API_KEY` for Qwen)
- Optional: Cloudflare R2 (or any S3-compatible) bucket, if not running
  fully local
- Only for worker mode: a Postgres database matching the schema described
  in [Worker mode / database integration](#worker-mode--database-integration)

Model weights (detector, OCR, inpainting) are downloaded automatically on
first use from their public hosts — nothing to download manually.

## Quick start (PC / local use, no database)

This is the easiest way to use the project — no cloud account, no database.

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY at minimum
```

Drop your raw page images into the `input/` folder (already exists in the
repo, empty), then run:

```bash
python scripts/local_batch.py
```

By default this reads everything in `input/`, translates it according to
your `.env` settings, and writes translated pages into `output/`. You can
point at different folders with `--input`/`--output`, or pass
`--zip-output` (or set `ZIP_OUTPUT=true` in `.env`) to also produce an
`output.zip` next to the output folder once the run finishes — handy if you
just want one file to grab or share.

Re-running with the same `output/` folder skips pages that were already
translated, so you can stop and resume a large batch safely.

## Worker mode / database integration

`scripts/worker.py` is **not** meant to be run as-is against an empty
database — it's the exact code the original author runs in production,
wired to their own site's existing `Manga`/`Chapter` tables. Treat it as a
reference implementation to adapt, not a turnkey component.

**Important: the worker never creates jobs by itself.** It only *polls* a
queue table and processes whatever is already sitting in it with
`status = 'QUEUED'`. Something else — your own site/admin panel — has to
`INSERT` a row into that queue (e.g. the moment an admin uploads a new raw
chapter). Wiring that part up is on you; it's straightforward enough that
an AI coding assistant (Claude, GPT, etc.) can write it for you once it has
this section and `scripts/worker.py` as context — just point it at this
file and the table list below and ask it to generate the matching
migration + the "create a job" code for your own backend/admin panel.

### How the queue loop actually behaves

Every `WORKER_IDLE_SECONDS` (default 15s), the worker:

1. Claims one `QUEUED` row from `pipeline_jobs`, ordered by `priority` then
   `created_at` (`FOR UPDATE SKIP LOCKED`, so you can safely run more than
   one worker process against the same queue without double-processing a
   job).
2. Looks at `job_type` and does one of:
   - `TRANSLATE_PROJECT` — finds every untranslated chapter for a manga and
     splits the work into one `TRANSLATE_CHAPTER` job per chapter (so
     multiple workers can pick them up in parallel), then exits.
   - `TRANSLATE_CHAPTER` / `RETRANSLATE_CHAPTER` — translates a single
     chapter: looks up the chapter/manga row, finds the raw images' R2
     prefix, runs `local_batch.py` against it exactly like Scenario A
     above, then writes the results back to the database (see below).
   - `CLEANUP_RAW_CHAPTER` — deletes the raw (untranslated) images for a
     chapter from R2 once you no longer need them.
3. Marks the job `DONE` or `FAILED` (with an `error_message`), or
   `CANCELLED` if your app set its status to `CANCEL_REQUESTED` while it
   was running.

A background tick also reaps jobs stuck in `RUNNING` for >20 minutes
(crashed worker), expires stale `CANCEL_REQUESTED` rows, and deletes old
finished jobs after a few days — so the queue table doesn't grow forever.

### What a finished chapter translation writes back

Once a chapter's pages are translated and uploaded to R2, the worker:

- Inserts one `chapter_assets` row per page (`asset_type = 'TRANSLATED'`)
  with the R2 object key, public URL, and size.
- Inserts one `"ChapterPage"` row per page with the image URL plus
  SEO `altText`/`titleText` built from `SEO_TITLE_TEMPLATE` (see
  [Branding and SEO](#branding-and-seo-are-fully-overridable)).
- Sets `"Chapter"."publishStatus"` and `"Manga"."publishStatus"` to
  `PUBLISHED`, and bumps `"Manga"."latestChapterNo"`.
- Optionally deletes the raw assets, if the chapter/manga is configured to
  auto-delete after translation.
- Optionally calls a webhook (`WEB_INTERNAL_URL` + `INTERNAL_API_KEY`) so
  your frontend can invalidate its cache for that manga.

### Tables you need

| Table | Purpose | Key columns `worker.py` reads/writes |
|---|---|---|
| `pipeline_jobs` | The queue itself | `id`, `manga_id`, `chapter_id`, `job_type` (`TRANSLATE_PROJECT`/`TRANSLATE_CHAPTER`/`RETRANSLATE_CHAPTER`/`CLEANUP_RAW_CHAPTER`), `status` (`QUEUED`/`RUNNING`/`DONE`/`FAILED`/`CANCEL_REQUESTED`/`CANCELLED`), `priority`, `payload` (jsonb), `progress`, `error_message`, `created_at`, `updated_at`, `started_at`, `finished_at` |
| `"Manga"` / `"Chapter"` | Your existing content tables | manga: `id`, `slug`, `publishStatus`, `publishedAt`, `latestChapterNo`; chapter: `id`, `mangaId`, `number`, `slug`, `publishStatus`, `publishedAt` |
| `chapter_assets` | Tracks uploaded image batches per chapter | `id`, `chapter_id`, `page_index`, `asset_type` (`RAW`/`TRANSLATED`), `storage_provider`, `bucket`, `object_key`, `public_url`, `mime_type`, `size_bytes` |
| `"ChapterPage"` | Per-page rows your frontend reads to render a chapter | `id`, `"chapterId"`, `"pageIndex"`, `"imageUrl"`, `"altText"`, `"titleText"`, `"r2Key"` |
| `worker_heartbeats` | Lets you monitor that a worker process is alive | `worker_id`, `last_seen_at`, `jobs_processed` |
| `project_automations` (optional) | Per-manga automation settings | `manga_id`, `auto_delete_raw_after_translate` — if this table/row doesn't exist, the worker defaults to deleting raw assets after translation |

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

### Docker

```bash
docker build -t manga-translation .
docker run --env-file .env manga-translation
```

The image runs Qt in offscreen mode (`QT_QPA_PLATFORM=offscreen`), so no
display server is required.

## Configuration

All configuration lives in environment variables — see `.env.example` for
the full list with defaults and comments. The major groups:

| Group | Variables |
|---|---|
| Database | `DATABASE_URL` |
| Storage backend | `STORAGE_BACKEND` (`r2` or `local`), `LOCAL_STORAGE_ROOT`, `R2_*` |
| Branding / SEO | `SEO_SITE_NAME`, `SEO_LANGUAGE_PHRASE`, `SEO_TITLE_TEMPLATE`, `SEO_IMAGE_BRAND`, `SEO_FILENAME_LANGUAGE_SLUG`, `SEO_FILENAME_TEMPLATE` |
| Languages | `SOURCE_LANG`, `TARGET_LANG` |
| Engine selection | `DETECTOR`, `OCR`, `TRANSLATOR`, `OPENAI_API_KEY`, `OPENAI_MODEL` |
| Font / rendering | `FONT_FAMILY`, `FONT_FILE`, `MIN_FONT_SIZE`, `MAX_FONT_SIZE`, `UPPERCASE`, `NO_OUTLINE` |
| Output | `OUTPUT_SUFFIX`, `OUTPUT_FORMAT`, `WEBP_MAX_DIMENSION` |
| Cleanup tuning | `BUBBLE_*`, `LIGHT_*`, `CROP_CLEAN_*`, `PANEL_SKIP_*` heuristics |

### Branding and SEO are fully overridable

Both the alt-text written to your database and the uploaded image filenames
are built from templates, not hardcoded strings:

- `seo_alt()` in `worker.py` formats `SEO_TITLE_TEMPLATE` with
  `{site_name}`, `{language_phrase}`, `{slug}`, `{number}`, `{page}`.
- `seo_upload_name()` in `local_batch.py` formats `SEO_FILENAME_TEMPLATE`
  with `{brand}`, `{language_slug}`, `{series}`, `{chapter}`, `{page}`,
  `{ext}`.

Change `SEO_SITE_NAME`/`SEO_LANGUAGE_PHRASE`/`SEO_IMAGE_BRAND` (or the
templates themselves) to rebrand for a different site or language without
touching code.

### Storage backends

- `STORAGE_BACKEND=r2` (default): reads/writes through Cloudflare R2 or any
  S3-compatible bucket, configured via `R2_*` variables. This is required
  for worker mode, since the worker needs public URLs to write into the
  database.
- `STORAGE_BACKEND=local`: reads/writes plain files under
  `LOCAL_STORAGE_ROOT`. Works with `local_batch.py` for fully offline runs;
  not used by `worker.py`'s database write-back path.

### Fonts

A font is bundled at `assets/Comic Geek.ttf` and used by default
(`FONT_FILE` in `.env.example`). Swap in your own `.ttf`/`.otf` for a
different look, or clear `FONT_FILE` and set `FONT_FAMILY` to a font already
installed on the host/container. Offscreen Qt has no real system-font
fallback, so leaving `FONT_FILE` empty without a usable `FONT_FAMILY`
installed will render non-ASCII text as missing-glyph boxes.

## License

See `LICENSE`.
