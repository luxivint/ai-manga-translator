# manga-translation

A headless, queue-driven worker that translates manga/comic chapters end to
end: download raw pages → detect text → OCR → clean bubbles → translate →
render translated text → upload the result. No GUI, no desktop dependencies
beyond what's needed to render text offscreen.

It can run two ways:

- **Worker mode** (`scripts/worker.py`) — polls a Postgres `pipeline_jobs`
  queue and processes chapters as jobs arrive, writing results back to your
  database (page rows, SEO alt text, asset URLs).
- **CLI mode** (`scripts/local_batch.py`) — translates one folder of images
  into another, no database or queue required. Useful for local testing,
  one-off batches, or running entirely without a server.

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

## Requirements

- Python 3.12
- For worker mode: a Postgres database with a `pipeline_jobs` table your
  own application enqueues jobs into (see `scripts/worker.py` for the
  expected job shape)
- An API key for whichever translator/OCR engine you select (e.g.
  `OPENAI_API_KEY` for GPT, `DASHSCOPE_API_KEY` for Qwen)
- Optional: Cloudflare R2 (or any S3-compatible) bucket, if not running
  fully local

Model weights (detector, OCR, inpainting) are downloaded automatically on
first use from their public hosts — nothing to download manually.

## Quick start (local, no database)

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY at minimum

python scripts/local_batch.py --input ./input --output ./output
```

This reads every image in `./input`, translates it according to your `.env`
settings, and writes the result to `./output`.

## Running the worker

The worker continuously polls `DATABASE_URL` for queued jobs and shells out
to `local_batch.py` per chapter, downloading/uploading via whichever
`STORAGE_BACKEND` you've configured.

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
