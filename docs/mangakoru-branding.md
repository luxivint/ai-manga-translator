# MangaKoru image branding

New translations use `assets/mangakoru-stamp.png`, the approved transparent yellow
wordmark with “En güncel bölümler burada!”. Existing stamp frequency, safe blank-area
placement and opacity settings still apply. If no safe blank area exists, stamping
is skipped rather than covering dialogue or artwork.

```dotenv
SEO_SITE_NAME=MangaKoru
SEO_SITE_URL=https://mangakoru.lol
SEO_IMAGE_BRAND=mangakoru
SEO_LANGUAGE_PHRASE=Türkçe manga oku
SEO_TITLE_TEMPLATE={site_name} | {slug} Bölüm {number}, sayfa {page} — {language_phrase} | {site_url}
STAMP_ENABLED=true
STAMP_PATH=assets/mangakoru-stamp.png
STAMP_BLACK_BACKGROUND=false
```

New upload filenames begin with `mangakoru-`. New `ChapterPage.altText` and
`titleText` use the current brand. SEO settings are read when page metadata is
generated, after `.env` loading, including chapter numbers such as 10 and 100.

This change applies to translation output only. Website source files and existing
database alt text are unchanged. Existing R2 keys and URLs are not renamed;
publication selections, site keys, accounts and chapter identities are unchanged. Existing baked-in image
stamps require re-rendering to change. Deploy the updated code, asset and
environment together on other translation workers before they process new jobs.
