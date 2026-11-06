# Translation Pipeline — Staged Quality Plan

This is the acceptance checklist for PR #1. **Never merge solely because the
example screenshots look correct.** A complete manga chapter, original input
files, and independent deployment tests are required.

## Stage 1 — Translation correctness and clean art

- [x] OCR rejects missing, duplicate, malformed or out-of-range crop IDs even
  for one/two balloons; explicit empty text remains allowed.
- [x] First batched translation request includes strict stable OCR ID/order rule.
- [x] Existing joined-lobe rendering and white-balloon cleaner retained.
- [x] Synthetic OCR, balloon geometry and white-fill regression cases.
- [ ] **Real-page evidence:** original (untranslated) source pages with at least
  30 cases: single balloon, joined 2 lobes, 3+ lobes, overlapping/touching,
  open outline, black background, gradient, small text, and SFX.
- [ ] **Visual acceptance:** no swapped meaning across balloons, source glyph
  remnants, border damage, or unexpectedly erased untranslated balloons.
- [ ] Manual translation review of discourse order for multi-lobe dialogues.

## Stage 2 — Reliability, publication and CI

- [x] Worker retains raw R2 pages until the published URLs return valid image
  responses; disabled/network-inconclusive verification retains raw pages.
- [x] Standalone CLI defaults to keeping raw sources.
- [x] Per-thread output token limits for concurrent GPT batch requests.
- [x] Provider-origin aware OCR/translation cache with cache-version bump.
- [x] Local resume requires a matching source/settings/code fingerprint.
- [x] GitHub Actions workflow for source compilation and focused regression tests.
- [ ] Verify CI is green on the **latest** commit (older runs are not enough).
- [ ] Staging deployment against actual R2, database, CDN and fault injection
  (404, 429, 503, timeouts, interrupted uploads, worker SIGTERM).
- [ ] Check publication transaction semantics: record/mark PUBLISHED should
  ideally follow a successful public URL check, not precede it.

## Stage 3 — Performance and maintainability

- [x] Conservative white-only YOLO bypass behind
  `CLEAN_SKIP_SEGMENTER_WHITE_ONLY=false` (default off). Only proven full
  interiors qualify; fallback or non-white pages still use existing segmenter.
- [ ] Benchmark 30–100 real source pages before enabling the optimization:
  per-stage wall time, RAM/GPU memory, pixel damage and missed glyph ratio.
- [ ] Split `scripts/local_batch.py` into pipeline stages and
  `modules/rendering/render_v2.py` into geometry, text grouping, fitting and
  rendering, with behavior-preserving tests.
- [ ] Compare batch and single-page pipelines using identical inputs/settings.

## Stage 4 — Quality operations

- [x] Optional nonblocking `QUALITY_REPORT_ENABLED=true`: JSON warnings for
  possible residual ink, missing translation, blank render items and obvious
  off-page text. Diagnostics are **not** proof of an error.
- [ ] Add a full-page QA dashboard with before/after thumbnails, mask previews,
  and review/approve controls. Requires an application UI/backend integration.
- [ ] Allow manual correction of OCR assignment, translation and lobe bounding
  polygon, then selective re-render without paying OCR/translation API again.
- [ ] Add human-labelled reference images/metrics so QA thresholds can be tuned
  against false positives and real mistakes.

## Merge gate

- All latest-commit CI checks green.
- 30+ original-page visual regression fixture results inspected.
- Worker staging smoke with successful publish and raw-retention-on-failure.
- Opt-in performance optimization remains OFF until benchmarks pass.
- Human-reviewed reading order, translations and speech-bubble integrity.

The branch is intentionally a draft PR until these conditions are satisfied.
