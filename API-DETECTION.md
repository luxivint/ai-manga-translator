# Remote comic detection

The local RT-DETR path stays the default. Select the remote engine explicitly:

```dotenv
DETECTION_BACKEND=api
DETECTION_API_COORDINATES=pixels
DETECTION_API_MAX_SIDE=1536
DETECTION_API_TEXT_PADDING=0.05
DETECTION_API_TIMEOUT=120
DETECTION_API_VALIDATION_RETRIES=1
```

Existing `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `OPENAI_MODEL` are reused. Optional separate `DETECTION_API_BASE_URL`, `DETECTION_API_KEY` and `DETECTION_API_MODEL` override them. The base URL is an OpenAI-compatible Chat Completions service. No keys belong in this document. Existing OCR, translation, authentication, catalogue, database and R2 contracts are unchanged.

## What moves off the CPU

Both safe RECUT's `_detect_single_image` calls and normal `detect` calls select the remote engine through the same factory. API mode never creates or silently falls back to a local RT-DETR engine. The separate font/colour, segmentation, cleaning, image encoding and render stages may still use CPU or local models. This feature does not offload LaMa.

## Response correctness and operational bounds

- The request gives exact image dimensions and pixel coordinate conventions. `normalized` selects an explicit 0..1000 adapter for models that prefer that convention.
- Very long images use overlapping, bounded-height tiles, covering the last row. Regions map back to the original image; near-identical tile duplicates are removed.
- Text boxes get bounded context padding: this addresses small localization errors but cannot recover a missing detection. Boxes feed the existing segmentation pipeline; they are not rectangular erase masks.
- A complete `stop` response, valid JSON, explicit `has_text`, consistent region types, finite ordered coordinates and correct bounds are required. Invalid answers retry once by default and then raise. Network/auth/quota failures never become an empty page.
- The existing gateway retries, global quota guard, shared OCR concurrency slots, and manually committed response cache are reused. Only validated API responses enter the cache. This is response caching, not a guarantee that different RECUT and final-page requests reuse geometry.
- Logs include `[DETECT API] tile=... bubbles=... text=... elapsed=...`; credentials and request bodies are not logged by this engine.
- API selection does not deploy or restart anything. Set the variables in Dokploy and deploy only when no translation is running. Rollback: `DETECTION_BACKEND=onnx` and restart between jobs.
- General visual LLM boxes remain approximate. `has_text` is a structural consistency check, not proof against missed writing. Review output on representative full chapters before broad use. The four-image test is not a production accuracy or full-chapter speed guarantee.

## Evidence and alternatives (1 Nov 2026)

Initial experiment: `Manga/artifacts/api-detection-20261101/`, same four images, existing router `cx/gpt-6-luna`, CPU-only local detector. The API returned locations, but some text boxes cut through glyphs. Revised pixel-coordinate requests and padding need separate validation.

Official references:

- Gemini detection returns normalized bounding boxes; its documented order is **ymin,xmin,ymax,xmax**. This adapter's JSON contract is **x1,y1,x2,y2**; do not pass raw Gemini boxes without converting. https://ai.google.dev/gemini-api/docs/image-understanding
- Claude recommends explicit pixel coordinates and notes approximate spatial accuracy. https://platform.claude.com/docs/en/build-with-claude/vision-coordinates
- Roboflow hosts specialized detection models behind an inference API, but requires its own model/account/key. Existing RT-DETR weights are not automatically hosted by the current text/vision router. No Roboflow image uploads or paid provisioning were performed. https://docs.roboflow.com/deployment/roboflow-cloud/serverless-api.md

Run focused offline regression checks from this repository:

```sh
python -m unittest discover -s tests -p test_api_detection.py -v
```
