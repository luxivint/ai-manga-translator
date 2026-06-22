from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from modules.utils.textblock import TextBlock, adjust_text_line_coordinates

from .base import OCREngine


class QwenGridOCR(OCREngine):
    """Qwen visual OCR via one ID-labelled contact-sheet request."""

    _usage_events: list[dict[str, Any]] = []

    def __init__(self):
        self.api_key = ""
        self.model = "qwen3-vl-flash"
        self.api_base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.max_tokens = 2500
        self.input_price_per_1m = 0.10
        self.output_price_per_1m = 0.40

    def initialize(self, api_key: str = "", model: str = "qwen3-vl-flash", **kwargs) -> None:
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.api_base_url = os.environ.get("DASHSCOPE_BASE_URL", self.api_base_url).rstrip("/") + "/chat/completions"
        if self.api_base_url.endswith("/v1/chat/completions/chat/completions"):
            self.api_base_url = self.api_base_url.replace("/chat/completions/chat/completions", "/chat/completions")
        self.max_tokens = int(kwargs.get("max_tokens", os.environ.get("QWEN_GRID_OCR_MAX_TOKENS", "2500")))
        self.input_price_per_1m = float(os.environ.get("QWEN_INPUT_PRICE_PER_1M", "0.10"))
        self.output_price_per_1m = float(os.environ.get("QWEN_OUTPUT_PRICE_PER_1M", "0.40"))

    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        if not blk_list:
            return blk_list
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is missing for QwenGridOCR.")

        sheet, id_to_idx = self._make_sheet(img, blk_list)
        if sheet is None:
            return blk_list

        prompt = (
            "You are an OCR engine for comic/webtoon text crops. "
            "The image is a contact sheet. Each crop has a yellow ID label like ID 1, ID 2, etc. "
            "Read ONLY the comic text inside each crop, not the yellow ID label. Do NOT translate. "
            "Only return English/Latin alphabet dialogue, narration, and signs. "
            "If the crop is Korean, Japanese, Chinese, raw SFX, watermark text, or has no readable English text, return an empty string. "
            "Preserve the original wording as much as possible. Use spaces instead of line breaks. "
            "If a crop has no readable text, return an empty string for it. "
            'Return strict JSON only: [{"id":1,"text":"..."},{"id":2,"text":"..."}].'
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(np.asarray(sheet), 'jpg')}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }

        max_attempts = max(1, int(os.environ.get("QWEN_GRID_OCR_RETRIES", "2")))
        parsed: list[dict[str, Any]] = []
        data: dict[str, Any] = {}
        seconds = 0.0
        for attempt in range(1, max_attempts + 1):
            started = time.time()
            response = requests.post(
                self.api_base_url,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                data=json.dumps(payload),
                timeout=int(os.environ.get("QWEN_GRID_OCR_TIMEOUT", "90")),
            )
            seconds = time.time() - started
            if response.status_code != 200:
                print(f"Qwen grid OCR API error: {response.status_code} {response.text}", flush=True)
                return blk_list

            data = response.json()
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            parsed = self._parse_json(content)
            if not self._looks_like_failed_batch(parsed, len(id_to_idx)):
                break
            if attempt < max_attempts:
                print(
                    f"Qwen grid OCR retry {attempt}/{max_attempts}: suspicious empty result "
                    f"for {len(id_to_idx)} crops",
                    flush=True,
                )

        for item in parsed:
            try:
                crop_id = int(item.get("id"))
            except Exception:
                continue
            idx = id_to_idx.get(crop_id)
            if idx is not None and 0 <= idx < len(blk_list):
                blk_list[idx].text = self._normalize_text(str(item.get("text") or ""))

        self._log_usage(data.get("usage") or {}, seconds, len(blk_list))
        return blk_list

    @staticmethod
    def _looks_like_failed_batch(parsed: list[dict[str, Any]], crop_count: int) -> bool:
        if crop_count <= 2:
            return False
        if not parsed:
            return True
        recognized = 0
        empty = 0
        for item in parsed:
            if not isinstance(item, dict) or "id" not in item:
                continue
            recognized += 1
            if not str(item.get("text") or "").strip():
                empty += 1
        if recognized < max(1, int(crop_count * 0.6)):
            return True
        return empty / max(1, recognized) >= float(os.environ.get("QWEN_GRID_OCR_RETRY_EMPTY_RATIO", "0.85"))

    def _make_sheet(self, img: np.ndarray, blk_list: list[TextBlock]) -> tuple[Image.Image | None, dict[int, int]]:
        source = Image.fromarray(img).convert("RGB")
        max_cell_w = int(os.environ.get("QWEN_GRID_OCR_CELL_WIDTH", "360"))
        pad = int(os.environ.get("QWEN_GRID_OCR_CROP_PAD", "12"))
        cols = int(os.environ.get("QWEN_GRID_OCR_COLS", "2"))
        gap = 16
        header_h = 34
        cells: list[tuple[int, int, Image.Image]] = []
        id_to_idx: dict[int, int] = {}
        width, height = source.size

        for idx, blk in enumerate(blk_list):
            xyxy = adjust_text_line_coordinates(blk.xyxy, 12, 20, img)
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(width, x2 + pad)
            y2 = min(height, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = source.crop((x1, y1, x2, y2))
            if crop.width > max_cell_w:
                ratio = max_cell_w / crop.width
                crop = crop.resize((max_cell_w, max(1, int(crop.height * ratio))), Image.LANCZOS)
            crop_id = len(cells) + 1
            id_to_idx[crop_id] = idx
            cell = Image.new("RGB", (max(crop.width, 130), crop.height + header_h), "white")
            cell.paste(crop, (0, header_h))
            draw = ImageDraw.Draw(cell)
            draw.rectangle((0, 0, cell.width, header_h), fill=(255, 218, 0))
            draw.text((8, 7), f"ID {crop_id}", fill=(0, 0, 0), font=self._font(22))
            cells.append((cell.width, cell.height, cell))

        if not cells:
            return None, {}

        rows = (len(cells) + cols - 1) // cols
        col_w = [0] * cols
        row_h = [0] * rows
        for i, (w, h, _cell) in enumerate(cells):
            c = i % cols
            r = i // cols
            col_w[c] = max(col_w[c], w)
            row_h[r] = max(row_h[r], h)
        sheet = Image.new("RGB", (sum(col_w) + gap * (cols + 1), sum(row_h) + gap * (rows + 1)), (32, 32, 32))
        y = gap
        for r in range(rows):
            x = gap
            for c in range(cols):
                i = r * cols + c
                if i >= len(cells):
                    break
                _w, _h, cell = cells[i]
                sheet.paste(cell, (x, y))
                x += col_w[c] + gap
            y += row_h[r] + gap
        return sheet, id_to_idx

    @staticmethod
    def _font(size: int) -> ImageFont.ImageFont:
        for path in [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"]:
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _log_usage(self, usage: dict[str, Any], seconds: float, block_count: int) -> None:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        cost = (
            prompt_tokens * self.input_price_per_1m / 1_000_000
            + completion_tokens * self.output_price_per_1m / 1_000_000
        )
        event = {
            "model": self.model,
            "blocks": block_count,
            "seconds": round(seconds, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(cost, 8),
        }
        self._usage_events.append(event)
        path = os.environ.get("QWEN_GRID_OCR_USAGE_PATH")
        if path:
            aggregate = {
                "events": self._usage_events,
                "totals": {
                    "calls": len(self._usage_events),
                    "blocks": sum(e["blocks"] for e in self._usage_events),
                    "seconds": round(sum(e["seconds"] for e in self._usage_events), 3),
                    "prompt_tokens": sum(e["prompt_tokens"] for e in self._usage_events),
                    "completion_tokens": sum(e["completion_tokens"] for e in self._usage_events),
                    "total_tokens": sum(e["total_tokens"] for e in self._usage_events),
                    "estimated_cost_usd": round(sum(e["estimated_cost_usd"] for e in self._usage_events), 8),
                },
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(aggregate, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _parse_json(text: str) -> list[dict[str, Any]]:
        text = text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
        if fence:
            text = fence.group(1).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1 or end <= start:
                return []
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return []
        if isinstance(parsed, dict):
            parsed = parsed.get("items") or parsed.get("blocks") or []
        return parsed if isinstance(parsed, list) else []

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
