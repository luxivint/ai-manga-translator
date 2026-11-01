"""Remote comic region detection; deliberately never falls back to local inference."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
import re
import time
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from .base import DetectionEngine
from modules.utils import gateway
from modules.utils.openai_endpoint import get_openai_base_url, get_openai_model


class DetectionAPIError(gateway.GatewayError):
    """No trustworthy detection response was obtained; preserve the source."""


def configuration():
    base = (os.environ.get('DETECTION_API_BASE_URL') or get_openai_base_url()).strip().rstrip('/')
    parsed = urlsplit(base)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise ValueError('DETECTION_API_BASE_URL must be an HTTP(S) URL')
    return {
        'url': base if base.endswith('/chat/completions') else base + '/chat/completions',
        'key': os.environ.get('DETECTION_API_KEY') or os.environ.get('OPENAI_API_KEY', ''),
        'model': os.environ.get('DETECTION_API_MODEL') or get_openai_model(None),
        'max_side': int(os.environ.get('DETECTION_API_MAX_SIDE', '1536')),
        'coordinates': os.environ.get('DETECTION_API_COORDINATES', 'pixels').lower(),
        'padding': float(os.environ.get('DETECTION_API_TEXT_PADDING', '0.05')),
        'timeout': float(os.environ.get('DETECTION_API_TIMEOUT', '120')),
        'retries': int(os.environ.get('DETECTION_API_VALIDATION_RETRIES', '1')),
    }


def cache_identity():
    # Include credentials in a one-way digest, never in the log/cache name.
    return hashlib.sha256(json.dumps(configuration(), sort_keys=True).encode()).hexdigest()


def tile_starts(height, tile_height, overlap):
    """Cover the final row even when the last tile would otherwise be too short."""
    if height <= tile_height:
        return [0]
    step = max(1, tile_height - overlap)
    starts=[0]
    while starts[-1]+tile_height < height:
        nxt=min(starts[-1]+step,height-tile_height)
        if 0 < height-(nxt+tile_height) <= overlap:
            nxt=height-tile_height
        starts.append(nxt)
    return starts


def deduplicate(boxes):
    """Remove near-identical overlap-window boxes; never merge nearby dialogues."""
    kept = []
    for box in sorted(boxes, key=lambda b: -(b[2]-b[0])*(b[3]-b[1])):
        duplicate = False
        for other in kept:
            iw = max(0, min(box[2],other[2])-max(box[0],other[0]))
            ih = max(0, min(box[3],other[3])-max(box[1],other[1]))
            inter = iw*ih
            union = (box[2]-box[0])*(box[3]-box[1])+(other[2]-other[0])*(other[3]-other[1])-inter
            if union and inter/union > .65:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return np.asarray(kept,dtype=np.int32).reshape(-1,4)


class APIDetection(DetectionEngine):
    """Same bubble/text and TextBlock contracts as the local RT-DETR engine."""

    parallel_safe = True  # only immutable config + remote I/O in raw detection

    def initialize(self, **kwargs):
        self.config = configuration()
        self.concurrency = gateway.detection_concurrency()
        self.reuse_identity = cache_identity()
        c = self.config
        if not c['key']:
            raise ValueError('DETECTION_API_KEY or OPENAI_API_KEY is required')
        if not 512 <= c['max_side'] <= 2048:
            raise ValueError('DETECTION_API_MAX_SIDE must be between 512 and 2048')
        if c['coordinates'] not in ('pixels', 'normalized'):
            raise ValueError('DETECTION_API_COORDINATES must be pixels or normalized')
        if not 0 <= c['padding'] <= .15 or not 0 <= c['retries'] <= 2 or not 1 <= c['timeout'] <= 300:
            raise ValueError('Invalid detection API padding, retries or timeout')
        # safe_recut temporarily modifies this field on every detection engine.
        self.confidence_threshold = .3

    def detect(self, image):
        bubbles, texts = self._detect_single_image(image)
        # Retain the existing downstream font/colour and TextBlock contracts.
        # This may still run the separate font model; RT-DETR is never loaded here.
        return self.create_text_blocks(image, texts, bubbles)

    def _detect_single_image(self, image):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Detection API expects a uint8 RGB image')
        h,w = image.shape[:2]
        if not h or not w:
            raise ValueError('Empty detection image')
        # Keep small letters readable instead of shrinking an entire long strip.
        tile_h = max(512, min(self.config['max_side'], round(w*2)))
        starts = tile_starts(h,tile_h,round(tile_h*.2)) if h > tile_h*1.3 else [0]
        bubbles,texts = [],[]
        started = time.monotonic()
        def request(top):
            bottom = min(h, top+tile_h) if len(starts)>1 else h
            b,t = self._request_tile(image[top:bottom])
            return top,b,t

        def consume(results):
            for i,(top,b,t) in enumerate(results):
                for source,target in ((b,bubbles),(t,texts)):
                    for box in source:
                        box=list(box);box[1]+=top;box[3]+=top;target.append(box)
                print(f'[DETECT API] tile={i+1}/{len(starts)} bubbles={len(b)} text={len(t)} '
                      f'elapsed={time.monotonic()-started:.2f}s parallel={min(self.concurrency,len(starts))}',flush=True)

        if len(starts)>1 and self.concurrency>1:
            # Submit only coordinates; image views/encoding are made in running workers.
            with ThreadPoolExecutor(max_workers=min(self.concurrency,len(starts))) as pool:
                consume(pool.map(request,starts))
        else:
            consume(map(request,starts))
        return deduplicate(bubbles),deduplicate(texts)

    def _prompt(self, width, height):
        units = (f'ABSOLUTE PIXELS for this exact {width} by {height} image; x ranges 0..{width}, y ranges 0..{height}'
                 if self.config['coordinates']=='pixels' else 'coordinates normalized to 0..1000 along each axis')
        return (
            'Detect every visible comic text group and speech balloon/narration container. Do not translate. '
            f'Use {units}. Boxes are [x1,y1,x2,y2], top-left origin, x horizontal and y vertical. '
            'Group the lines of each dialogue into ONE text box. Include ALL ink, outlines, punctuation and diacritics; '
            'leave 2 pixels around the complete glyphs. Inspect rightmost/bottommost letters. '
            'Include signs, captions, watermarks and sound effects, including non-Latin text. '
            'A bubble box covers its complete visible container, separate from its text box. '
            'Include partial containers at image edges but never infer invisible text. '
            'Do not detect characters, panels or scenery. has_text indicates visible writing, NOT empty balloons. '
            'Only return strict JSON: {"has_text":true,"regions":[{"kind":"text","box":[1,2,30,40]},'
            '{"kind":"bubble","box":[0,0,35,45]}]}. '
            'Use {"has_text":false,"regions":[]} only when there are no text or balloon regions. '
            'If only empty balloons exist, has_text is false and regions contains the bubble boxes. '
            'Do not add text transcripts, prose or markdown.'
        )

    def _parse(self, body, sent_size, original_size):
        try:
            choice = body['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise ValueError('incomplete or refused response')
            content = choice['message']['content']
            if isinstance(content,list):
                content = ''.join(x.get('text','') for x in content if isinstance(x,dict))
            content = re.sub(r'^```(?:json)?\s*|\s*```$','',content.strip())
            data = json.loads(content)
            if not isinstance(data,dict) or type(data.get('has_text')) is not bool or not isinstance(data.get('regions'),list):
                raise ValueError('missing explicit has_text / regions')
            if len(data['regions']) > 250:
                raise ValueError('too many regions')
            sw,sh = sent_size;ow,oh = original_size
            mx,my = (sw,sh) if self.config['coordinates']=='pixels' else (1000,1000)
            bubbles,texts = [],[]
            for region in data['regions']:
                if not isinstance(region,dict) or region.get('kind') not in ('bubble','text'):
                    raise ValueError('unknown region type')
                box = region.get('box')
                if not isinstance(box,list) or len(box)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in box):
                    raise ValueError('invalid coordinates')
                x1,y1,x2,y2=box
                if not (0 <= x1 < x2 <= mx and 0 <= y1 < y2 <= my):
                    raise ValueError('out-of-range or inverted box')
                b=[math.floor(x1*ow/mx),math.floor(y1*oh/my),math.ceil(x2*ow/mx),math.ceil(y2*oh/my)]
                if region['kind']=='text':
                    # Context for the segmenter/OCR, not a rectangular erase mask.
                    px=min(32,max(4,math.ceil((b[2]-b[0])*self.config['padding']))) if self.config['padding'] else 0
                    py=min(20,max(4,math.ceil((b[3]-b[1])*self.config['padding']))) if self.config['padding'] else 0
                    b=[max(0,b[0]-px),max(0,b[1]-py),min(ow,b[2]+px),min(oh,b[3]+py)]
                    # Edge fragments are common before RECUT; include the visible
                    # clipped glyphs rather than ending a few pixels before them.
                    ex,ey=min(20,max(4,round(ow*.02))),min(20,max(4,round(oh*.02)))
                    if b[0] <= ex:b[0]=0
                    if b[1] <= ey:b[1]=0
                    if ow-b[2] <= ex:b[2]=ow
                    if oh-b[3] <= ey:b[3]=oh
                    texts.append(b)
                else:
                    bubbles.append(b)
            if data['has_text'] != bool(texts):
                raise ValueError('has_text contradicts text regions')
            return bubbles,texts
        except (ValueError,TypeError,KeyError,IndexError,AttributeError) as exc:
            raise DetectionAPIError('Invalid detection response: '+str(exc)) from None

    def _request_tile(self, image):
        pil=Image.fromarray(image);original=pil.size
        pil.thumbnail((self.config['max_side'],self.config['max_side']),Image.Resampling.LANCZOS)
        buf=io.BytesIO();pil.save(buf,format='JPEG',quality=95)
        payload={'model':self.config['model'],'temperature':0,'max_completion_tokens':6000,
                 'messages':[{'role':'user','content':[
                     {'type':'text','text':self._prompt(*pil.size)},
                     {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()}}
                 ]}]}
        last=None
        for attempt in range(self.config['retries']+1):
            response=gateway.resilient_post(self.config['url'],headers={'Content-Type':'application/json','Authorization':'Bearer '+self.config['key']},
                data=json.dumps(payload),timeout=self.config['timeout'],kind='detection',limiter=gateway.detection_limiter(),cache='manual')
            if response.status_code!=200:
                gateway.cache_discard()
                raise DetectionAPIError(f'Detection API HTTP {response.status_code}')
            try:
                result=self._parse(response.json(),pil.size,original)
            except DetectionAPIError as exc:
                gateway.cache_discard();last=exc
                print(f'[DETECT API] invalid response attempt={attempt+1}; no local fallback',flush=True)
                continue
            gateway.cache_commit()
            return result
        raise last or DetectionAPIError('Detection failed')
