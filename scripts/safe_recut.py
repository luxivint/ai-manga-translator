"""Bounded-memory, lossless preprocessing before chapter detection/OCR.

Sources are a virtual vertical strip. Only a few source images and one small
analysis window are decoded at once. No network, OCR, R2 writes or source edits.
"""
from __future__ import annotations

import bisect
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from modules.detection.reuse import window_observations, merge_observations, page_record


class RecutError(RuntimeError):
    pass


class VerticalStrip:
    def __init__(self, paths):
        self.paths = list(paths)
        self.sizes = []
        self.ends = []
        self.height = 0
        self.cache = OrderedDict()
        for path in self.paths:
            with Image.open(path) as im:
                self.sizes.append(ImageOps.exif_transpose(im).size)
            self.height += self.sizes[-1][1]
            self.ends.append(self.height)
        if not self.sizes or len({s[0] for s in self.sizes}) != 1:
            raise RecutError('Chapter image widths differ; images cannot be joined without distortion.')
        self.width = self.sizes[0][0]

    def crop(self, top, bottom):
        if not 0 <= top < bottom <= self.height:
            raise RecutError('Invalid strip crop coordinates')
        out = Image.new('RGB', (self.width, bottom - top))
        index = bisect.bisect_right(self.ends, top)
        while index < len(self.paths):
            start = self.ends[index - 1] if index else 0
            end = self.ends[index]
            if start >= bottom:
                break
            if index not in self.cache:
                with Image.open(self.paths[index]) as im:
                    self.cache[index] = ImageOps.exif_transpose(im).convert('RGB')
                while len(self.cache) > 2:
                    _, old = self.cache.popitem(last=False)
                    old.close()
            self.cache.move_to_end(index)
            a, b = max(top, start), min(bottom, end)
            out.paste(self.cache[index].crop((0, a - start, self.width, b - start)), (0, a - top))
            index += 1
        return out

    def close(self):
        for image in self.cache.values():
            image.close()
        self.cache.clear()


@dataclass
class RecutPlan:
    images: list[Path]
    cuts: list[int]
    width: int
    height: int
    digest: str
    detections: dict | None = None


def safe_bands(uniform, minimum=24):
    transitions = np.diff(np.r_[False, uniform, False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))
            if b - a >= minimum]


def choose_cuts(height, bands, target=2000, minimum=800, soft_max=3600):
    cuts = [0]
    while height - cuts[-1] > soft_max:
        start = cuts[-1]
        candidates = [(a + b) // 2 for a, b in bands if start + minimum <= (a + b) // 2 < height - minimum]
        near = [y for y in candidates if y <= start + soft_max]
        if near:
            cuts.append(min(near, key=lambda y: abs(y - start - target)))
        elif candidates:
            cuts.append(min(candidates))
        else:
            break
    return cuts + [height]



def detect_windows(strip, starts, detect, window, workers):
    """Only the coordinator reads the strip cache. At most workers image arrays live."""
    def pixels(top):
        with strip.crop(top,min(strip.height,top+window)) as crop:
            return np.asarray(crop).copy()
    if workers==1:
        for index,top in enumerate(starts):
            image=pixels(top)
            yield index,top,len(image),detect(image)
        return
    pool=ThreadPoolExecutor(max_workers=workers)
    pending={};next_index=0
    try:
        def submit(index):
            top=starts[index];image=pixels(top)
            pending[pool.submit(detect,image)]=(index,top,len(image))
        while next_index<min(workers,len(starts)):
            submit(next_index);next_index+=1
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in sorted(done,key=lambda f:pending[f][0]):
                index,top,height=pending.pop(future)
                boxes=future.result()  # a failed detection aborts before any output is written
                yield index,top,height,boxes
                if next_index<len(starts):
                    submit(next_index);next_index+=1
    finally:
        pool.shutdown(wait=True,cancel_futures=True)


def prepare(paths, workspace, detect, *, target=2000, minimum=800, soft_max=3600,
            hard_max=8000, padding=24, window=1600, stride=1000, workers=1):
    if not 0 < minimum <= target <= soft_max <= hard_max <= 16000 or not 0 < stride < window:
        raise RecutError('Invalid safe recut size settings')
    if type(workers) is not int or not 1 <= workers <= 8:
        raise RecutError("Safe recut workers must be between 1 and 8")
    strip = VerticalStrip(paths)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if any(workspace.iterdir()):
        strip.close()
        raise RecutError('Recut workspace must be empty to avoid stale image reuse')
    try:
        uniform = np.zeros(strip.height, dtype=bool)
        protected = np.zeros(strip.height, dtype=bool)
        digest = hashlib.sha256()
        for top in range(0, strip.height, 2048):
            with strip.crop(top, min(strip.height, top + 2048)) as crop:
                digest.update(crop.tobytes())
                a = np.asarray(crop, dtype=np.int16)
                span = (a.max(axis=1) - a.min(axis=1)).max(axis=1)
                std = a.std(axis=1).max(axis=1)
                uniform[top:top + len(a)] = (span <= 22) & (std <= 3.5)
        starts = sorted(set(list(range(0, max(1, strip.height - window + 1), stride)) + [max(0, strip.height - window)]))
        regions = []
        started = time.monotonic()
        by_window={}
        typed_by_window={}
        print(f'[RECUT] scanning {len(starts)} windows parallel={workers}',flush=True)
        for completed,(index,top,height,result) in enumerate(detect_windows(strip,starts,detect,window,workers),1):
            boxes,typed = window_observations(result,top,height,strip.height,strip.width)
            typed_by_window[index]=typed
            current=[]
            for box in boxes:
                x1,y1,x2,y2=map(int,box)
                y1,y2=max(top,top+y1),min(strip.height,top+height,top+y2)
                if x2>x1 and y2>y1:
                    current.append([x1,y1,x2,y2])
                    protected[max(0,y1-padding):min(strip.height,y2+padding+1)]=True
            by_window[index]=current
            if completed==1 or completed%10==0 or completed==len(starts):
                print(f'[RECUT] protect {completed}/{len(starts)} windows '
                      f'({time.monotonic()-started:.1f}s) parallel={workers}',flush=True)
        # Deterministic manifest/order regardless of API completion order.
        regions=[box for index in range(len(starts)) for box in by_window[index]]
        typed_groups = (merge_observations([item for index in range(len(starts)) for item in typed_by_window[index]])
                        if all(typed_by_window[index] is not None for index in range(len(starts))) else None)
        bands = safe_bands(uniform & ~protected)
        cuts = choose_cuts(strip.height, bands, target, minimum, soft_max)
        # Prefer gutters, but an uninterrupted illustration need not be blank.
        # Only supplement oversized spans with rows outside every detected balloon/text.
        fallback_bands = safe_bands(~protected)
        expanded = [cuts[0]]
        fallback_cuts = []
        for end in cuts[1:]:
            while end - expanded[-1] > hard_max:
                start = expanded[-1]
                candidates = []
                for a, b in fallback_bands:
                    lo, hi = max(a, start + minimum), min(b - 1, start + soft_max, end - minimum)
                    if lo <= hi:
                        candidates.append(min(max(start + target, lo), hi))
                if not candidates:
                    break
                y = min(candidates, key=lambda y: abs(y - start - target))
                expanded.append(y); fallback_cuts.append(y)
            expanded.append(end)
        cuts = expanded
        if any(b - a > hard_max for a, b in zip(cuts, cuts[1:])):
            raise RecutError(f'No safe cut within {hard_max}px. Raw images kept; inspect this chapter or disable safe recut explicitly.')
        if any(protected[y] for y in cuts[1:-1]):
            raise RecutError('Cut intersects a protected text/bubble region')
        images = []
        detections = {}
        rebuilt_digest = hashlib.sha256()
        for index, (top, bottom) in enumerate(zip(cuts, cuts[1:]), 1):
            path = workspace / f'recut-{index:05d}.png'
            with strip.crop(top, bottom) as crop:
                crop.save(path, compress_level=3)
            with Image.open(path) as saved:
                if saved.size != (strip.width, bottom - top):
                    raise RecutError('Recut output dimensions changed')
                pixels = saved.convert('RGB').tobytes()
                rebuilt_digest.update(pixels)
            images.append(path)
            if typed_groups is not None:
                detections[path.name]=page_record(typed_groups,top,bottom,strip.width,hashlib.sha256(pixels).hexdigest())
        if rebuilt_digest.hexdigest() != digest.hexdigest():
            raise RecutError('Recut images did not preserve source pixels')
        manifest = dict(width=strip.width, height=strip.height, source_files=[p.name for p in paths],
                        source_ends=strip.ends, cuts=cuts, protected_regions=regions, safe_bands=bands,
                        fallback_cuts=fallback_cuts,
                        source_sha256=digest.hexdigest(), pixel_exact=True,
                        oversized_heights=[b - a for a, b in zip(cuts, cuts[1:]) if b - a > soft_max])
        if typed_groups is not None:
            (workspace / 'recut-detection.json').write_text(json.dumps(detections,indent=2),encoding='utf8')
            ready=sum(record['reusable'] for record in detections.values())
            print(f'[RECUT] detection reuse eligible={ready}/{len(images)}; fallback={len(images)-ready}',flush=True)
        (workspace / 'recut-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'[RECUT] {len(paths)} source pages -> {len(images)} safe pieces; pixel verification OK; '
              f'max height={max(b - a for a, b in zip(cuts, cuts[1:]))}', flush=True)
        return RecutPlan(images, cuts, strip.width, strip.height, digest.hexdigest(),
                         detections if typed_groups is not None else None)
    finally:
        strip.close()
