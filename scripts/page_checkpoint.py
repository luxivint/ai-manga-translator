"""Atomic, content-verified page checkpoints. JSON/PNG only; never unpickle data."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
from PIL import Image
from modules.utils.textblock import TextBlock


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('Object arrays cannot be checkpointed')
        return {'_array': value.tolist(), 'dtype': str(value.dtype), 'shape': list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {'_tuple': [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported checkpoint field: {type(value).__name__}')


def _decode(value):
    if isinstance(value, dict):
        if '_array' in value:
            dtype = np.dtype(value['dtype'])
            if dtype.hasobject:
                raise ValueError('Object arrays cannot be restored')
            result = np.asarray(value['_array'], dtype=dtype)
            return result.reshape(value['shape']) if 'shape' in value else result
        if '_tuple' in value:
            return tuple(_decode(v) for v in value['_tuple'])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


class PageCheckpoints:
    def __init__(self, directory):
        self.directory = Path(directory)

    def _paths(self, stage, source, fingerprint):
        key = hashlib.sha256(f'{stage}\0{Path(source).name}\0{fingerprint}'.encode()).hexdigest()
        return self.directory / (key + '.json'), self.directory / (key + '.png')

    def load(self, stage, source, fingerprint):
        marker, image_path = self._paths(stage, source, fingerprint)
        try:
            row = json.loads(marker.read_text(encoding='utf8'))
            if row['version'] != 1 or row['fingerprint'] != fingerprint:
                return None
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != row['image_sha256']:
                return None
            with Image.open(image_path) as image:
                pixels = np.asarray(image.convert('RGB')).copy()
            blocks = []
            for attrs in row['blocks']:
                block = TextBlock()
                block.__dict__.update(_decode(attrs))
                blocks.append(block)
            return pixels, blocks, row['meta']
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def signature(self, stage, source, fingerprint):
        """Bind chapter context to accepted OCR metadata without decoding every image."""
        marker, _ = self._paths(stage, source, fingerprint)
        try:
            data = marker.read_bytes()
            row = json.loads(data)
            if row['version'] == 1 and row['fingerprint'] == fingerprint and not row['meta'].get('ocr_failed'):
                return hashlib.sha256(data).hexdigest()
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return None

    def save(self, stage, source, fingerprint, image, blocks, meta):
        # A failed OCR or untranslated fallback must never become a checkpoint.
        if meta.get('ocr_failed') or not isinstance(image, np.ndarray):
            return
        marker, image_path = self._paths(stage, source, fingerprint)
        row = {'version': 1, 'fingerprint': fingerprint, 'meta': meta,
               'source': Path(source).name, 'stage': stage,
               'blocks': [_encode(vars(block)) for block in blocks]}
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary_image = image_path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        Image.fromarray(image).save(temporary_image, format='PNG', compress_level=1)
        row['image_sha256'] = hashlib.sha256(temporary_image.read_bytes()).hexdigest()
        os.replace(temporary_image, image_path)
        temporary_marker = marker.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        temporary_marker.write_text(json.dumps(row, ensure_ascii=False), encoding='utf8')
        os.replace(temporary_marker, marker)
