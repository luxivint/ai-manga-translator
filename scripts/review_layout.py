"""Replay real checkpoint blocks through the production cleaner/typesetter for visual QA.

Offline by default. Never uploads or changes source/checkpoint files.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))


def fixtures(log, checkpoints):
    from scripts.replay_chapter import parse_dump_lines
    from scripts.page_checkpoint import _decode
    from modules.utils.textblock import TextBlock
    entries = parse_dump_lines(Path(log).read_text(encoding='utf8').splitlines())
    index = defaultdict(set)
    def key(text, trans, box):
        return text.strip(), trans.strip(), tuple(int(x) for x in box)
    for entry in entries:
        if entry.get('page'):
            index[key(entry['text'], entry['trans'], entry['box'])].add(entry['page'])
    result = {}
    for path in Path(checkpoints).glob('*.json'):
        row = json.loads(path.read_text(encoding='utf8'))
        blocks = []
        votes = Counter()
        for attrs in row['blocks']:
            block = TextBlock()
            block.__dict__.update(_decode(attrs))
            blocks.append(block)
            for page in index.get(key(block.text, block.translation, block.xyxy), []):
                if block.translation:
                    votes[page] += 1
        if not votes:
            continue
        page, count = votes.most_common(1)[0]
        if len(votes) > 1 and votes.most_common(2)[1][1] == count:
            continue
        if page not in result or count > result[page]['matches']:
            result[page] = dict(blocks=blocks, marker=path, image=path.with_suffix('.png'), matches=count)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pages', type=int, nargs='*')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--reocr', action='store_true', help='Use the configured OCR API; never publishes')
    parser.add_argument('--blocks-from', type=Path, help='Replay saved reviewed blocks without API calls')
    ns = parser.parse_args()
    # The desktop runner redirects LOCALAPPDATA; use the installed model cache
    # read-only instead of attempting a second download in that redirected path.
    installed_data = Path.home() / 'AppData/Local'
    if (installed_data / 'MangaTranslationWorker/models').is_dir():
        os.environ['LOCALAPPDATA'] = str(installed_data)
    # Apply the real worker's non-secret runtime overrides, without printing .env.
    from scripts.r2_storage import load_env
    load_env(ROOT / '.env')
    services = json.loads((ROOT.parent / 'trend-tray/services.json').read_text(encoding='utf-8-sig'))
    os.environ.update(services['services']['translation'].get('env', {}))
    os.environ['RENDER_DEBUG'] = 'true'
    rows = fixtures(ns.log, ns.checkpoints)
    if ns.list:
        for page, row in sorted(rows.items()):
            for block in row['blocks']:
                if block.translation:
                    print(json.dumps(dict(page=page, box=list(map(int, block.xyxy)),
                        bubble=None if block.bubble_xyxy is None else list(map(int, block.bubble_xyxy)),
                        source=block.text, translation=block.translation), ensure_ascii=False))
        return
    from scripts.replay_chapter import block_network
    if not ns.reocr:
        block_network()
    from scripts import local_batch as lb
    from scripts.runtime_config import configure_cleaner
    from PySide6 import QtWidgets
    import numpy as np
    from PIL import Image
    configure_cleaner()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    saved_argv = sys.argv
    try:
        sys.argv = ['local_batch.py', '--output-format', 'png', '--stamp-enabled', 'false']
        args = lb.parse_args()
    finally:
        sys.argv = saved_argv
    args.resolved_font = lb.resolve_font(args)
    main = lb.LocalMain(lb.LocalSettings(args), args.source_lang, args.target_lang)
    ns.output.mkdir(parents=True, exist_ok=True)
    real_build = lb.build_text_items_state
    def build(blocks, image, cleaned, args, main):
        cleaned, items = real_build(blocks, image, cleaned, args, main)
        (ns.output / (current + '.items.json')).write_text(json.dumps(items, ensure_ascii=False,
            indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)), encoding='utf8')
        return cleaned, items
    lb.build_text_items_state = build
    prepared = []
    for number in ns.pages or []:
        current = f'recut-{number:05d}.png'
        row = rows[current]
        blocks = copy.deepcopy(row['blocks'])
        with Image.open(row['image']) as src:
            image = np.asarray(src.convert('RGB')).copy()
        path = ns.input / current
        if ns.blocks_from:
            from scripts.page_checkpoint import _decode
            from modules.utils.textblock import TextBlock
            blocks = []
            for attrs in json.loads((ns.blocks_from / (current + '.blocks.json')).read_text(encoding='utf8')):
                block = TextBlock(); block.__dict__.update(_decode(attrs)); blocks.append(block)
        if ns.reocr:
            from modules.detection.processor import TextBlockDetector
            from modules.ocr.processor import OCRProcessor
            from modules.utils import gateway
            from modules.detection.reuse import pixel_hash
            record = json.loads((ns.input / 'recut-detection.json').read_text(encoding='utf8'))[current]
            detector = TextBlockDetector(lb.LocalSettings(args))
            if record['reusable'] and pixel_hash(image) == record['pixel_sha256']:
                blocks = detector.detect_from_regions(image, record['texts'], record['bubbles'])
            else:
                blocks = detector.detect(image)
            ocr = OCRProcessor(); ocr.initialize(main, args.source_lang); ocr.process(image, blocks)
            blocks = lb.split_merged_blocks_ocr(image, blocks, ocr)
            blocks = lb.prune_ocr_blocks(lb.drop_non_english_ocr_blocks(blocks), image)
            old = {' '.join(b.text.casefold().split()): b for b in row['blocks'] if b.translation}
            needs_translation = []
            for block in blocks:
                previous = old.get(' '.join(block.text.casefold().split()))
                if previous:
                    block.translation, block.style = previous.translation, previous.style
                elif block.text:
                    needs_translation.append(block)
            if needs_translation:
                lb._batch_translate_all([(path, image, needs_translation)], args)
        from scripts.page_checkpoint import _encode
        (ns.output / (current + '.blocks.json')).write_text(json.dumps([_encode(vars(b)) for b in blocks],
            ensure_ascii=False, indent=2), encoding='utf8')
        _, _, _, blocks, cleaned = lb._stage_clean((number-1, path, image, blocks, args))
        lb._do_render_and_save(number-1, 169, path, image, blocks, cleaned, ns.output, args, main)
    app.quit()


if __name__ == '__main__':
    main()
