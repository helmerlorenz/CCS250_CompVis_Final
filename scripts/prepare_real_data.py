"""Download, extract, inventory, and split the paired SIDD/PolyU data.

Run from the project root: python -m scripts.prepare_real_data
Downloads are resumable. Existing split manifests are never overwritten.
"""
from __future__ import annotations

import csv
import os
import time
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from .download_archives import ROOT, download
from .prepare_mixed_dataset import prepare

DATA = ROOT / 'data'
SOURCES = {
    'PolyU': ('PolyU-Real-World-Noisy-Images-Dataset', 'train_sub', '.jpg', 4000),
    'SIDD': ('SIDD', 'train_sub512', '.png', 16000),
}


def extract(dataset: str, archive: Path) -> Path:
    folder, subset, suffix, pair_count = SOURCES[dataset]
    destination = (DATA / 'raw' / dataset).resolve()
    base = destination / folder / subset
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive) as file:
        entries = [entry for entry in file.infolist() if not entry.is_dir()]
        if len(entries) != pair_count * 2:
            raise ValueError(f'{dataset}: expected {pair_count * 2} image files, found {len(entries)}')
        for entry in entries:
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination) or not entry.filename.lower().endswith(suffix):
                raise ValueError(f'{dataset}: unsafe or unexpected archive entry: {entry.filename}')
            if target.is_file() and target.stat().st_size == entry.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.extracting')
            with file.open(entry) as source, temporary.open('wb') as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
            os.replace(temporary, target)
            extracted += 1
            if extracted % 2000 == 0:
                print(f'{dataset}: extracted {extracted} new images', flush=True)
    if not (base / 'GT').is_dir() or not (base / 'LQ').is_dir():
        raise FileNotFoundError(f'{dataset}: missing paired GT/LQ folders under {base}')
    print(f'{dataset}: extracted {extracted} new images; paired folder {base}', flush=True)
    return base


def inventory(dataset: str, base: Path) -> Path:
    _, _, suffix, pair_count = SOURCES[dataset]
    output = DATA / 'inventory' / f'{dataset}.csv'
    if output.exists():
        with output.open(newline='', encoding='utf8') as file:
            rows = list(csv.DictReader(file))
        if len(rows) != pair_count:
            raise ValueError(f'{dataset}: existing inventory has {len(rows)} rows, expected {pair_count}')
        print(f'{dataset}: using existing inventory with {len(rows)} pairs', flush=True)
        return output
    clean_files = sorted((base / 'GT').glob(f'*{suffix}'))
    if len(clean_files) != pair_count:
        raise ValueError(f'{dataset}: expected {pair_count} clean images, found {len(clean_files)}')
    rows = []
    for index, clean in enumerate(clean_files, 1):
        noisy = base / 'LQ' / clean.name
        if not noisy.is_file():
            raise FileNotFoundError(f'{dataset}: missing noisy mate for {clean}')
        with Image.open(clean) as a, Image.open(noisy) as b:
            if a.size != b.size:
                raise ValueError(f'{dataset}: dimension mismatch for {clean.name}')
        stem = clean.stem.rsplit('_s', 1)[0]
        group = stem.split('_')[0] if dataset == 'SIDD' else stem
        rows.append({'noisy_path': noisy.relative_to(DATA).as_posix(),
                     'clean_path': clean.relative_to(DATA).as_posix(),
                     'group_id': group})
        if index % 2000 == 0:
            print(f'{dataset}: inspected {index}/{pair_count} pairs', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf8') as file:
        writer = csv.DictWriter(file, fieldnames=['noisy_path', 'clean_path', 'group_id'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'{dataset}: inventory has {len(rows)} pairs in {len({r["group_id"] for r in rows})} scenes', flush=True)
    return output


def main():
    for dataset in ('PolyU', 'SIDD'):
        for attempt in range(10):
            try:
                archive = download(dataset, workers=16)
                break
            except Exception as error:
                if attempt == 9:
                    raise
                print(f'{dataset}: transfer interrupted ({error}); retrying in 60 seconds', flush=True)
                time.sleep(60)
        base = extract(dataset, archive)
        inventory(dataset, base)
    manifest = DATA / 'splits.csv'
    if manifest.exists():
        print(f'Existing manifest preserved: {manifest}', flush=True)
    else:
        manifest = prepare(seed=42)
    with manifest.open(newline='', encoding='utf8') as file:
        rows = list(csv.DictReader(file))
    print('Final source/split counts:', dict(Counter((r['source'], r['split']) for r in rows)), flush=True)
    print('Data preparation complete:', manifest, flush=True)


if __name__ == '__main__':
    main()
