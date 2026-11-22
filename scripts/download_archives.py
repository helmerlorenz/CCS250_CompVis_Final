"""Resume and verify the two paired-image archives used by notebook 01.

Run from the project root: python -m scripts.download_archives PolyU
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data' / 'downloads'
BASE = 'https://huggingface.co/datasets/VCLab-PolyU/GGT-100K/resolve/main/Existing-dataset/'
ARCHIVES = {
    'PolyU': (
        'PolyU-Real-World-Noisy-Images-Dataset.zip',
        399_667_479,
        'b6f0b2fb72e42659bf5b24d4dbbd8ccf75bfaca00ebf62a7803ae75d923ad48c',
    ),
    'SIDD': (
        'SIDD.zip',
        13_826_569_481,
        'a61a2c6816a32ee467333da597bef85c60b270e9adae7be76fb792341d16926d',
    ),
}
CHUNK = 16 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, part: Path, index: int, size: int) -> int:
    start = index * CHUNK
    end = min(start + CHUNK, size) - 1
    for attempt in range(5):
        try:
            response = requests.get(
                url,
                headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                timeout=(30, 120),
            )
            response.raise_for_status()
            expected_range = f'bytes {start}-{end}/{size}'
            if response.status_code != 206 or response.headers.get('Content-Range') != expected_range:
                raise OSError(f'Unexpected range response for chunk {index}')
            if len(response.content) != end - start + 1:
                raise OSError(f'Incomplete chunk {index}')
            with part.open('r+b') as file:
                file.seek(start)
                file.write(response.content)
            return index
        except (requests.RequestException, OSError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError('unreachable')


def download(dataset: str, workers: int = 8) -> Path:
    name, size, expected_hash = ARCHIVES[dataset]
    DEST.mkdir(parents=True, exist_ok=True)
    final = DEST / name
    if final.is_file():
        if final.stat().st_size == size and sha256(final) == expected_hash:
            print(f'Already verified: {final}', flush=True)
            return final
        raise ValueError(f'Existing archive failed size/SHA-256 check: {final}')
    part = DEST / f'{name}.part'
    done_file = DEST / f'{name}.done'
    if part.exists() and part.stat().st_size != size:
        raise ValueError(f'Partial archive has wrong size: {part}')
    if not part.exists():
        with part.open('wb') as file:
            file.truncate(size)
    completed = set()
    if done_file.exists():
        completed = {int(line) for line in done_file.read_text(encoding='ascii').splitlines() if line}
    total = (size + CHUNK - 1) // CHUNK
    pending = [i for i in range(total) if i not in completed]
    print(f'{dataset}: {len(pending)}/{total} chunks to download ({size / 1e9:.2f} GB)', flush=True)
    url = BASE + name
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, url, part, i, size) for i in pending]
        for future in as_completed(futures):
            index = future.result()
            with done_file.open('a', encoding='ascii') as file:
                file.write(f'{index}\n')
            completed.add(index)
            if len(completed) % 10 == 0 or len(completed) == total:
                print(f'{dataset}: {len(completed)}/{total} chunks', flush=True)
    if sha256(part) != expected_hash:
        raise ValueError(f'{dataset} SHA-256 mismatch; archive was not promoted to {final}')
    os.replace(part, final)
    done_file.unlink()
    print(f'Verified SHA-256: {final}', flush=True)
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=ARCHIVES)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error('--workers must be from 1 to 16')
    download(args.dataset, args.workers)
