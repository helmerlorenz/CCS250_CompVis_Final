"""Prepare exact SIDD/PolyU splits from curated paired-image CSV inventories.
Run from project root: python -m scripts.prepare_mixed_dataset
Each inventory: data/inventory/{SIDD,PolyU}.csv, headers
  noisy_path,clean_path,group_id
Paths must be relative to data/, and pair images must already be aligned.
Groups represent original scenes; prevents scene leakage across splits.
"""
from pathlib import Path
import csv
import hashlib
import random
from collections import Counter, defaultdict
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
QUOTAS = {
    "SIDD": {
        "train": 2800,
        "val": 350,
        "test": 350
    },
    "PolyU": {
        "train": 1200,
        "val": 150,
        "test": 150
    }
}

IMG_EXT={'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}


def read_inventory(source):
    file=DATA/'inventory'/f'{source}.csv'
    if not file.exists():
        raise FileNotFoundError(f'Missing {file}. Prepare real noisy/clean pairs first, then list them in this CSV.')
    with file.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not {'noisy_path','clean_path','group_id'} <= set(reader.fieldnames or []):
            raise ValueError(f'{file}: required columns noisy_path,clean_path,group_id')
        rows=list(reader)
    if not rows: raise ValueError(f'{file}: empty inventory')
    seen=set()
    for row in rows:
        group=row['group_id'].strip()
        if not group: raise ValueError(f'{file}: group_id cannot be blank')
        for col in ('noisy_path','clean_path'):
            rel=row[col].strip()
            p=(DATA/rel).resolve()
            if not p.is_relative_to(DATA.resolve()) or p.suffix.lower() not in IMG_EXT or not p.is_file():
                raise ValueError(f'{source}: invalid or missing {col}: {rel}')
            row[col]=p.relative_to(DATA.resolve()).as_posix()
        if row['noisy_path'] in seen:raise ValueError(f'{source}: repeated noisy sample {row["noisy_path"]}')
        seen.add(row['noisy_path'])
        with Image.open(DATA/row['noisy_path']) as a, Image.open(DATA/row['clean_path']) as b:
            if a.size!=b.size: raise ValueError(f'Dimension mismatch: {row["noisy_path"]} vs {row["clean_path"]}')
            if min(a.size)<128:raise ValueError(f'Image smaller than 128 pixels: {row["noisy_path"]}')
            a.load(); b.load()
        row['group_id']=group
    return rows


def allocate(source, rows, seed=42):
    """Split scenes first, then sample pairs evenly within each partition."""
    groups=defaultdict(list)
    for row in rows:groups[row['group_id']].append(row)
    pool=list(groups.items())
    rng=random.Random(seed+sum(map(ord,source)))
    targets=QUOTAS[source]
    count_needed=sum(targets.values())
    if len(rows)<count_needed:raise ValueError(f'{source}: {len(rows)} pairs; need at least {count_needed}')
    if len(pool)<sum(value>0 for value in targets.values()):
        raise ValueError(f'{source}: too few independent scenes for three splits')
    n_val=max(1,round(len(pool)*targets['val']/count_needed))
    n_test=max(1,round(len(pool)*targets['test']/count_needed))
    n_train=len(pool)-n_val-n_test
    if n_train<1:raise ValueError(f'{source}: too few scenes for split proportions')
    partitions=None
    for _ in range(256):
        candidate=pool.copy()
        rng.shuffle(candidate)
        proposed={'train':candidate[:n_train],
                  'val':candidate[n_train:n_train+n_val],
                  'test':candidate[n_train+n_val:]}
        if all(sum(len(items) for _,items in proposed[split])>=target
               for split,target in targets.items()):
            partitions=proposed
            break
    if partitions is None:
        raise ValueError(f'{source}: no scene assignment met exact quotas after 256 attempts. '
                         'Review scene grouping and sample counts; no splits were written.')
    chosen=[]
    for split,target in targets.items():
        buckets=[]
        for _,items in partitions[split]:
            shuffled=items.copy()
            rng.shuffle(shuffled)
            buckets.append(shuffled)
        count=0
        while count<target:
            for bucket in buckets:
                if bucket and count<target:
                    chosen.append({**bucket.pop(),'source':source,'split':split})
                    count+=1
    return chosen


def prepare(output=None,seed=42):
    output=Path(output or DATA/'splits.csv').resolve()
    if output.exists():raise FileExistsError(f'{output} already exists. Back it up before regenerating the split.')
    all_rows=[]
    for source in QUOTAS:
        rows=read_inventory(source)
        selected=allocate(source,rows,seed)
        all_rows.extend(selected)
        print(source,dict(Counter(r['split'] for r in selected)))
    assert len(all_rows)==sum(sum(parts.values()) for parts in QUOTAS.values())
    noisy_paths=[r['noisy_path'] for r in all_rows]
    if len(set(noisy_paths)) != len(noisy_paths):
        raise ValueError('The same noisy image appears in multiple inventories. No splits were written.')
    clean_splits=defaultdict(set)
    for r in all_rows:clean_splits[r['clean_path']].add(r['split'])
    if any(len(splits)>1 for splits in clean_splits.values()):
        raise ValueError('A clean image crosses splits. No splits were written.')
    # No file can cross source partitions, and no group can leak across splits.
    groups=defaultdict(set)
    for r in all_rows:groups[(r['source'],r['group_id'])].add(r['split'])
    assert all(len(s)==1 for s in groups.values())
    output.parent.mkdir(exist_ok=True,parents=True)
    with output.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['source','split','noisy_path','clean_path','group_id'])
        w.writeheader();w.writerows(all_rows)
    print(f'Created {output} with {len(all_rows)} paired samples')
    print('Manifest SHA256:',hashlib.sha256(output.read_bytes()).hexdigest())
    return output

if __name__=='__main__':prepare()
