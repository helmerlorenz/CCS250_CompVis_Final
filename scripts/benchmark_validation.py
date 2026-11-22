"""Measure pretrained denoisers on validation crops without touching the test split.

Results are stored under HeraldFolder/validation. Set --limit-per-source 0
for the complete 500-image validation set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from app.data import DenoiseDataset
from app.experiments import get_network
from app.models import NAMES, ROOT


def score(clean: np.ndarray, image: np.ndarray) -> tuple[float, float]:
    return (
        float(peak_signal_noise_ratio(clean, image, data_range=1)),
        float(structural_similarity(clean, image, data_range=1, channel_axis=2)),
    )


def run(model_names: tuple[str, ...], limit_per_source: int, patch: int,
        checkpoint: Path | None = None, tag: str | None = None) -> None:
    torch.set_num_threads(4)
    manifest = ROOT / "data/splits.csv"
    ds = DenoiseDataset(manifest, "val", patch=patch)
    indices: list[tuple[int, str]] = []
    seen: dict[str, int] = {}
    for i, row in enumerate(ds.files):
        source = row["source"]
        if limit_per_source and seen.get(source, 0) >= limit_per_source:
            continue
        indices.append((i, source))
        seen[source] = seen.get(source, 0) + 1
    destination = ROOT / "HeraldFolder" / "validation"
    destination.mkdir(parents=True, exist_ok=True)
    for name in model_names:
        net, source_checkpoint = get_network(name, "cpu", checkpoint)
        net.eval()
        rows = []
        with torch.inference_mode():
            for count, (i, source) in enumerate(indices, 1):
                noisy, clean = ds[i]
                ground = clean.permute(1, 2, 0).numpy()
                input_image = noisy.permute(1, 2, 0).numpy()
                start = time.perf_counter()
                output = net(noisy[None]).clamp(0, 1)[0].permute(1, 2, 0).numpy()
                runtime_ms = (time.perf_counter() - start) * 1000
                input_psnr, input_ssim = score(ground, input_image)
                output_psnr, output_ssim = score(ground, output)
                rows.append(dict(index=i, source=source, noisy_psnr_db=input_psnr,
                                 noisy_ssim=input_ssim, model_psnr_db=output_psnr,
                                 model_ssim=output_ssim, runtime_ms=runtime_ms))
                if count % 25 == 0 or count == len(indices):
                    print(f"{name}: {count}/{len(indices)} validation crops", flush=True)
        label = tag or name
        with (destination / f"{label}_validation.csv").open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        summary = dict(model=name, split="val", number_of_images=len(rows),
                       limit_per_source=limit_per_source, patch=patch, device="cpu",
                       checkpoint=str(source_checkpoint),
                       checkpoint_sha256=hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
                       manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        for key in ("noisy_psnr_db", "noisy_ssim", "model_psnr_db", "model_ssim", "runtime_ms"):
            summary[key] = round(float(np.mean([row[key] for row in rows])), 5)
        summary["by_source"] = {
            source: {key: round(float(np.mean([row[key] for row in rows if row["source"] == source])), 5)
                     for key in ("noisy_psnr_db", "model_psnr_db", "model_ssim")}
            for source in sorted(seen)
        }
        (destination / f"{label}_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
        print(json.dumps(summary, indent=2), flush=True)
        del net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=(*NAMES, "all"), default="all")
    parser.add_argument("--limit-per-source", type=int, default=0)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tag", help="Output filename prefix for this checkpoint")
    args = parser.parse_args()
    if args.limit_per_source < 0 or args.patch < 8:
        parser.error("limit-per-source must be nonnegative and patch must be at least 8")
    if args.checkpoint and args.model == "all":
        parser.error("--checkpoint requires one --model")
    names = NAMES if args.model == "all" else (args.model,)
    run(names, args.limit_per_source, args.patch, args.checkpoint, args.tag)
