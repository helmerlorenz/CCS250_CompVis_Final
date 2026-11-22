"""Shared image I/O for the three method modules."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from app.experiments import evaluate_model, get_network
from app.inference import tiled_forward


def run_method(name: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run the pretrained {name} denoiser")
    parser.add_argument("--input", type=Path, help="Noisy RGB image")
    parser.add_argument("--output", type=Path, help="Denoised PNG path")
    parser.add_argument("--checkpoint", type=Path, help="Optional fine-tuned checkpoint")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate all 500 test pairs")
    args = parser.parse_args()
    if args.evaluate:
        evaluate_model(name, checkpoint=args.checkpoint,
                       output_dir=Path("HeraldFolder") / "test")
        return
    if args.input is None or args.output is None:
        parser.error("supply --input and --output, or use --evaluate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network, _ = get_network(name, device, args.checkpoint)
    network.eval()
    with Image.open(args.input) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255.)
    tensor = tensor.permute(2, 0, 1)[None].to(device)
    with torch.inference_mode():
        result = tiled_forward(network, tensor, name, tile=256)
    pixels = np.clip(result[0].permute(1, 2, 0).cpu().numpy() * 255., 0, 255)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.round().astype(np.uint8)).save(args.output)
    print(args.output)
