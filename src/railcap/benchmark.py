from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

from railcap.config import load_config
from railcap.data.dataset import RailwayCaptionDataset
from railcap.train import collate_batch
from railcap.utils.runtime import cuda_memory_mb, pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark caption generation latency.")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint directory.")
    parser.add_argument("--manifest", type=Path, help="Override benchmark manifest.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raw = cfg.raw

    checkpoint = args.checkpoint or (cfg.output_dir / "final")
    manifest_path = args.manifest or Path(raw["data"]["test_manifest"])
    device = pick_device()

    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint).to(device)
    model.eval()

    dataset = RailwayCaptionDataset(
        manifest_path,
        image_root=raw["data"].get("image_root", "."),
        image_processor=image_processor,
        tokenizer=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(raw.get("eval", {}).get("batch_size", 8)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    generation_kwargs = dict(raw.get("model", {}).get("generation", {}))
    timings: list[float] = []
    images = 0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= args.warmup + args.steps:
                break
            pixel_values = batch["pixel_values"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            generated_ids = model.generate(pixel_values=pixel_values, **generation_kwargs)
            tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step >= args.warmup:
                timings.append(elapsed)
                images += pixel_values.shape[0]

    total_time = sum(timings)
    result = {
        "device": str(device),
        "steps": len(timings),
        "images": images,
        "batch_size": int(raw.get("eval", {}).get("batch_size", 8)),
        "latency_ms_per_batch": (total_time / len(timings) * 1000) if timings else 0.0,
        "latency_ms_per_image": (total_time / images * 1000) if images else 0.0,
        "throughput_img_s": (images / total_time) if total_time else 0.0,
        "cuda_max_memory_mb": cuda_memory_mb(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
