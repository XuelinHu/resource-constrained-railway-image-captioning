from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

from railcap.config import load_config
from railcap.data.dataset import RailwayCaptionDataset
from railcap.train import collate_batch
from railcap.utils.runtime import ensure_dir, pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate railway image captions.")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint directory.")
    parser.add_argument("--manifest", type=Path, help="Override manifest path.")
    parser.add_argument("--output", type=Path, help="Output JSONL path.")
    return parser.parse_args()


def _generation_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("model", {}).get("generation", {}))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raw = cfg.raw

    checkpoint = args.checkpoint or (cfg.output_dir / "final")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    manifest_path = args.manifest or Path(raw["data"]["test_manifest"])
    output_path = args.output or (cfg.output_dir / "predictions.jsonl")
    ensure_dir(output_path.parent)

    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
    device = pick_device()
    model.to(device)
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
        num_workers=int(raw["data"].get("num_workers", 4)),
        collate_fn=collate_batch,
    )

    generation_kwargs = _generation_kwargs(raw)
    with output_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            pixel_values = batch["pixel_values"].to(device)
            generated_ids = model.generate(pixel_values=pixel_values, **generation_kwargs)
            captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for sample, prediction in zip(batch["samples"], captions, strict=True):
                record = {
                    "image": sample.image,
                    "prediction": prediction.strip(),
                    "reference": sample.caption,
                    "equipment": sample.equipment,
                    "faults": sample.faults,
                    "risks": sample.risks,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
