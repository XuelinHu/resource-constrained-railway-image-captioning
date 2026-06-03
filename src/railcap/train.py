from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from railcap.config import load_config
from railcap.data.dataset import RailwayCaptionDataset
from railcap.models.modeling import build_model, build_processors, count_parameters
from railcap.utils.runtime import ensure_dir, pick_device, set_seed


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "samples": [item["sample"] for item in batch],
        "captions": [item["caption"] for item in batch],
    }
    tensor_keys = ["pixel_values", "labels", "decoder_attention_mask"]
    for key in tensor_keys:
        if key in batch[0]:
            output[key] = torch.stack([item[key] for item in batch])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train railway image captioning model.")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raw = cfg.raw
    set_seed(int(raw.get("seed", 42)))
    output_dir = ensure_dir(cfg.output_dir)

    image_processor, tokenizer = build_processors(raw)
    model = build_model(raw, tokenizer)
    device = pick_device()
    model.to(device)

    params = count_parameters(model)
    print(f"parameters total={params['total']:,} trainable={params['trainable']:,}")

    data_cfg = raw["data"]
    train_dataset = RailwayCaptionDataset(
        data_cfg["train_manifest"],
        image_root=data_cfg.get("image_root", "."),
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_caption_length=int(data_cfg.get("max_caption_length", 96)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(raw["train"].get("batch_size", 8)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        collate_fn=collate_batch,
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(raw["train"].get("learning_rate", 5e-5)),
        weight_decay=float(raw["train"].get("weight_decay", 0.01)),
    )

    use_amp = bool(raw["train"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_accum_steps = int(raw["train"].get("grad_accum_steps", 1))
    log_every = int(raw["train"].get("log_every", 20))

    model.train()
    global_step = 0
    for epoch in range(int(raw["train"].get("epochs", 3))):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress, start=1):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            if step % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if global_step and global_step % log_every == 0:
                progress.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}")

        if raw["train"].get("save_every_epoch", True):
            checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch + 1}"
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            image_processor.save_pretrained(checkpoint_dir)

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    image_processor.save_pretrained(final_dir)
    print(f"saved model to {final_dir}")


if __name__ == "__main__":
    main()
