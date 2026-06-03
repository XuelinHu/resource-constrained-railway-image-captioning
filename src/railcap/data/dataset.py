from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from railcap.data.schema import CaptionSample


def load_manifest(path: str | Path) -> list[CaptionSample]:
    manifest_path = Path(path)
    samples: list[CaptionSample] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(CaptionSample.from_json(json.loads(line)))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{manifest_path}:{line_no}: {exc}") from exc
    return samples


class RailwayCaptionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path = ".",
        image_processor: Any | None = None,
        tokenizer: Any | None = None,
        max_caption_length: int = 96,
    ) -> None:
        self.samples = load_manifest(manifest_path)
        self.image_root = Path(image_root)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_caption_length = max_caption_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = sample.image_path(self.image_root)
        image = Image.open(image_path).convert("RGB")

        item: dict[str, Any] = {
            "sample": sample,
            "image_path": str(image_path),
            "caption": sample.caption,
            "equipment": sample.equipment,
            "faults": sample.faults,
            "risks": sample.risks,
        }

        if self.image_processor is not None:
            item.update(self.image_processor(images=image, return_tensors="pt"))

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                sample.caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_caption_length,
                return_tensors="pt",
            )
            item["labels"] = encoded["input_ids"].squeeze(0)
            if "attention_mask" in encoded:
                item["decoder_attention_mask"] = encoded["attention_mask"].squeeze(0)

        if "pixel_values" in item:
            item["pixel_values"] = item["pixel_values"].squeeze(0)

        return item
