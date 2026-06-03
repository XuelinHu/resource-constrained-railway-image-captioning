from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class CaptionSample:
    image: str
    caption: str
    equipment: list[str] = field(default_factory=list)
    faults: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    split: Split | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, item: dict[str, Any]) -> "CaptionSample":
        required = {"image", "caption"}
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        split = item.get("split")
        if split is not None and split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {split!r}; expected train, val or test")

        known = {"image", "caption", "equipment", "faults", "risks", "split"}
        meta = {key: value for key, value in item.items() if key not in known}
        return cls(
            image=str(item["image"]),
            caption=str(item["caption"]),
            equipment=list(item.get("equipment") or []),
            faults=list(item.get("faults") or []),
            risks=list(item.get("risks") or []),
            split=split,
            meta=meta,
        )

    def image_path(self, image_root: str | Path = ".") -> Path:
        path = Path(self.image)
        if path.is_absolute():
            return path
        return Path(image_root) / path
