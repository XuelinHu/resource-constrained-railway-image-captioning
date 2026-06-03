from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def apply_dynamic_quantization(model: torch.nn.Module) -> torch.nn.Module:
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )


def export_onnx_placeholder(model: torch.nn.Module, output_path: str | Path, example_batch: Any) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "ONNX export needs a fixed model/input signature. Add an example pixel_values batch "
        f"and decoder inputs before exporting to {path}."
    )
