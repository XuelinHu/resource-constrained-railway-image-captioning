from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)


def build_processors(config: dict[str, Any]) -> tuple[Any, Any]:
    model_cfg = config["model"]
    if model_cfg.get("pretrained_vision_encoder_decoder"):
        name = model_cfg["pretrained_vision_encoder_decoder"]
        image_processor = AutoImageProcessor.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
    else:
        image_processor = AutoImageProcessor.from_pretrained(model_cfg["vision_encoder"])
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["text_decoder"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return image_processor, tokenizer


def build_model(config: dict[str, Any], tokenizer: Any) -> VisionEncoderDecoderModel:
    model_cfg = config["model"]
    pretrained = model_cfg.get("pretrained_vision_encoder_decoder")

    if pretrained:
        model = VisionEncoderDecoderModel.from_pretrained(pretrained)
    else:
        decoder_name = model_cfg["text_decoder"]
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            model_cfg["vision_encoder"],
            decoder_name,
        )

    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    generation_cfg = model_cfg.get("generation", {})
    for key, value in generation_cfg.items():
        setattr(model.generation_config, key, value)

    if model_cfg.get("freeze_vision_encoder", False):
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    return model


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def empty_vision_encoder_decoder_config() -> VisionEncoderDecoderConfig:
    return VisionEncoderDecoderConfig()
