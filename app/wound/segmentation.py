"""Local helpers reconstructed for the supplied single-pass inference CLI.

The original helper module was not supplied. Use longest-edge resizing,
centered black letterboxing, and ImageNet normalization. Undo the letterbox
on logits before applying sigmoid/threshold, without cleanup or ROI passes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

PREPROCESSING_POLICY = "reconstructed_centered_black_letterbox_imagenet_v1"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class Geometry:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    left: int
    top: int


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {value}")
    return torch.device(value)


def _compatible_weights(state, target_keys):
    if set(state) == set(target_keys):
        return state, False
    # The installed Transformers 5.14 renamed SegFormer parameters. These
    # names follow its own conversion_mapping.py; tensor values are unchanged.
    patterns = (
        (r"encoder\.patch_embeddings\.(\d+)\.", r"stages.\1.patch_embeddings."),
        (r"encoder\.block\.(\d+)\.", r"stages.\1.blocks."),
        (r"encoder\.layer_norm\.(\d+)", r"stages.\1.layer_norm"),
    )
    names = (
        ("attention.self.query", "attention.q_proj"),
        ("attention.self.key", "attention.k_proj"),
        ("attention.self.value", "attention.v_proj"),
        ("attention.self.sr", "attention.sequence_reduction.sequence_reduction"),
        ("attention.self.layer_norm", "attention.sequence_reduction.layer_norm"),
        ("attention.output.dense", "attention.o_proj"),
        ("mlp.dense1", "mlp.fc1"),
        ("mlp.dense2", "mlp.fc2"),
        ("layer_norm_1", "layernorm_before"),
        ("layer_norm_2", "layernorm_after"),
        ("decode_head.linear_c", "decode_head.linear_projections"),
    )
    converted = {}
    for key, tensor in state.items():
        for pattern, replacement in patterns:
            key = re.sub(pattern, replacement, key)
        for old, new in names:
            key = key.replace(old, new)
        if key in converted:
            raise ValueError(f"Duplicate mapped checkpoint key: {key}")
        converted[key] = tensor
    if set(converted) != set(target_keys):
        missing = set(target_keys) - set(converted)
        unexpected = set(converted) - set(target_keys)
        raise ValueError(f"Checkpoint schema mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    return converted, True


def load_checkpoint(path: Path, device: torch.device):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"model", "model_config", "size", "threshold"}.issubset(checkpoint):
        raise ValueError("Expected the provided experiment C checkpoint format.")
    size = checkpoint["size"]
    threshold = float(checkpoint["threshold"])
    if isinstance(size, bool) or not isinstance(size, int) or size < 32:
        raise ValueError("Wound checkpoint size must be an integer of at least 32.")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Wound checkpoint threshold must be between 0 and 1.")
    config = SegformerConfig.from_dict(checkpoint["model_config"])
    if config.num_labels != 1 or config.num_channels != 3:
        raise ValueError("The supplied inference expects RGB input and one wound logit channel.")
    config._attn_implementation = "eager"
    model = SegformerForSemanticSegmentation(config)
    state, converted = _compatible_weights(checkpoint["model"], model.state_dict())
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    checkpoint["inference_weight_keys_converted"] = converted
    return model, checkpoint


def image_tensor(original: Image.Image, size: int):
    width, height = original.size
    scale = size / max(width, height)
    resized_width = max(1, min(size, round(width * scale)))
    resized_height = max(1, min(size, round(height * scale)))
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    resized = original.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, (left, top))
    tensor = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255.0
    return (tensor - MEAN) / STD, Geometry(width, height, resized_width, resized_height, left, top)


def native_logits(logits: torch.Tensor, geometries: list[Geometry], size: int):
    if len(geometries) != logits.shape[0]:
        raise ValueError("Geometry count does not match inference batch size.")
    square = F.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)
    restored = []
    for index, geometry in enumerate(geometries):
        crop = square[index:index + 1, :,
                      geometry.top:geometry.top + geometry.resized_height,
                      geometry.left:geometry.left + geometry.resized_width]
        original = F.interpolate(crop, size=(geometry.original_height, geometry.original_width),
                                 mode="bilinear", align_corners=False)
        restored.append(original[0])
    return restored
