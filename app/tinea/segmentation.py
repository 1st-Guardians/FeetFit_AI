"""Offline inference for the trained binary SegFormer-B2 Student checkpoint."""
from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

PREPROCESSING_POLICY = "rgb_square_bilinear_imagenet_384_v1"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ARCHITECTURE = "segformer_b2_teacher_student"


def compatible_state_dict(state: dict, target_keys) -> tuple[dict, bool]:
    """Rename HF 4.x SegFormer keys for HF 5.x, preserving every tensor."""
    if set(state) == set(target_keys):
        return state, False
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
            raise ValueError(f"Duplicate fungal checkpoint key: {key}")
        converted[key] = tensor
    if set(converted) != set(target_keys):
        missing = set(target_keys) - set(converted)
        unexpected = set(converted) - set(target_keys)
        raise ValueError(f"Fungal checkpoint schema mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    return converted, True


def image_tensor(image: Image.Image, size: int, device: str) -> torch.Tensor:
    """Match training: RGB, square PIL bilinear resize, then ImageNet normalize."""
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1)[None].float().to(device) / 255.0
    mean = tensor.new_tensor(MEAN).view(1, 3, 1, 1)
    std = tensor.new_tensor(STD).view(1, 3, 1, 1)
    return ((tensor - mean) / std).contiguous()


class FungalSegmenter:
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        required = {"architecture", "model", "model_config", "input_size", "normalization", "threshold"}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise ValueError("Expected a portable fungal SegFormer Teacher-Student checkpoint.")
        if checkpoint["architecture"] != ARCHITECTURE:
            raise ValueError("Expected the fungal SegFormer-B2 Teacher-Student architecture.")
        self.input_size = checkpoint["input_size"]
        if isinstance(self.input_size, bool) or not isinstance(self.input_size, int) or self.input_size != 384:
            raise ValueError("Fungal checkpoint input size must be 384.")
        self.threshold = float(checkpoint["threshold"])
        if not np.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("Fungal checkpoint threshold must be between 0 and 1.")
        if checkpoint["normalization"] != {"mean": MEAN, "std": STD}:
            raise ValueError("Fungal checkpoint must use the training ImageNet normalization.")
        config = SegformerConfig.from_dict(checkpoint["model_config"])
        if (config.num_labels != 1 or config.num_channels != 3
                or list(config.hidden_sizes) != [64, 128, 320, 512]
                or list(config.depths) != [3, 4, 6, 3] or config.decoder_hidden_size != 768):
            raise ValueError("Fungal inference requires RGB SegFormer-B2 with one foreground logit.")
        config._attn_implementation = "eager"
        # Construct from local configuration only; never call from_pretrained.
        self.network = SegformerForSemanticSegmentation(config)
        state, self.weight_keys_converted = compatible_state_dict(checkpoint["model"], self.network.state_dict())
        self.network.load_state_dict(state, strict=True)
        self.network.to(device).eval()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.source_checkpoint_sha256 = checkpoint.get("source_checkpoint_sha256")
        self.threshold_selection = checkpoint.get("threshold_selection", "validation_mean_dice")
        self._inference_lock = threading.Lock()

    def predict_probability(self, image: Image.Image) -> np.ndarray:
        tensor = image_tensor(image, self.input_size, self.device)
        with self._inference_lock, torch.inference_mode():
            logits = self.network(pixel_values=tensor).logits
            # The trained wrapper upsamples logits to 384 before sigmoid.
            # Resize probabilities (not logits) back to native coordinates.
            logits = F.interpolate(logits.float(), size=(self.input_size, self.input_size),
                                   mode="bilinear", align_corners=False)
            probability = F.interpolate(logits.sigmoid(), size=(image.height, image.width),
                                        mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        if (probability.shape != (image.height, image.width) or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))):
            raise ValueError("Invalid fungal probability map.")
        return probability
