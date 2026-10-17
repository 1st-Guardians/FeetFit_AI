"""SegFormer-B0 binary plantar-contact segmentation.

The wrapper shares the geometry and prediction dataclasses used by the local
plantar segmentation runtime. Images are aspect-ratio resized, top-left
letterboxed, and restored to their original resolution after inference.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .model import ImageGeometry, ImageLike, PreparedSamBatch, PressureForwardOutput, PressurePrediction


SEGFORMER_B0_MODEL_ID = "nvidia/mit-b0"


def _transformers_classes():
    """Import the optional Hugging Face implementation only when requested."""

    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "SegFormer requires transformers and safetensors. Install project "
            "requirements with `python -m pip install -r requirements.txt`."
        ) from exc
    return SegformerConfig, SegformerForSemanticSegmentation


class SegFormerB0PressureSegmenter(nn.Module):
    """Pretrained MiT-B0 encoder and SegFormer decode head for one binary mask."""

    model_type = "segformer_b0"

    def __init__(
        self,
        pretrained_model: str = SEGFORMER_B0_MODEL_ID,
        *,
        input_size: int = 512,
        output_size: int = 256,
        pretrained: bool = True,
        local_files_only: bool = False,
        network_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive")
        SegformerConfig, SegformerForSemanticSegmentation = _transformers_classes()
        self.pretrained_model = str(pretrained_model)
        self.input_size = int(input_size)
        self._output_size = int(output_size)
        if pretrained:
            self.network = SegformerForSemanticSegmentation.from_pretrained(
                self.pretrained_model,
                num_labels=1,
                id2label={0: "contact"},
                label2id={"contact": 0},
                ignore_mismatched_sizes=True,
                local_files_only=local_files_only,
                use_safetensors=True,
            )
        elif network_config is not None:
            config = SegformerConfig.from_dict(dict(network_config))
            self.network = SegformerForSemanticSegmentation(config)
        else:
            config = SegformerConfig(
                num_labels=1,
                id2label={0: "contact"},
                label2id={"contact": 0},
                depths=[2, 2, 2, 2],
                hidden_sizes=[32, 64, 160, 256],
                decoder_hidden_size=256,
            )
            self.network = SegformerForSemanticSegmentation(config)
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def device(self) -> torch.device:
        return next(self.network.parameters()).device

    def trainable_parameters(self):
        return self.network.parameters()

    @staticmethod
    def _as_rgb_array(image: ImageLike) -> np.ndarray:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"))
        else:
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=2)
            if array.ndim != 3 or array.shape[2] not in (3, 4):
                raise ValueError(f"Expected HWC grayscale/RGB/RGBA, got {array.shape}")
            array = array[..., :3]
        if np.issubdtype(array.dtype, np.floating):
            finite_max = float(np.nanmax(array)) if array.size else 0.0
            if finite_max <= 1.0:
                array = array * 255.0
        return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))

    def preprocess_tensor_batch(
        self,
        images: torch.Tensor,
        input_range: str = "0_1",
        input_sizes: Sequence[Sequence[int] | torch.Tensor] | torch.Tensor | None = None,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB images, got {tuple(images.shape)}")
        values = images.to(self.device, dtype=torch.float32, non_blocking=non_blocking)
        if input_range == "0_255":
            values = values / 255.0
        elif input_range != "0_1":
            raise ValueError("input_range must be '0_1' or '0_255'")
        values = F.interpolate(
            values,
            (self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        return (values - self.pixel_mean) / self.pixel_std

    @staticmethod
    def geometries_from_sizes(
        original_sizes: Sequence[Sequence[int] | torch.Tensor],
        input_sizes: Sequence[Sequence[int] | torch.Tensor],
    ) -> tuple[ImageGeometry, ...]:
        def pair(value: Sequence[int] | torch.Tensor) -> tuple[int, int]:
            raw = value.detach().cpu().reshape(-1).tolist() if isinstance(value, torch.Tensor) else list(value)
            if len(raw) != 2 or min(int(raw[0]), int(raw[1])) <= 0:
                raise ValueError(f"Invalid (height, width) size: {raw}")
            return int(raw[0]), int(raw[1])

        if len(original_sizes) != len(input_sizes):
            raise ValueError("original_sizes and input_sizes have different lengths")
        return tuple(
            ImageGeometry(original_size=pair(original), input_size=pair(resized))
            for original, resized in zip(original_sizes, input_sizes)
        )

    def prepare_batch(
        self,
        images: Iterable[ImageLike],
        device: torch.device | str | None = None,
    ) -> PreparedSamBatch:
        arrays = [self._as_rgb_array(image) for image in images]
        if not arrays:
            raise ValueError("images must contain at least one image")
        tensors: list[torch.Tensor] = []
        geometries: list[ImageGeometry] = []
        for array in arrays:
            height, width = array.shape[:2]
            scale = self.input_size / max(height, width)
            resized_h = max(1, round(height * scale))
            resized_w = max(1, round(width * scale))
            tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor = F.interpolate(
                tensor,
                (resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
            )
            canvas = torch.zeros(1, 3, self.input_size, self.input_size)
            canvas[:, :, :resized_h, :resized_w] = tensor
            tensors.append(canvas[0])
            geometries.append(
                ImageGeometry(
                    original_size=(height, width),
                    input_size=(resized_h, resized_w),
                )
            )
        target = self.device if device is None else torch.device(device)
        batch = torch.stack(tensors).to(target)
        batch = (batch - self.pixel_mean) / self.pixel_std
        return PreparedSamBatch(pixel_values=batch, geometries=tuple(geometries))

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | PressureForwardOutput:
        output = self.network(
            pixel_values=pixel_values,
            output_hidden_states=return_embeddings,
        )
        logits = F.interpolate(
            output.logits,
            (self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        if return_embeddings:
            return PressureForwardOutput(
                low_res_logits=logits,
                image_embeddings=output.hidden_states[-1],
            )
        return logits

    def forward_images(
        self,
        images: Iterable[ImageLike],
        device: torch.device | str | None = None,
    ) -> tuple[PressureForwardOutput, PreparedSamBatch]:
        prepared = self.prepare_batch(images, device=device)
        output = self.forward(prepared.pixel_values, return_embeddings=True)
        if not isinstance(output, PressureForwardOutput):  # pragma: no cover - typing guard
            raise RuntimeError("Unexpected segmentation output")
        return output, prepared

    def low_res_valid_mask(
        self,
        geometries: Sequence[ImageGeometry],
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        result = torch.zeros(
            len(geometries),
            1,
            self.output_size,
            self.output_size,
            device=self.device if device is None else torch.device(device),
            dtype=dtype,
        )
        for index, geometry in enumerate(geometries):
            canvas = max(geometry.input_size)
            if canvas < max(geometry.original_size):
                canvas = self.input_size
            height = min(
                self.output_size,
                math.ceil(geometry.input_size[0] * self.output_size / canvas),
            )
            width = min(
                self.output_size,
                math.ceil(geometry.input_size[1] * self.output_size / canvas),
            )
            result[index, :, :height, :width] = 1
        return result

    def postprocess_logits(
        self,
        logits: torch.Tensor,
        geometries: Sequence[ImageGeometry],
    ) -> tuple[torch.Tensor, ...]:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"Expected logits [B, 1, H, W], got {tuple(logits.shape)}")
        if logits.shape[0] != len(geometries):
            raise ValueError("Batch size and geometry count differ")
        restored: list[torch.Tensor] = []
        for index, geometry in enumerate(geometries):
            height_ratio = geometry.input_size[0] / max(geometry.input_size)
            width_ratio = geometry.input_size[1] / max(geometry.input_size)
            height = max(1, min(logits.shape[-2], math.ceil(height_ratio * logits.shape[-2])))
            width = max(1, min(logits.shape[-1], math.ceil(width_ratio * logits.shape[-1])))
            crop = logits[index : index + 1, :, :height, :width]
            value = F.interpolate(
                crop,
                geometry.original_size,
                mode="bilinear",
                align_corners=False,
            )
            restored.append(value[0, 0])
        return tuple(restored)

    @torch.no_grad()
    def predict(
        self,
        images: Iterable[ImageLike],
        device: torch.device | str | None = None,
    ) -> PressurePrediction:
        was_training = self.training
        self.eval()
        try:
            output, prepared = self.forward_images(images, device=device)
            original_logits = self.postprocess_logits(output.low_res_logits, prepared.geometries)
            return PressurePrediction(
                low_res_logits=output.low_res_logits,
                image_embeddings=output.image_embeddings,
                original_logits=original_logits,
                original_probabilities=tuple(torch.sigmoid(value) for value in original_logits),
                geometries=prepared.geometries,
            )
        finally:
            self.train(was_training)

    def model_config(self) -> dict[str, Any]:
        return {
            "architecture": self.model_type,
            "pretrained_model": self.pretrained_model,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "network_config": self.network.config.to_dict(),
        }


def save_segformer_checkpoint(
    path: str | Path,
    model: SegFormerB0PressureSegmenter,
    **state: Any,
) -> Path:
    """Atomically save model weights and optional training state."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "model_config": model.model_config(),
        "model_state_dict": model.network.state_dict(),
        **state,
    }
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def _safe_torch_load(path: Path, map_location: str | torch.device) -> Any:
    """Load tensor-only checkpoints without permitting arbitrary pickle globals."""

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # Compatibility with PyTorch versions before weights_only.
        return torch.load(path, map_location=map_location)


def load_segformer_checkpoint(
    path: str | Path,
    model: SegFormerB0PressureSegmenter,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    **objects: Any,
) -> dict[str, Any]:
    """Load a SegFormer checkpoint, preferring PyTorch's safe tensor-only mode."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"SegFormer checkpoint not found: {source}")
    state = _safe_torch_load(source, map_location=map_location)
    if not isinstance(state, Mapping) or "model_state_dict" not in state:
        raise ValueError(f"Unsupported SegFormer checkpoint format: {source}")
    if state.get("model_config", {}).get("architecture") != model.model_type:
        raise ValueError("Checkpoint is not a SegFormer-B0 pressure checkpoint")
    model.network.load_state_dict(state["model_state_dict"], strict=strict)
    for name, obj in objects.items():
        key = f"{name}_state_dict"
        if obj is not None and key in state:
            obj.load_state_dict(state[key])
    return dict(state)


# Concise aliases for code shared with other segmentation backends.
save_checkpoint = save_segformer_checkpoint
load_checkpoint = load_segformer_checkpoint
