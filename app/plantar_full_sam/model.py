"""SAM ViT-B based binary pressure-map segmentation.

The module intentionally keeps the official Segment Anything preprocessing
contract: RGB images are resized with ``ResizeLongestSide``, normalized and
padded by ``Sam.preprocess``, and 256 x 256 logits are restored to the source
resolution with ``Sam.postprocess_masks``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


Size2D = tuple[int, int]
ImageLike = np.ndarray | Image.Image


@dataclass(frozen=True)
class ImageGeometry:
    """Spatial metadata required by ``Sam.postprocess_masks``.

    Sizes follow the PyTorch convention ``(height, width)``.
    """

    original_size: Size2D
    input_size: Size2D


@dataclass
class PreparedSamBatch:
    """A fixed-size SAM batch and the geometry of every source image."""

    pixel_values: torch.Tensor
    geometries: tuple[ImageGeometry, ...]

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "PreparedSamBatch":
        return PreparedSamBatch(
            pixel_values=self.pixel_values.to(device, non_blocking=non_blocking),
            geometries=self.geometries,
        )


@dataclass
class PressureForwardOutput:
    """Low-resolution decoder output and its SAM image embeddings."""

    low_res_logits: torch.Tensor
    image_embeddings: torch.Tensor


@dataclass
class PressurePrediction:
    """Prediction result before and after restoration to original resolution."""

    low_res_logits: torch.Tensor
    image_embeddings: torch.Tensor
    original_logits: tuple[torch.Tensor, ...]
    original_probabilities: tuple[torch.Tensor, ...]
    geometries: tuple[ImageGeometry, ...]


def _group_count(channels: int, preferred_groups: int = 8) -> int:
    """Return the largest useful GroupNorm divisor up to ``preferred_groups``."""

    for groups in range(min(preferred_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualGroupNormBlock(nn.Module):
    """Two 3 x 3 convolutions with GroupNorm and a residual projection."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        norm_groups = _group_count(out_channels, groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(norm_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(norm_groups, out_channels)
        self.activation = nn.SiLU(inplace=True)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        outputs = self.activation(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        return self.activation(outputs + residual)


class PressureMaskDecoder(nn.Module):
    """Decode a 64 x 64 SAM embedding into one 256 x 256 pressure logit map."""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: Sequence[int] = (128, 64),
        output_size: int = 256,
        groups: int = 8,
    ) -> None:
        super().__init__()
        if len(hidden_channels) != 2:
            raise ValueError("hidden_channels must contain exactly two values")
        first, second = (int(value) for value in hidden_channels)
        if min(in_channels, first, second, output_size) <= 0:
            raise ValueError("channel counts and output_size must be positive")

        self.in_channels = int(in_channels)
        self.hidden_channels = (first, second)
        self.output_size = int(output_size)
        self.stem = ResidualGroupNormBlock(self.in_channels, first, groups=groups)
        self.up_block1 = ResidualGroupNormBlock(first, first, groups=groups)
        self.channel_reduce = ResidualGroupNormBlock(first, second, groups=groups)
        self.up_block2 = ResidualGroupNormBlock(second, second, groups=groups)
        self.output = nn.Conv2d(second, 1, kernel_size=1)

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        if image_embeddings.ndim != 4 or image_embeddings.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected embeddings [B, {self.in_channels}, H, W], "
                f"got {tuple(image_embeddings.shape)}"
            )
        outputs = self.stem(image_embeddings)
        outputs = F.interpolate(outputs, scale_factor=2.0, mode="bilinear", align_corners=False)
        outputs = self.up_block1(outputs)
        outputs = self.channel_reduce(outputs)
        outputs = F.interpolate(
            outputs,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        outputs = self.up_block2(outputs)
        return self.output(outputs)


class SamPressureSegmenter(nn.Module):
    """Frozen official SAM ViT-B image encoder plus a trainable binary decoder."""

    def __init__(
        self,
        sam_checkpoint: str | Path,
        decoder_channels: Sequence[int] = (128, 64),
        output_size: int = 256,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(sam_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

        # This project deliberately targets the official ViT-B architecture.
        self.model_type = "vit_b"
        self.sam_checkpoint = str(checkpoint_path)
        self.sam = sam_model_registry[self.model_type](checkpoint=str(checkpoint_path))
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        self.decoder = PressureMaskDecoder(
            in_channels=256,
            hidden_channels=decoder_channels,
            output_size=output_size,
        )
        self.freeze_sam()

    @property
    def output_size(self) -> int:
        return self.decoder.output_size

    @property
    def device(self) -> torch.device:
        return next(self.decoder.parameters()).device

    def freeze_sam(self) -> None:
        """Freeze every official SAM parameter; only ``decoder`` stays trainable."""

        self.sam.requires_grad_(False)
        self.sam.eval()
        self.decoder.requires_grad_(True)

    def train(self, mode: bool = True) -> "SamPressureSegmenter":
        # ``super().train`` also toggles SAM, so restore the frozen backbone to
        # eval mode to avoid accidental behavioral changes in future variants.
        super().train(mode)
        self.sam.eval()
        self.decoder.train(mode)
        return self

    @staticmethod
    def _as_rgb_array(image: ImageLike) -> np.ndarray:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"))
        else:
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=2)
            if array.ndim != 3 or array.shape[2] not in (3, 4):
                raise ValueError(f"Expected an HWC grayscale/RGB/RGBA image, got {array.shape}")
            array = array[..., :3]

        if np.issubdtype(array.dtype, np.floating):
            finite_max = float(np.nanmax(array)) if array.size else 0.0
            if finite_max <= 1.0:
                array = array * 255.0
        return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))

    def prepare_image(
        self,
        image: ImageLike,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, ImageGeometry]:
        """Resize, normalize and pad one RGB image using official SAM logic."""

        rgb = self._as_rgb_array(image)
        original_size = (int(rgb.shape[0]), int(rgb.shape[1]))
        resized = self.transform.apply_image(rgb)
        input_size = (int(resized.shape[0]), int(resized.shape[1]))
        target_device = self.device if device is None else torch.device(device)
        tensor = torch.as_tensor(resized.copy(), dtype=torch.float32, device=target_device)
        if tensor.device != self.sam.pixel_mean.device:
            raise ValueError(
                "prepare_image must run on the same device as the model; "
                f"model={self.sam.pixel_mean.device}, requested={tensor.device}"
            )
        tensor = tensor.permute(2, 0, 1).contiguous().unsqueeze(0)
        tensor = self.sam.preprocess(tensor)[0]
        return tensor, ImageGeometry(original_size=original_size, input_size=input_size)

    def preprocess_tensor_batch(
        self,
        images: torch.Tensor,
        input_range: str = "0_1",
        input_sizes: Sequence[Sequence[int] | torch.Tensor] | torch.Tensor | None = None,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        """Apply official SAM normalization/padding to a BCHW tensor batch.

        This is the efficient path for dataloaders that already performed the
        same aspect-ratio-preserving, top-left letterbox resize. Pass its
        unpadded ``(height, width)`` values as ``input_sizes`` so black canvas
        pixels are cropped before SAM adds zero-valued normalized padding. Use
        ``input_range='0_1'`` for ordinary image tensors and ``'0_255'`` for
        raw SAM-range values.
        """

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB images, got {tuple(images.shape)}")
        sam_size = int(self.sam.image_encoder.img_size)
        if images.shape[-2] > sam_size or images.shape[-1] > sam_size:
            raise ValueError(
                f"Tensor spatial size must not exceed SAM's {sam_size}x{sam_size} input"
            )
        inputs = images.to(
            device=self.sam.pixel_mean.device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )
        if input_range == "0_1":
            inputs = inputs * 255.0
        elif input_range != "0_255":
            raise ValueError("input_range must be '0_1' or '0_255'")
        if input_sizes is None:
            return self.sam.preprocess(inputs)

        if len(input_sizes) != inputs.shape[0]:
            raise ValueError("input_sizes and image batch sizes differ")
        processed: list[torch.Tensor] = []
        for index, size in enumerate(input_sizes):
            values = size.detach().cpu().reshape(-1).tolist() if isinstance(size, torch.Tensor) else list(size)
            if len(values) != 2:
                raise ValueError(f"Invalid input size: {values}")
            height, width = int(values[0]), int(values[1])
            if height <= 0 or width <= 0 or height > inputs.shape[-2] or width > inputs.shape[-1]:
                raise ValueError(f"Input size {(height, width)} is outside tensor bounds {inputs.shape[-2:]}")
            processed.append(self.sam.preprocess(inputs[index : index + 1, :, :height, :width])[0])
        return torch.stack(processed, dim=0)

    @staticmethod
    def geometries_from_sizes(
        original_sizes: Sequence[Sequence[int] | torch.Tensor],
        input_sizes: Sequence[Sequence[int] | torch.Tensor],
    ) -> tuple[ImageGeometry, ...]:
        """Build geometry objects from dataloader ``source/resized_size`` fields."""

        if len(original_sizes) != len(input_sizes):
            raise ValueError("original_sizes and input_sizes have different lengths")

        def convert(size: Sequence[int] | torch.Tensor) -> Size2D:
            values = size.detach().cpu().reshape(-1).tolist() if isinstance(size, torch.Tensor) else list(size)
            if len(values) != 2 or min(int(values[0]), int(values[1])) <= 0:
                raise ValueError(f"Invalid (height, width) size: {values}")
            return int(values[0]), int(values[1])

        return tuple(
            ImageGeometry(original_size=convert(original), input_size=convert(resized))
            for original, resized in zip(original_sizes, input_sizes)
        )

    def prepare_batch(
        self,
        images: Iterable[ImageLike],
        device: torch.device | str | None = None,
    ) -> PreparedSamBatch:
        """Prepare differently shaped images and stack their padded SAM tensors."""

        tensors: list[torch.Tensor] = []
        geometries: list[ImageGeometry] = []
        for image in images:
            tensor, geometry = self.prepare_image(image, device=device)
            tensors.append(tensor)
            geometries.append(geometry)
        if not tensors:
            raise ValueError("images must contain at least one image")
        return PreparedSamBatch(torch.stack(tensors, dim=0), tuple(geometries))

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return detached SAM image embeddings for a preprocessed batch."""

        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError(f"Expected preprocessed BCHW RGB input, got {tuple(pixel_values.shape)}")
        expected = int(self.sam.image_encoder.img_size)
        if tuple(pixel_values.shape[-2:]) != (expected, expected):
            raise ValueError(
                f"SAM expects padded {expected}x{expected} inputs; got {tuple(pixel_values.shape[-2:])}"
            )
        with torch.no_grad():
            return self.sam.image_encoder(pixel_values).detach()

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | PressureForwardOutput:
        embeddings = self.encode(pixel_values)
        logits = self.decoder(embeddings)
        if return_embeddings:
            return PressureForwardOutput(low_res_logits=logits, image_embeddings=embeddings)
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

    def postprocess_logits(
        self,
        low_res_logits: torch.Tensor,
        geometries: Sequence[ImageGeometry],
    ) -> tuple[torch.Tensor, ...]:
        """Restore each 256 x 256 logit map to its source image resolution."""

        if low_res_logits.ndim != 4 or low_res_logits.shape[1] != 1:
            raise ValueError(f"Expected logits [B, 1, H, W], got {tuple(low_res_logits.shape)}")
        if low_res_logits.shape[0] != len(geometries):
            raise ValueError("Batch size and geometry count differ")
        restored: list[torch.Tensor] = []
        for index, geometry in enumerate(geometries):
            logits = self.sam.postprocess_masks(
                low_res_logits[index : index + 1],
                input_size=geometry.input_size,
                original_size=geometry.original_size,
            )
            restored.append(logits[0, 0])
        return tuple(restored)

    def low_res_valid_mask(
        self,
        geometries: Sequence[ImageGeometry],
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Mark the unpadded area of every low-resolution decoder output."""

        target_device = self.device if device is None else torch.device(device)
        masks = torch.zeros(
            (len(geometries), 1, self.output_size, self.output_size),
            device=target_device,
            dtype=dtype,
        )
        sam_size = int(self.sam.image_encoder.img_size)
        for index, geometry in enumerate(geometries):
            valid_height = min(
                self.output_size,
                int(math.ceil(geometry.input_size[0] * self.output_size / sam_size)),
            )
            valid_width = min(
                self.output_size,
                int(math.ceil(geometry.input_size[1] * self.output_size / sam_size)),
            )
            masks[index, :, :valid_height, :valid_width] = 1
        return masks

    @torch.no_grad()
    def predict(
        self,
        images: Iterable[ImageLike],
        device: torch.device | str | None = None,
    ) -> PressurePrediction:
        """Run segmentation and return probabilities at every original size."""

        was_training = self.training
        self.eval()
        try:
            output, prepared = self.forward_images(images, device=device)
            original_logits = self.postprocess_logits(output.low_res_logits, prepared.geometries)
            original_probabilities = tuple(torch.sigmoid(logits) for logits in original_logits)
            return PressurePrediction(
                low_res_logits=output.low_res_logits,
                image_embeddings=output.image_embeddings,
                original_logits=original_logits,
                original_probabilities=original_probabilities,
                geometries=prepared.geometries,
            )
        finally:
            self.train(was_training)

    def decoder_config(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "decoder_channels": list(self.decoder.hidden_channels),
            "output_size": self.output_size,
        }


def save_segmentation_checkpoint(
    path: str | Path,
    model: SamPressureSegmenter,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int | None = None,
    metrics: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save the compact trainable decoder and optional train state."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    state: dict[str, Any] = {
        "format_version": 1,
        "model_config": model.decoder_config(),
        "decoder_state_dict": model.decoder.state_dict(),
        "epoch": epoch,
        "metrics": dict(metrics or {}),
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, temporary)
    temporary.replace(destination)
    return destination


def _trusted_torch_load(path: Path, map_location: str | torch.device) -> Any:
    # Optimizer/scheduler state requires the full trusted checkpoint loader.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        return torch.load(path, map_location=map_location)


def load_segmentation_checkpoint(
    path: str | Path,
    model: SamPressureSegmenter,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint created by :func:`save_segmentation_checkpoint`."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {source}")
    state = _trusted_torch_load(source, map_location=map_location)
    if not isinstance(state, Mapping) or "decoder_state_dict" not in state:
        raise ValueError(f"Unsupported segmentation checkpoint format: {source}")

    saved_config = state.get("model_config", {})
    current_config = model.decoder_config()
    for key in ("model_type", "decoder_channels", "output_size"):
        if key in saved_config and saved_config[key] != current_config[key]:
            raise ValueError(
                f"Checkpoint/model {key} mismatch: saved={saved_config[key]!r}, "
                f"current={current_config[key]!r}"
            )
    # Legacy SAM-decoder checkpoints predate the architecture key and remain
    # loadable.  New architecture-aware checkpoints must, however, fail
    # closed instead of relying on a later state-dict shape/key error.
    if (
        "architecture" in saved_config
        and "architecture" in current_config
        and saved_config["architecture"] != current_config["architecture"]
    ):
        raise ValueError(
            "Checkpoint/model architecture mismatch: "
            f"saved={saved_config['architecture']!r}, "
            f"current={current_config['architecture']!r}"
        )
    model.decoder.load_state_dict(state["decoder_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])

    return {
        "format_version": state.get("format_version"),
        "epoch": state.get("epoch"),
        "metrics": dict(state.get("metrics") or {}),
        "extra": dict(state.get("extra") or {}),
        "model_config": dict(saved_config),
    }


# Concise aliases for training scripts.
save_checkpoint = save_segmentation_checkpoint
load_checkpoint = load_segmentation_checkpoint
