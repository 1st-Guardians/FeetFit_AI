"""High-resolution colour-aware pressure segmentation on frozen SAM features.

The original :class:`~arch_ai.model.SamPressureSegmenter` decodes only the
64 x 64 SAM embedding.  That representation is useful for global foot shape,
but it can discard weak, thin contact-colour evidence.  This module keeps the
official frozen SAM ViT-B encoder and adds a differentiable 256 x 256
RGB/Lab/HSV path with U-Net/FPN-style skip fusion.

``HybridSamPressureSegmenter`` intentionally follows the public API of
``SamPressureSegmenter``.  Existing dataloaders can therefore continue to use
``preprocess_tensor_batch`` and existing checkpoint helpers can continue to
save/load the trainable ``decoder`` only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import PressureForwardOutput, ResidualGroupNormBlock, SamPressureSegmenter


def _normalization_tensor(
    value: torch.Tensor | Sequence[float],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return a broadcastable BCHW colour-normalization tensor."""

    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.numel() != 3:
        raise ValueError(f"Expected three RGB normalization values, got {tensor.numel()}")
    return tensor.reshape(1, 3, 1, 1)


def sam_valid_rectangle(pixel_values: torch.Tensor) -> torch.Tensor:
    """Recover SAM's top-left valid rectangle from a normalized padded batch.

    Official SAM preprocessing normalizes the resized image and then pads its
    bottom and right edges with exact zeros.  Looking only for non-zero pixels
    would create holes if a valid RGB pixel happened to equal SAM's mean.  We
    instead find the last occupied row/column and return the complete top-left
    rectangle, which matches the preprocessing contract.

    Returns a boolean tensor of shape ``[B, 1, H, W]``.
    """

    if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
        raise ValueError(f"Expected normalized BCHW RGB input, got {tuple(pixel_values.shape)}")

    occupied = pixel_values.ne(0).any(dim=1)
    row_occupied = occupied.any(dim=2)
    col_occupied = occupied.any(dim=1)
    height_indices = torch.arange(
        1,
        pixel_values.shape[-2] + 1,
        device=pixel_values.device,
        dtype=torch.long,
    ).unsqueeze(0)
    width_indices = torch.arange(
        1,
        pixel_values.shape[-1] + 1,
        device=pixel_values.device,
        dtype=torch.long,
    ).unsqueeze(0)
    valid_heights = torch.where(row_occupied, height_indices, 0).amax(dim=1)
    valid_widths = torch.where(col_occupied, width_indices, 0).amax(dim=1)

    rows = torch.arange(pixel_values.shape[-2], device=pixel_values.device).view(1, 1, -1, 1)
    cols = torch.arange(pixel_values.shape[-1], device=pixel_values.device).view(1, 1, 1, -1)
    return (rows < valid_heights.view(-1, 1, 1, 1)) & (
        cols < valid_widths.view(-1, 1, 1, 1)
    )


def recover_rgb_from_sam_tensor(
    pixel_values: torch.Tensor,
    pixel_mean: torch.Tensor | Sequence[float],
    pixel_std: torch.Tensor | Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert SAM normalization and remove normalized padding safely.

    Args:
        pixel_values: SAM-normalized and padded BCHW tensor.
        pixel_mean: SAM RGB mean in the 0..255 convention.
        pixel_std: SAM RGB standard deviation in the 0..255 convention.

    Returns:
        ``(rgb, valid_mask)`` where RGB is in ``[0, 1]`` and padded pixels are
        exactly zero.  All arithmetic on valid RGB pixels is differentiable.
    """

    if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
        raise ValueError(f"Expected normalized BCHW RGB input, got {tuple(pixel_values.shape)}")
    mean = _normalization_tensor(pixel_mean, pixel_values)
    std = _normalization_tensor(pixel_std, pixel_values)
    if torch.any(std <= 0):
        raise ValueError("SAM pixel_std values must be positive")
    valid_mask = sam_valid_rectangle(pixel_values)
    rgb = (pixel_values * std + mean) / 255.0
    rgb = rgb.clamp(0.0, 1.0) * valid_mask.to(dtype=rgb.dtype)
    return rgb, valid_mask


def rgb_to_hsv(rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable RGB-to-HSV conversion for RGB tensors in ``[0, 1]``."""

    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB input, got {tuple(rgb.shape)}")
    red, green, blue = rgb.unbind(dim=1)
    maximum, maximum_index = rgb.max(dim=1)
    minimum = rgb.min(dim=1).values
    delta = maximum - minimum
    safe_delta = delta.clamp_min(eps)

    red_hue = torch.remainder((green - blue) / safe_delta, 6.0)
    green_hue = (blue - red) / safe_delta + 2.0
    blue_hue = (red - green) / safe_delta + 4.0
    hue = torch.where(
        maximum_index == 0,
        red_hue,
        torch.where(maximum_index == 1, green_hue, blue_hue),
    ) / 6.0
    hue = torch.where(delta > eps, hue, torch.zeros_like(hue))
    saturation = torch.where(maximum > eps, delta / maximum.clamp_min(eps), torch.zeros_like(maximum))
    return torch.stack((hue, saturation, maximum), dim=1)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable sRGB-to-CIE Lab conversion with network-friendly scale.

    The output channels are ``L/100``, ``a/128`` and ``b/128``.  L is thus in
    approximately ``[0, 1]`` while the chromatic channels remain centred at 0.
    """

    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB input, got {tuple(rgb.shape)}")
    rgb = rgb.clamp(0.0, 1.0)
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).clamp_min(0.0).pow(2.4),
    )
    red, green, blue = linear.unbind(dim=1)
    # sRGB, D65 illuminant. XYZ is divided by the D65 reference white.
    x = (0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047
    y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    z = (0.0193339 * red + 0.1191920 * green + 0.9503041 * blue) / 1.08883

    delta = 6.0 / 29.0
    threshold = delta**3
    linear_scale = 1.0 / (3.0 * delta**2)

    def lab_curve(component: torch.Tensor) -> torch.Tensor:
        return torch.where(
            component > threshold,
            component.clamp_min(1e-8).pow(1.0 / 3.0),
            component * linear_scale + 4.0 / 29.0,
        )

    fx, fy, fz = lab_curve(x), lab_curve(y), lab_curve(z)
    lightness = (116.0 * fy - 16.0) / 100.0
    channel_a = 500.0 * (fx - fy) / 128.0
    channel_b = 200.0 * (fy - fz) / 128.0
    return torch.stack((lightness, channel_a, channel_b), dim=1)


def rgb_lab_hsv_features(rgb: torch.Tensor) -> torch.Tensor:
    """Concatenate raw RGB, normalized CIE Lab and HSV channels."""

    return torch.cat((rgb, rgb_to_lab(rgb), rgb_to_hsv(rgb)), dim=1)


class _DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        norm_groups = max(
            group for group in range(min(groups, out_channels), 0, -1) if out_channels % group == 0
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(inplace=True),
        )
        self.refine = ResidualGroupNormBlock(out_channels, out_channels, groups=groups)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.refine(self.downsample(inputs))


class HybridPressureDecoder(nn.Module):
    """Fuse high-resolution colour skips with the frozen SAM embedding.

    The local path runs at ``output_size``, ``output_size/2`` and
    ``output_size/4``.  The deepest local feature and SAM embedding are fused,
    then decoded through two U-Net/FPN-style skip stages.
    """

    def __init__(
        self,
        sam_in_channels: int = 256,
        hidden_channels: Sequence[int] = (32, 64, 96),
        output_size: int = 256,
        groups: int = 8,
    ) -> None:
        super().__init__()
        if len(hidden_channels) != 3:
            raise ValueError("hidden_channels must contain exactly three values")
        fine, middle, coarse = (int(value) for value in hidden_channels)
        if min(sam_in_channels, fine, middle, coarse, output_size) <= 0:
            raise ValueError("channel counts and output_size must be positive")
        if output_size % 4 != 0:
            raise ValueError("output_size must be divisible by 4")

        self.sam_in_channels = int(sam_in_channels)
        self.hidden_channels = (fine, middle, coarse)
        self.output_size = int(output_size)
        self.color_channels = 9

        self.fine_encoder = ResidualGroupNormBlock(self.color_channels, fine, groups=groups)
        self.middle_encoder = _DownsampleBlock(fine, middle, groups=groups)
        self.coarse_encoder = _DownsampleBlock(middle, coarse, groups=groups)
        self.sam_projection = ResidualGroupNormBlock(
            self.sam_in_channels,
            coarse,
            groups=groups,
        )
        self.coarse_fusion = ResidualGroupNormBlock(coarse * 2, coarse, groups=groups)
        self.middle_fusion = ResidualGroupNormBlock(coarse + middle, middle, groups=groups)
        self.fine_fusion = ResidualGroupNormBlock(middle + fine, fine, groups=groups)
        self.output = nn.Conv2d(fine, 1, kernel_size=1)

    @staticmethod
    def _apply_mask(features: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        if valid_mask is None:
            return features
        resized = F.interpolate(valid_mask.to(dtype=features.dtype), size=features.shape[-2:], mode="nearest")
        return features * resized

    def forward(
        self,
        image_embeddings: torch.Tensor,
        raw_rgb: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_embeddings.ndim != 4 or image_embeddings.shape[1] != self.sam_in_channels:
            raise ValueError(
                f"Expected SAM embeddings [B, {self.sam_in_channels}, H, W], "
                f"got {tuple(image_embeddings.shape)}"
            )
        if raw_rgb.ndim != 4 or raw_rgb.shape[1] != 3:
            raise ValueError(f"Expected raw RGB [B, 3, H, W], got {tuple(raw_rgb.shape)}")
        if raw_rgb.shape[0] != image_embeddings.shape[0]:
            raise ValueError("SAM embedding and raw RGB batch sizes differ")
        if valid_mask is not None and (
            valid_mask.ndim != 4
            or valid_mask.shape[0] != raw_rgb.shape[0]
            or valid_mask.shape[1] != 1
            or valid_mask.shape[-2:] != raw_rgb.shape[-2:]
        ):
            raise ValueError("valid_mask must have shape [B, 1, raw_height, raw_width]")

        local_rgb = F.interpolate(
            raw_rgb,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        local_mask = None
        if valid_mask is not None:
            local_mask = F.interpolate(
                valid_mask.to(dtype=local_rgb.dtype),
                size=(self.output_size, self.output_size),
                mode="nearest",
            )
            local_rgb = local_rgb * local_mask

        colour = rgb_lab_hsv_features(local_rgb)
        fine = self._apply_mask(self.fine_encoder(colour), local_mask)
        middle = self._apply_mask(self.middle_encoder(fine), local_mask)
        coarse = self._apply_mask(self.coarse_encoder(middle), local_mask)

        sam_feature = self.sam_projection(image_embeddings)
        sam_feature = F.interpolate(
            sam_feature,
            size=coarse.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused_coarse = self.coarse_fusion(torch.cat((coarse, sam_feature), dim=1))
        fused_coarse = self._apply_mask(fused_coarse, local_mask)

        up_middle = F.interpolate(
            fused_coarse,
            size=middle.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused_middle = self.middle_fusion(torch.cat((up_middle, middle), dim=1))
        fused_middle = self._apply_mask(fused_middle, local_mask)
        up_fine = F.interpolate(
            fused_middle,
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused_fine = self.fine_fusion(torch.cat((up_fine, fine), dim=1))
        fused_fine = self._apply_mask(fused_fine, local_mask)
        return self.output(fused_fine)


class HybridSamPressureSegmenter(SamPressureSegmenter):
    """Frozen official SAM ViT-B plus a trainable colour-aware hybrid decoder."""

    def __init__(
        self,
        sam_checkpoint: str | Path,
        decoder_channels: Sequence[int] = (32, 64, 96),
        output_size: int = 256,
    ) -> None:
        # The parent establishes the complete official SAM preprocessing and
        # postprocessing contract.  Its small decoder is immediately replaced.
        super().__init__(
            sam_checkpoint=sam_checkpoint,
            decoder_channels=(128, 64),
            output_size=output_size,
        )
        self.decoder = HybridPressureDecoder(
            sam_in_channels=256,
            hidden_channels=decoder_channels,
            output_size=output_size,
        )
        self.freeze_sam()

    def recover_raw_rgb(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover valid 0..1 RGB and its mask from preprocessed SAM input."""

        return recover_rgb_from_sam_tensor(
            pixel_values,
            pixel_mean=self.sam.pixel_mean,
            pixel_std=self.sam.pixel_std,
        )

    def extract_local_rgb(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return the decoder's zero-padded RGB view at ``output_size``.

        This explicit method lets feature-caching callers cache SAM embeddings
        and the matching colour input together.  Decoding an embedding without
        its RGB view is intentionally unsupported because it would make cached
        training and ordinary forward inference behave differently.
        """

        raw_rgb, valid_mask = self.recover_raw_rgb(pixel_values)
        local_rgb = F.interpolate(
            raw_rgb,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        local_mask = F.interpolate(
            valid_mask.to(dtype=local_rgb.dtype),
            size=(self.output_size, self.output_size),
            mode="nearest",
        )
        return local_rgb * local_mask

    def decode_embeddings(
        self,
        image_embeddings: torch.Tensor,
        local_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """Decode cached SAM embeddings with their required local RGB view."""

        if local_rgb.ndim != 4 or local_rgb.shape[1] != 3:
            raise ValueError(f"Expected local RGB [B, 3, H, W], got {tuple(local_rgb.shape)}")
        if tuple(local_rgb.shape[-2:]) != (self.output_size, self.output_size):
            raise ValueError(
                f"local_rgb must be {self.output_size}x{self.output_size}; "
                f"got {tuple(local_rgb.shape[-2:])}"
            )
        return self.decoder(image_embeddings, local_rgb)

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | PressureForwardOutput:
        embeddings = self.encode(pixel_values)
        local_rgb = self.extract_local_rgb(pixel_values)
        logits = self.decode_embeddings(embeddings, local_rgb)
        if return_embeddings:
            return PressureForwardOutput(low_res_logits=logits, image_embeddings=embeddings)
        return logits

    def decoder_config(self) -> dict[str, Any]:
        # Keep the three keys checked by arch_ai.model's generic checkpoint
        # loader, and add enough metadata for a checkpoint-driven factory.
        return {
            "model_type": self.model_type,
            "decoder_channels": list(self.decoder.hidden_channels),
            "output_size": self.output_size,
            "architecture": "hybrid_color_fpn",
            "architecture_version": 1,
            "color_channels": self.decoder.color_channels,
            "sam_in_channels": self.decoder.sam_in_channels,
        }


__all__ = [
    "HybridPressureDecoder",
    "HybridSamPressureSegmenter",
    "recover_rgb_from_sam_tensor",
    "rgb_lab_hsv_features",
    "rgb_to_hsv",
    "rgb_to_lab",
    "sam_valid_rectangle",
]
