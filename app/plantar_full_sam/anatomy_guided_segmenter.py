"""Foot-anatomy-guided pressure segmentation on frozen SAM features.

The pressure decoder in this module has three independent sources of visual
context:

* the frozen official SAM image embedding;
* a high-resolution RGB/Lab/HSV path;
* the feature pyramid learned by :class:`FootAnatomyModel`.

The anatomy model is deliberately registered *inside* the pressure decoder.
Consequently the project's compact decoder-only segmentation checkpoint also
contains the exact anatomy representation used during pressure training.  By
default that representation is frozen and evaluated under ``torch.no_grad``;
only its context projections and the pressure decoder are optimized.

Anatomy information is context, not a prediction constraint.  No anatomy
silhouette/part logit is multiplied into the pressure mask, and no union,
morphology or forced bridge is applied.  This matters for cavus feet, whose
non-contact medial arch must remain a valid prediction even when the visible
foot silhouette is continuous.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .foot_anatomy import (
    FOOT_ANATOMY_CHECKPOINT_VERSION,
    FootAnatomyFeatures,
    FootAnatomyModel,
    FootAnatomySpec,
    build_foot_anatomy_from_checkpoint,
)
from .hybrid_segmenter import (
    HybridPressureDecoder,
    recover_rgb_from_sam_tensor,
    rgb_lab_hsv_features,
)
from .model import PressureForwardOutput, ResidualGroupNormBlock, SamPressureSegmenter


ANATOMY_GUIDED_SEGMENTER_ARCHITECTURE = "hybrid_foot_anatomy_fpn"
ANATOMY_GUIDED_SEGMENTER_VERSION = 1
ANATOMY_FUSION_VERSION = 1


class _PyramidContextProjection(nn.Module):
    """Resize, concatenate and project every anatomy FPN level.

    A separate instance is used at each pressure-decoder scale.  Concatenating
    all levels preserves both local boundaries and coarse whole-foot context;
    the projection is learned solely from pressure supervision.
    """

    def __init__(
        self,
        input_channels: Sequence[int],
        output_channels: int,
        *,
        groups: int,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in input_channels)
        if not channels or min(*channels, int(output_channels)) <= 0:
            raise ValueError("Anatomy input and output channels must be positive")
        self.input_channels = channels
        self.output_channels = int(output_channels)
        self.projection = ResidualGroupNormBlock(
            sum(channels), self.output_channels, groups=groups
        )

    def forward(
        self,
        pyramid: Sequence[torch.Tensor],
        spatial_size: Sequence[int],
    ) -> torch.Tensor:
        if len(pyramid) != len(self.input_channels):
            raise ValueError(
                f"Expected {len(self.input_channels)} anatomy levels, got {len(pyramid)}"
            )
        target_size = (int(spatial_size[0]), int(spatial_size[1]))
        resized: list[torch.Tensor] = []
        batch_size: int | None = None
        for index, (feature, channels) in enumerate(zip(pyramid, self.input_channels)):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError(
                    f"Anatomy level {index} must be [B, {channels}, H, W], "
                    f"got {tuple(feature.shape)}"
                )
            if batch_size is None:
                batch_size = int(feature.shape[0])
            elif int(feature.shape[0]) != batch_size:
                raise ValueError("Anatomy pyramid batch sizes differ")
            resized.append(
                F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return self.projection(torch.cat(resized, dim=1))


class AnatomyGuidedPressureDecoder(HybridPressureDecoder):
    """RGB/Lab/HSV + SAM decoder augmented with a foot-anatomy FPN.

    ``anatomy_model`` is a registered child module, so ``state_dict()`` is a
    complete downstream artifact.  Use :meth:`from_anatomy_checkpoint` for
    normal construction from the separate pretraining stage.
    """

    def __init__(
        self,
        anatomy_model: FootAnatomyModel,
        *,
        sam_in_channels: int = 256,
        hidden_channels: Sequence[int] = (32, 64, 96),
        anatomy_context_channels: Sequence[int] = (16, 24, 32),
        output_size: int = 256,
        groups: int = 8,
        freeze_anatomy: bool = True,
    ) -> None:
        super().__init__(
            sam_in_channels=sam_in_channels,
            hidden_channels=hidden_channels,
            output_size=output_size,
            groups=groups,
        )
        if not isinstance(anatomy_model, FootAnatomyModel):
            raise TypeError("anatomy_model must be a FootAnatomyModel")
        context_channels = tuple(int(value) for value in anatomy_context_channels)
        if len(context_channels) != 3 or min(context_channels) <= 0:
            raise ValueError(
                "anatomy_context_channels must contain three positive values"
            )

        # Registration under the decoder is intentional: save_checkpoint()
        # persists decoder.state_dict(), including these pretrained weights.
        self.anatomy_model = anatomy_model
        self.anatomy_context_channels = context_channels
        anatomy_channels = tuple(int(value) for value in anatomy_model.projected_feature_channels)
        self.anatomy_feature_channels = anatomy_channels
        if not anatomy_channels:
            raise ValueError("Foot anatomy model did not expose an FPN")

        fine, middle, coarse = self.hidden_channels
        anatomy_fine, anatomy_middle, anatomy_coarse = context_channels
        self.anatomy_fine_projection = _PyramidContextProjection(
            anatomy_channels, anatomy_fine, groups=groups
        )
        self.anatomy_middle_projection = _PyramidContextProjection(
            anatomy_channels, anatomy_middle, groups=groups
        )
        self.anatomy_coarse_projection = _PyramidContextProjection(
            anatomy_channels, anatomy_coarse, groups=groups
        )

        # Replace the ordinary hybrid fusion blocks with context-aware blocks.
        # Anatomy contributes through concatenation only, never multiplication
        # with or gating of the pressure prediction.
        self.coarse_fusion = ResidualGroupNormBlock(
            coarse * 2 + anatomy_coarse, coarse, groups=groups
        )
        self.middle_fusion = ResidualGroupNormBlock(
            coarse + middle + anatomy_middle, middle, groups=groups
        )
        self.fine_fusion = ResidualGroupNormBlock(
            middle + fine + anatomy_fine, fine, groups=groups
        )
        self._anatomy_frozen = False
        self.set_anatomy_trainable(not freeze_anatomy)

    @classmethod
    def from_anatomy_checkpoint(
        cls,
        anatomy_checkpoint: str | Path | Mapping[str, Any],
        *,
        sam_in_channels: int = 256,
        hidden_channels: Sequence[int] = (32, 64, 96),
        anatomy_context_channels: Sequence[int] = (16, 24, 32),
        output_size: int = 256,
        groups: int = 8,
        freeze_anatomy: bool = True,
        map_location: str | torch.device | None = "cpu",
    ) -> "AnatomyGuidedPressureDecoder":
        """Initialize the registered anatomy model from its pretraining artifact."""

        anatomy_model, _ = build_foot_anatomy_from_checkpoint(
            anatomy_checkpoint,
            map_location=map_location,
            strict=True,
        )
        return cls(
            anatomy_model,
            sam_in_channels=sam_in_channels,
            hidden_channels=hidden_channels,
            anatomy_context_channels=anatomy_context_channels,
            output_size=output_size,
            groups=groups,
            freeze_anatomy=freeze_anatomy,
        )

    @property
    def anatomy_frozen(self) -> bool:
        return self._anatomy_frozen

    def set_anatomy_trainable(self, trainable: bool) -> None:
        """Explicitly opt into fine-tuning; the safe default is frozen."""

        self._anatomy_frozen = not bool(trainable)
        self.anatomy_model.requires_grad_(bool(trainable))
        if self._anatomy_frozen:
            self.anatomy_model.eval()
        else:
            self.anatomy_model.train(self.training)

    def train(self, mode: bool = True) -> "AnatomyGuidedPressureDecoder":
        super().train(mode)
        # nn.Module.train() recursively toggles every child.  Restore the
        # frozen pretrainer to deterministic eval behavior afterwards.
        if getattr(self, "_anatomy_frozen", False):
            self.anatomy_model.eval()
        return self

    def _extract_anatomy_features(self, local_rgb: torch.Tensor) -> FootAnatomyFeatures:
        anatomy_size = int(self.anatomy_model.spec.input_size)
        anatomy_rgb = F.interpolate(
            local_rgb,
            size=(anatomy_size, anatomy_size),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        if self._anatomy_frozen:
            # extract_frozen_features also detaches all returned maps.  The
            # context projection remains outside no_grad and is trainable.
            return self.anatomy_model.extract_frozen_features(anatomy_rgb)
        return self.anatomy_model.extract_features(anatomy_rgb)

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

        anatomy = self._extract_anatomy_features(local_rgb)
        colour = rgb_lab_hsv_features(local_rgb)
        fine = self._apply_mask(self.fine_encoder(colour), local_mask)
        middle = self._apply_mask(self.middle_encoder(fine), local_mask)
        coarse = self._apply_mask(self.coarse_encoder(middle), local_mask)

        anatomy_fine = self._apply_mask(
            self.anatomy_fine_projection(anatomy.pyramid, fine.shape[-2:]), local_mask
        )
        anatomy_middle = self._apply_mask(
            self.anatomy_middle_projection(anatomy.pyramid, middle.shape[-2:]), local_mask
        )
        anatomy_coarse = self._apply_mask(
            self.anatomy_coarse_projection(anatomy.pyramid, coarse.shape[-2:]), local_mask
        )

        sam_feature = self.sam_projection(image_embeddings)
        sam_feature = F.interpolate(
            sam_feature,
            size=coarse.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused_coarse = self.coarse_fusion(
            torch.cat((coarse, sam_feature, anatomy_coarse), dim=1)
        )
        fused_coarse = self._apply_mask(fused_coarse, local_mask)

        up_middle = F.interpolate(
            fused_coarse,
            size=middle.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused_middle = self.middle_fusion(
            torch.cat((up_middle, middle, anatomy_middle), dim=1)
        )
        fused_middle = self._apply_mask(fused_middle, local_mask)
        up_fine = F.interpolate(
            fused_middle,
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused_fine = self.fine_fusion(
            torch.cat((up_fine, fine, anatomy_fine), dim=1)
        )
        fused_fine = self._apply_mask(fused_fine, local_mask)
        return self.output(fused_fine)

    def anatomy_config(self) -> dict[str, Any]:
        """Return a self-contained, checkpoint-reconstruction specification."""

        return {
            "checkpoint_format": "foot_anatomy_pretraining",
            "checkpoint_version": FOOT_ANATOMY_CHECKPOINT_VERSION,
            "spec": self.anatomy_model.config_dict(),
            "feature_channels": list(self.anatomy_feature_channels),
            "context_channels": list(self.anatomy_context_channels),
            "frozen": self.anatomy_frozen,
            "input_resize": [
                int(self.anatomy_model.spec.input_size),
                int(self.anatomy_model.spec.input_size),
            ],
            "fusion": "all_pyramid_levels_concat_projection_per_decoder_scale",
            "fusion_version": ANATOMY_FUSION_VERSION,
            "uses_dense_anatomy_logits": False,
            "pressure_mask_gating": False,
        }


class AnatomyGuidedSamPressureSegmenter(SamPressureSegmenter):
    """Official frozen SAM plus colour FPN and pretrained foot anatomy context."""

    def __init__(
        self,
        sam_checkpoint: str | Path,
        anatomy_checkpoint: str | Path | Mapping[str, Any] | None,
        decoder_channels: Sequence[int] = (32, 64, 96),
        anatomy_context_channels: Sequence[int] = (16, 24, 32),
        output_size: int = 256,
        *,
        anatomy_spec: FootAnatomySpec | Mapping[str, Any] | None = None,
        freeze_anatomy: bool = True,
        anatomy_map_location: str | torch.device | None = "cpu",
    ) -> None:
        # Establish the official SAM preprocessing/postprocessing contract,
        # then replace its baseline decoder with the self-contained decoder.
        super().__init__(
            sam_checkpoint=sam_checkpoint,
            decoder_channels=(128, 64),
            output_size=output_size,
        )
        if anatomy_checkpoint is not None and anatomy_spec is not None:
            raise ValueError("Use anatomy_checkpoint or anatomy_spec, not both")
        if anatomy_checkpoint is not None:
            self.decoder = AnatomyGuidedPressureDecoder.from_anatomy_checkpoint(
                anatomy_checkpoint,
                sam_in_channels=256,
                hidden_channels=decoder_channels,
                anatomy_context_channels=anatomy_context_channels,
                output_size=output_size,
                freeze_anatomy=freeze_anatomy,
                map_location=anatomy_map_location,
            )
        else:
            if anatomy_spec is None:
                raise ValueError(
                    "A fresh model requires anatomy_checkpoint; anatomy_spec is reserved "
                    "for reconstructing a self-contained pressure checkpoint before its "
                    "decoder_state_dict is loaded"
                )
            resolved_spec = (
                anatomy_spec
                if isinstance(anatomy_spec, FootAnatomySpec)
                else FootAnatomySpec.from_dict(anatomy_spec)
            )
            self.decoder = AnatomyGuidedPressureDecoder(
                FootAnatomyModel(resolved_spec, pretrained=False),
                sam_in_channels=256,
                hidden_channels=decoder_channels,
                anatomy_context_channels=anatomy_context_channels,
                output_size=output_size,
                freeze_anatomy=freeze_anatomy,
            )
        # The inherited helper enables the new decoder recursively, so restore
        # the requested anatomy freeze immediately afterwards.
        self.freeze_sam()
        self.decoder.set_anatomy_trainable(not freeze_anatomy)

    @classmethod
    def from_decoder_config(
        cls,
        sam_checkpoint: str | Path,
        config: Mapping[str, Any],
        *,
        anatomy_map_location: str | torch.device | None = "cpu",
    ) -> "AnatomyGuidedSamPressureSegmenter":
        """Rebuild the module shape before loading a decoder-only checkpoint.

        The serialized decoder state already contains the complete anatomy
        network.  Only its specification is needed here to allocate modules;
        callers must immediately load the checkpoint with ``strict=True``.
        """

        architecture = str(config.get("architecture", ""))
        if architecture != ANATOMY_GUIDED_SEGMENTER_ARCHITECTURE:
            raise ValueError(
                "Expected an anatomy-guided decoder config, got "
                f"{architecture!r}"
            )
        if "anatomy_spec" not in config:
            raise ValueError("Anatomy-guided decoder config is missing anatomy_spec")
        return cls(
            sam_checkpoint,
            anatomy_checkpoint=None,
            decoder_channels=tuple(config.get("decoder_channels", (32, 64, 96))),
            anatomy_context_channels=tuple(
                config.get("anatomy_context_channels", (16, 24, 32))
            ),
            output_size=int(config.get("output_size", 256)),
            anatomy_spec=FootAnatomySpec.from_dict(config["anatomy_spec"]),
            freeze_anatomy=bool(config.get("anatomy_frozen", True)),
            anatomy_map_location=anatomy_map_location,
        )

    def recover_raw_rgb(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return recover_rgb_from_sam_tensor(
            pixel_values,
            pixel_mean=self.sam.pixel_mean,
            pixel_std=self.sam.pixel_std,
        )

    def extract_local_rgb(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return zero-padded 0..1 RGB matching the pressure output grid."""

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
        anatomy = self.decoder.anatomy_config()
        return {
            "model_type": self.model_type,
            "decoder_channels": list(self.decoder.hidden_channels),
            "output_size": self.output_size,
            "architecture": ANATOMY_GUIDED_SEGMENTER_ARCHITECTURE,
            "architecture_version": ANATOMY_GUIDED_SEGMENTER_VERSION,
            "color_channels": self.decoder.color_channels,
            "sam_in_channels": self.decoder.sam_in_channels,
            "anatomy_checkpoint_format": anatomy["checkpoint_format"],
            "anatomy_checkpoint_version": anatomy["checkpoint_version"],
            "anatomy_spec": anatomy["spec"],
            "anatomy_feature_channels": anatomy["feature_channels"],
            "anatomy_context_channels": anatomy["context_channels"],
            "anatomy_frozen": anatomy["frozen"],
            "anatomy_input_resize": anatomy["input_resize"],
            "anatomy_fusion": anatomy["fusion"],
            "anatomy_fusion_version": anatomy["fusion_version"],
            "uses_dense_anatomy_logits": anatomy["uses_dense_anatomy_logits"],
            "pressure_mask_gating": anatomy["pressure_mask_gating"],
        }


__all__ = [
    "ANATOMY_FUSION_VERSION",
    "ANATOMY_GUIDED_SEGMENTER_ARCHITECTURE",
    "ANATOMY_GUIDED_SEGMENTER_VERSION",
    "AnatomyGuidedPressureDecoder",
    "AnatomyGuidedSamPressureSegmenter",
]
