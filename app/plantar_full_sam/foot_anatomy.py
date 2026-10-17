"""Foot-specific visual pretraining for downstream pressure and arch models.

This module intentionally does *not* predict flat/normal/cavus labels.  Its
job is to learn what a foot and its visible parts look like before a pressure
segmenter or arch classifier is trained.  The model exposes a reusable
feature pyramid and supports two complementary pretraining signals:

* supervised anatomy tasks: whole-foot silhouette, plantar regions,
  landmarks, laterality, view and image usability;
* label-free masked-image reconstruction and paired-view consistency.

The separation is important: ordinary Internet photographs can teach foot
appearance and anatomy, but they must not be treated as pressure/contact
ground truth.  A downstream model should load the encoder checkpoint and
learn contact segmentation only from acquisition-matched pressure images.

The default encoder is the same timm Swin-Tiny family used by the downstream
classifier.  ``tiny_cnn`` remains available for tests or small offline
experiments.  A timm backbone can be initialized from ImageNet or a local
checkpoint without downloading weights at training time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


FOOT_PART_NAMES = (
    "background",
    "heel",
    "midfoot",
    "forefoot",
    "hallux",
    "lesser_toes",
)

FOOT_LANDMARK_NAMES = (
    "heel_center",
    "hallux_tip",
    "small_toe_tip",
    "second_toe_tip",
    "first_metatarsal_head",
    "fifth_metatarsal_head",
    "medial_midfoot",
    "lateral_midfoot",
    "forefoot_center",
)

FOOT_VIEW_NAMES = ("plantar", "dorsal", "medial", "lateral", "other")
FOOT_LATERALITY_NAMES = ("left", "right")
FOOT_ANATOMY_CHECKPOINT_VERSION = 1


def _as_tuple(value: Sequence[Any], *, name: str) -> tuple[Any, ...]:
    converted = tuple(value)
    if not converted:
        raise ValueError(f"{name} must not be empty")
    return converted


@dataclass(frozen=True)
class FootAnatomySpec:
    """Serializable configuration of :class:`FootAnatomyModel`.

    Inputs to the public model API are RGB tensors in ``[0, 1]``.  ``input_size``
    records the recommended square training crop; the network itself remains
    fully convolutional and accepts other spatial sizes.
    """

    backbone_name: str = "swin_tiny_patch4_window7_224.ms_in1k"
    input_size: int = 224
    base_channels: int = 32
    # A deliberately small shared projection keeps four FPN maps practical on
    # an 8 GB GPU.  Downstream segmentation can consume these maps directly.
    decoder_channels: int = 64
    embedding_dim: int = 256
    part_names: tuple[str, ...] = FOOT_PART_NAMES
    landmark_names: tuple[str, ...] = FOOT_LANDMARK_NAMES
    view_names: tuple[str, ...] = FOOT_VIEW_NAMES
    laterality_names: tuple[str, ...] = FOOT_LATERALITY_NAMES
    timm_out_indices: tuple[int, ...] = (0, 1, 2, 3)
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    reconstruction_enabled: bool = True
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if not self.backbone_name:
            raise ValueError("backbone_name must not be empty")
        if min(
            int(self.input_size),
            int(self.base_channels),
            int(self.decoder_channels),
            int(self.embedding_dim),
        ) <= 0:
            raise ValueError("input and channel dimensions must be positive")
        if len(self.timm_out_indices) < 2:
            raise ValueError("At least two feature levels are required")
        if len(set(int(index) for index in self.timm_out_indices)) != len(
            self.timm_out_indices
        ):
            raise ValueError("timm_out_indices must be unique")
        _as_tuple(self.part_names, name="part_names")
        _as_tuple(self.landmark_names, name="landmark_names")
        _as_tuple(self.view_names, name="view_names")
        if len(self.laterality_names) != 2:
            raise ValueError("laterality_names must contain exactly left and right")
        if len(self.imagenet_mean) != 3 or len(self.imagenet_std) != 3:
            raise ValueError("Image normalization requires three RGB values")
        if min(float(value) for value in self.imagenet_std) <= 0:
            raise ValueError("imagenet_std values must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def num_parts(self) -> int:
        return len(self.part_names)

    @property
    def num_landmarks(self) -> int:
        return len(self.landmark_names)

    @property
    def num_views(self) -> int:
        return len(self.view_names)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FootAnatomySpec":
        """Build a spec from JSON/checkpoint data while rejecting unknown keys."""

        valid_names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - valid_names)
        if unknown:
            raise ValueError("Unknown FootAnatomySpec fields: " + ", ".join(unknown))
        converted = dict(values)
        tuple_fields = {
            "part_names",
            "landmark_names",
            "view_names",
            "laterality_names",
            "timm_out_indices",
            "imagenet_mean",
            "imagenet_std",
        }
        for name in tuple_fields:
            if name in converted:
                converted[name] = tuple(converted[name])
        return cls(**converted)


@dataclass
class FootAnatomyFeatures:
    """Reusable anatomy-aware representation for downstream networks."""

    pyramid: tuple[torch.Tensor, ...]
    embedding: torch.Tensor


@dataclass
class FootAnatomyOutput:
    """All dense and global outputs produced during anatomy pretraining."""

    features: FootAnatomyFeatures
    foot_logits: torch.Tensor
    part_logits: torch.Tensor
    landmark_logits: torch.Tensor
    landmark_coordinates: torch.Tensor
    landmark_confidence: torch.Tensor
    laterality_logits: torch.Tensor
    view_logits: torch.Tensor
    quality_logit: torch.Tensor
    reconstruction: torch.Tensor | None = None
    masked_pixel_mask: torch.Tensor | None = None

    @property
    def feature_pyramid(self) -> tuple[torch.Tensor, ...]:
        return self.features.pyramid

    @property
    def embedding(self) -> torch.Tensor:
        return self.features.embedding


@dataclass
class FootAnatomyTargets:
    """Optional labels for a heterogeneous anatomy-pretraining batch.

    ``foot_mask`` means the *complete visible foot silhouette*, never the
    pressure/contact mask from the current dataset.  Missing dense labels may
    be filled with ``-1``.  Missing class labels use
    ``-1``.  Landmark coordinates are normalized ``(x, y)`` values in
    ``[0, 1]``; non-finite coordinates or false visibility flags are ignored.
    This lets licensed datasets with different annotation subsets be mixed in
    one training loader without inventing labels.
    """

    foot_mask: torch.Tensor | None = None
    part_mask: torch.Tensor | None = None
    landmarks: torch.Tensor | None = None
    landmark_visibility: torch.Tensor | None = None
    landmark_heatmaps: torch.Tensor | None = None
    laterality: torch.Tensor | None = None
    view: torch.Tensor | None = None
    quality: torch.Tensor | None = None
    reconstruction_image: torch.Tensor | None = None


@dataclass(frozen=True)
class FootAnatomyLossWeights:
    foot: float = 1.0
    parts: float = 1.0
    landmarks: float = 1.0
    laterality: float = 0.2
    view: float = 0.2
    quality: float = 0.1
    reconstruction: float = 1.0
    consistency: float = 0.1

    def __post_init__(self) -> None:
        for field in fields(self):
            if float(getattr(self, field.name)) < 0:
                raise ValueError(f"Loss weight {field.name} must be non-negative")


@dataclass
class FootAnatomyLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    active_tasks: tuple[str, ...]


def _group_count(channels: int, maximum: int = 8) -> int:
    return max(group for group in range(min(maximum, channels), 0, -1) if channels % group == 0)


class _ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _ConvNormAct(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(self.conv1(inputs)) + self.skip(inputs))


class _TinyPyramidEncoder(nn.Module):
    """Small four-level encoder used offline and in unit tests."""

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        channels = (
            int(base_channels),
            int(base_channels) * 2,
            int(base_channels) * 4,
            int(base_channels) * 8,
        )
        self.channels = channels
        self.stem = _ConvNormAct(3, channels[0], stride=2)
        self.stages = nn.ModuleList(
            (
                _ResidualBlock(channels[0], channels[0], stride=2),
                _ResidualBlock(channels[0], channels[1], stride=2),
                _ResidualBlock(channels[1], channels[2], stride=2),
                _ResidualBlock(channels[2], channels[3], stride=2),
            )
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feature = self.stem(inputs)
        outputs: list[torch.Tensor] = []
        for stage in self.stages:
            feature = stage(feature)
            outputs.append(feature)
        return tuple(outputs)


class _TimmPyramidEncoder(nn.Module):
    """Lazy timm wrapper retaining classifier-compatible backbone keys.

    We deliberately create the ordinary ``num_classes=0`` model and call
    ``forward_intermediates``.  timm's ``features_only`` wrapper rewrites Swin
    keys from ``layers.0`` to ``layers_0``; those weights cannot be loaded
    directly into the existing whole-foot classifier.  Keeping the ordinary
    model avoids that incompatibility.
    """

    def __init__(
        self,
        model_name: str,
        *,
        out_indices: Sequence[int],
        pretrained: bool,
        pretrained_checkpoint: str | Path | None,
    ) -> None:
        super().__init__()
        try:
            import timm
            from timm.models import load_checkpoint as load_timm_checkpoint
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "A timm anatomy backbone was requested. Install timm with "
                "`python -m pip install timm`."
            ) from exc
        if pretrained and pretrained_checkpoint is not None:
            raise ValueError("Use either pretrained=True or pretrained_checkpoint, not both")
        self.out_indices = tuple(int(index) for index in out_indices)
        self.model = timm.create_model(
            model_name,
            pretrained=bool(pretrained),
            num_classes=0,
        )
        if not hasattr(self.model, "forward_intermediates"):
            raise ValueError(
                f"timm backbone {model_name!r} does not expose forward_intermediates; "
                "choose a compatible Swin/ConvNeXt model"
            )
        if pretrained_checkpoint is not None:
            checkpoint_path = Path(pretrained_checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Anatomy backbone checkpoint not found: {checkpoint_path}")
            load_timm_checkpoint(self.model, str(checkpoint_path), strict=False)
        feature_info = self.model.feature_info
        if hasattr(feature_info, "channels"):
            all_channels = tuple(int(value) for value in feature_info.channels())
        else:
            try:
                all_channels = tuple(int(item["num_chs"]) for item in feature_info)
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Cannot read intermediate channels from timm backbone {model_name!r}"
                ) from exc
        try:
            self.channels = tuple(all_channels[index] for index in self.out_indices)
        except IndexError as exc:
            raise ValueError(
                f"Invalid intermediate indices {self.out_indices} for {len(all_channels)} levels"
            ) from exc

    @staticmethod
    def _as_bchw(feature: torch.Tensor, channels: int) -> torch.Tensor:
        if feature.ndim != 4:
            raise RuntimeError(f"Expected a four-dimensional feature map, got {tuple(feature.shape)}")
        if feature.shape[1] == channels:
            return feature
        if feature.shape[-1] == channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        raise RuntimeError(f"Cannot locate {channels} channels in feature shape {tuple(feature.shape)}")

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.model.forward_intermediates(
            inputs,
            indices=self.out_indices,
            norm=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        return tuple(
            self._as_bchw(feature, channels)
            for feature, channels in zip(features, self.channels)
        )


class _FeaturePyramid(nn.Module):
    def __init__(self, input_channels: Sequence[int], output_channels: int) -> None:
        super().__init__()
        if len(input_channels) < 2:
            raise ValueError("A feature pyramid requires at least two levels")
        self.lateral = nn.ModuleList(
            nn.Conv2d(int(channels), output_channels, kernel_size=1)
            for channels in input_channels
        )
        self.smooth = nn.ModuleList(
            _ResidualBlock(output_channels, output_channels) for _ in input_channels
        )

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(features) != len(self.lateral):
            raise ValueError(
                f"Expected {len(self.lateral)} feature levels, got {len(features)}"
            )
        lateral = [projection(feature) for projection, feature in zip(self.lateral, features)]
        outputs: list[torch.Tensor | None] = [None] * len(lateral)
        current = lateral[-1]
        outputs[-1] = self.smooth[-1](current)
        for index in range(len(lateral) - 2, -1, -1):
            current = lateral[index] + F.interpolate(
                current,
                size=lateral[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            current = self.smooth[index](current)
            outputs[index] = current
        return tuple(feature for feature in outputs if feature is not None)


def landmark_soft_argmax(heatmap_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode heatmaps into differentiable normalized ``(x, y)`` coordinates."""

    if heatmap_logits.ndim != 4:
        raise ValueError(f"Expected landmark logits [B, K, H, W], got {tuple(heatmap_logits.shape)}")
    batch, landmarks, height, width = heatmap_logits.shape
    probabilities = F.softmax(heatmap_logits.flatten(2), dim=2).reshape(
        batch, landmarks, height, width
    )
    x_axis = torch.linspace(
        0.0, 1.0, width, device=heatmap_logits.device, dtype=heatmap_logits.dtype
    ).view(1, 1, 1, width)
    y_axis = torch.linspace(
        0.0, 1.0, height, device=heatmap_logits.device, dtype=heatmap_logits.dtype
    ).view(1, 1, height, 1)
    x_coordinate = (probabilities * x_axis).sum(dim=(2, 3))
    y_coordinate = (probabilities * y_axis).sum(dim=(2, 3))
    coordinates = torch.stack((x_coordinate, y_coordinate), dim=2)
    confidence = heatmap_logits.sigmoid().flatten(2).amax(dim=2)
    return coordinates, confidence


def make_patch_mask(
    images: torch.Tensor,
    *,
    mask_ratio: float = 0.6,
    patch_size: int = 16,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create a boolean BCHW mask with the same sampled ratio per image."""

    if images.ndim != 4:
        raise ValueError(f"Expected BCHW images, got {tuple(images.shape)}")
    if not 0.0 <= float(mask_ratio) <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1]")
    if int(patch_size) <= 0:
        raise ValueError("patch_size must be positive")
    batch, _, height, width = images.shape
    grid_height = (height + int(patch_size) - 1) // int(patch_size)
    grid_width = (width + int(patch_size) - 1) // int(patch_size)
    patch_count = grid_height * grid_width
    masked_count = int(round(patch_count * float(mask_ratio)))
    if masked_count == 0:
        patch_mask = torch.zeros(
            batch, 1, grid_height, grid_width, dtype=torch.bool, device=images.device
        )
    elif masked_count == patch_count:
        patch_mask = torch.ones(
            batch, 1, grid_height, grid_width, dtype=torch.bool, device=images.device
        )
    else:
        random_scores = torch.rand(
            batch,
            patch_count,
            device=images.device,
            generator=generator,
        )
        indices = random_scores.topk(masked_count, dim=1, largest=False).indices
        flattened = torch.zeros(
            batch, patch_count, dtype=torch.bool, device=images.device
        )
        flattened.scatter_(1, indices, True)
        patch_mask = flattened.reshape(batch, 1, grid_height, grid_width)
    return F.interpolate(
        patch_mask.float(), size=(height, width), mode="nearest"
    ).bool()


class FootAnatomyModel(nn.Module):
    """Multi-task foot-anatomy encoder and self-supervised pretrainer."""

    def __init__(
        self,
        spec: FootAnatomySpec | None = None,
        *,
        pretrained: bool = False,
        pretrained_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec or FootAnatomySpec()
        if self.spec.backbone_name == "tiny_cnn":
            if pretrained or pretrained_checkpoint is not None:
                raise ValueError(
                    "tiny_cnn has no external initialization; load a full anatomy checkpoint instead"
                )
            self.encoder: nn.Module = _TinyPyramidEncoder(self.spec.base_channels)
        else:
            self.encoder = _TimmPyramidEncoder(
                self.spec.backbone_name,
                out_indices=self.spec.timm_out_indices,
                pretrained=pretrained,
                pretrained_checkpoint=pretrained_checkpoint,
            )
        encoder_channels = tuple(int(value) for value in getattr(self.encoder, "channels"))
        self.encoder_channels = encoder_channels
        self.fpn = _FeaturePyramid(encoder_channels, int(self.spec.decoder_channels))
        dense_channels = int(self.spec.decoder_channels)
        self.dense_refine = _ResidualBlock(dense_channels, dense_channels)
        self.foot_head = nn.Conv2d(dense_channels, 1, kernel_size=1)
        self.part_head = nn.Conv2d(dense_channels, self.spec.num_parts, kernel_size=1)
        self.landmark_head = nn.Conv2d(
            dense_channels, self.spec.num_landmarks, kernel_size=1
        )

        global_channels = encoder_channels[-1]
        self.embedding_head = nn.Sequential(
            nn.Linear(global_channels, int(self.spec.embedding_dim)),
            nn.LayerNorm(int(self.spec.embedding_dim)),
            nn.GELU(),
            nn.Dropout(float(self.spec.dropout)),
        )
        self.laterality_head = nn.Linear(
            int(self.spec.embedding_dim), len(self.spec.laterality_names)
        )
        self.view_head = nn.Linear(int(self.spec.embedding_dim), self.spec.num_views)
        self.quality_head = nn.Linear(int(self.spec.embedding_dim), 1)
        self.reconstruction_head: nn.Module | None = None
        if self.spec.reconstruction_enabled:
            reconstruction_channels = max(8, dense_channels // 2)
            self.reconstruction_head = nn.Sequential(
                _ConvNormAct(dense_channels, reconstruction_channels),
                nn.Conv2d(reconstruction_channels, 3, kernel_size=1),
            )
        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.register_buffer(
            "image_mean",
            torch.tensor(self.spec.imagenet_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(self.spec.imagenet_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _prepare_inputs(
        self,
        images_0_1: torch.Tensor,
        patch_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if images_0_1.ndim != 4 or images_0_1.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB input, got {tuple(images_0_1.shape)}")
        images = images_0_1.to(device=self.image_mean.device, dtype=torch.float32)
        if not torch.isfinite(images).all():
            raise ValueError("Input images contain non-finite values")
        if bool((images.detach() < -1e-4).any()) or bool((images.detach() > 1.0001).any()):
            raise ValueError("FootAnatomyModel expects RGB values in [0, 1]")
        normalized = (images - self.image_mean) / self.image_std
        if patch_mask is None:
            return normalized, None
        if patch_mask.ndim == 3:
            patch_mask = patch_mask.unsqueeze(1)
        if patch_mask.ndim != 4 or patch_mask.shape[0] != images.shape[0] or patch_mask.shape[1] != 1:
            raise ValueError("patch_mask must have shape [B, 1, H, W] or [B, H, W]")
        mask = F.interpolate(
            patch_mask.to(device=images.device, dtype=torch.float32),
            size=images.shape[-2:],
            mode="nearest",
        ).clamp(0.0, 1.0)
        normalized = normalized * (1.0 - mask) + self.mask_token.to(normalized.dtype) * mask
        return normalized, mask

    def extract_features(
        self,
        images_0_1: torch.Tensor,
        *,
        patch_mask: torch.Tensor | None = None,
    ) -> FootAnatomyFeatures:
        """Return the FPN and global embedding without running task heads."""

        prepared, _ = self._prepare_inputs(images_0_1, patch_mask)
        encoder_features = self.encoder(prepared)
        pyramid = self.fpn(encoder_features)
        pooled = F.adaptive_avg_pool2d(encoder_features[-1], output_size=1).flatten(1)
        embedding = self.embedding_head(pooled)
        return FootAnatomyFeatures(pyramid=pyramid, embedding=embedding)

    @torch.no_grad()
    def extract_frozen_features(
        self,
        images_0_1: torch.Tensor,
        *,
        patch_mask: torch.Tensor | None = None,
    ) -> FootAnatomyFeatures:
        """Memory-efficient feature extraction for frozen downstream use.

        The previous train/eval mode is restored before returning.  Every
        tensor is detached, making accidental pressure/classification gradient
        flow into the anatomy pretraining model impossible.
        """

        was_training = self.training
        self.eval()
        try:
            features = self.extract_features(images_0_1, patch_mask=patch_mask)
            return FootAnatomyFeatures(
                pyramid=tuple(feature.detach() for feature in features.pyramid),
                embedding=features.embedding.detach(),
            )
        finally:
            self.train(was_training)

    def forward(
        self,
        images_0_1: torch.Tensor,
        *,
        patch_mask: torch.Tensor | None = None,
        return_reconstruction: bool | None = None,
    ) -> FootAnatomyOutput:
        prepared, applied_mask = self._prepare_inputs(images_0_1, patch_mask)
        encoder_features = self.encoder(prepared)
        pyramid = self.fpn(encoder_features)
        spatial_size = images_0_1.shape[-2:]
        dense = self.dense_refine(
            F.interpolate(
                pyramid[0], size=spatial_size, mode="bilinear", align_corners=False
            )
        )
        pooled = F.adaptive_avg_pool2d(encoder_features[-1], output_size=1).flatten(1)
        embedding = self.embedding_head(pooled)
        landmark_logits = self.landmark_head(dense)
        landmark_coordinates, landmark_confidence = landmark_soft_argmax(landmark_logits)

        if return_reconstruction is None:
            return_reconstruction = applied_mask is not None
        reconstruction: torch.Tensor | None = None
        if return_reconstruction:
            if self.reconstruction_head is None:
                raise RuntimeError("Reconstruction was requested but is disabled in the spec")
            reconstruction = self.reconstruction_head(dense).sigmoid()

        return FootAnatomyOutput(
            features=FootAnatomyFeatures(pyramid=pyramid, embedding=embedding),
            foot_logits=self.foot_head(dense),
            part_logits=self.part_head(dense),
            landmark_logits=landmark_logits,
            landmark_coordinates=landmark_coordinates,
            landmark_confidence=landmark_confidence,
            laterality_logits=self.laterality_head(embedding),
            view_logits=self.view_head(embedding),
            quality_logit=self.quality_head(embedding).squeeze(1),
            reconstruction=reconstruction,
            masked_pixel_mask=applied_mask,
        )

    def set_encoder_trainable(self, trainable: bool, *, include_fpn: bool = True) -> None:
        """Freeze/unfreeze transferred representation parameters explicitly."""

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(bool(trainable))
        if include_fpn:
            for parameter in self.fpn.parameters():
                parameter.requires_grad_(bool(trainable))
            for parameter in self.embedding_head.parameters():
                parameter.requires_grad_(bool(trainable))

    def representation_parameters(self):
        """Yield encoder/FPN/embedding parameters for a separate optimizer LR."""

        yield from self.encoder.parameters()
        yield from self.fpn.parameters()
        yield from self.embedding_head.parameters()

    @property
    def projected_feature_channels(self) -> tuple[int, ...]:
        """Channels of each returned FPN level, ordered fine to coarse."""

        return (int(self.spec.decoder_channels),) * len(self.encoder_channels)

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        """Return state keys compatible with the underlying timm backbone.

        For timm encoders the wrapper's ``model.`` prefix is deliberately
        removed.  The result can therefore initialize
        ``SwinPressureArchClassifier.backbone`` with ``strict=False``.  The
        tiny fallback returns its native encoder keys.
        """

        backbone = getattr(self.encoder, "model", self.encoder)
        return {key: value.detach().cpu() for key, value in backbone.state_dict().items()}

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.spec)


# The alias makes the intended training-stage role explicit without forcing
# downstream code to use a second model type.
FootAnatomyPretrainer = FootAnatomyModel


def _resize_binary_target(target: torch.Tensor, spatial_size: Sequence[int]) -> torch.Tensor:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(f"Expected a binary target [B, 1, H, W], got {tuple(target.shape)}")
    return F.interpolate(target.float(), size=tuple(spatial_size), mode="nearest")


def _binary_segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = _resize_binary_target(target.to(logits.device), logits.shape[-2:])
    valid = target >= 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    target = target.clamp(0.0, 1.0)
    binary_cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    binary_cross_entropy = (binary_cross_entropy * valid).sum() / valid.sum().clamp_min(1)
    probability = logits.sigmoid() * valid
    target = target * valid
    reduce_dims = (1, 2, 3)
    intersection = (probability * target).sum(dim=reduce_dims)
    denominator = probability.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return binary_cross_entropy + dice_loss


def _part_segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError(f"Expected a part target [B, H, W], got {tuple(target.shape)}")
    target = F.interpolate(
        target.to(device=logits.device, dtype=torch.float32).unsqueeze(1),
        size=logits.shape[-2:],
        mode="nearest",
    ).squeeze(1).long()
    invalid_positive = (target >= logits.shape[1]) & (target != -1)
    if bool(invalid_positive.any()):
        raise ValueError("part_mask contains a class index outside part_logits")
    if not bool((target != -1).any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits, target, ignore_index=-1)


def _class_loss(logits: torch.Tensor, target: torch.Tensor, *, name: str) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=torch.long).reshape(-1)
    if target.shape[0] != logits.shape[0]:
        raise ValueError(f"{name} labels and output batch sizes differ")
    valid = target >= 0
    if bool(((target >= logits.shape[1]) & valid).any()):
        raise ValueError(f"{name} contains a class index outside its logits")
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], target[valid])


def _landmark_loss(output: FootAnatomyOutput, targets: FootAnatomyTargets) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    if targets.landmarks is not None:
        coordinates = targets.landmarks.to(
            device=output.landmark_coordinates.device,
            dtype=output.landmark_coordinates.dtype,
        )
        if coordinates.shape != output.landmark_coordinates.shape:
            raise ValueError(
                "landmarks must match predicted [B, K, 2] coordinates; "
                f"got {tuple(coordinates.shape)}"
            )
        visible = torch.isfinite(coordinates).all(dim=2)
        if targets.landmark_visibility is not None:
            supplied_visibility = targets.landmark_visibility.to(
                device=visible.device, dtype=torch.bool
            )
            if supplied_visibility.shape != visible.shape:
                raise ValueError("landmark_visibility must have shape [B, K]")
            visible = visible & supplied_visibility
        if bool(visible.any()):
            losses.append(
                F.smooth_l1_loss(
                    output.landmark_coordinates[visible], coordinates[visible], beta=0.02
                )
            )
    if targets.landmark_heatmaps is not None:
        heatmaps = targets.landmark_heatmaps.to(
            device=output.landmark_logits.device, dtype=output.landmark_logits.dtype
        )
        if heatmaps.ndim != 4 or heatmaps.shape[:2] != output.landmark_logits.shape[:2]:
            raise ValueError("landmark_heatmaps must have shape [B, K, H, W]")
        heatmaps = F.interpolate(
            heatmaps, size=output.landmark_logits.shape[-2:], mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        if targets.landmark_visibility is None:
            visible = torch.ones(
                heatmaps.shape[:2], device=heatmaps.device, dtype=torch.bool
            )
        else:
            visible = targets.landmark_visibility.to(device=heatmaps.device, dtype=torch.bool)
            if visible.shape != heatmaps.shape[:2]:
                raise ValueError("landmark_visibility must have shape [B, K]")
        if bool(visible.any()):
            heatmap_loss = F.binary_cross_entropy_with_logits(
                output.landmark_logits, heatmaps, reduction="none"
            ).mean(dim=(2, 3))
            losses.append(heatmap_loss[visible].mean())
    if not losses:
        return output.landmark_logits.sum() * 0.0
    return torch.stack(losses).mean()


def anatomy_embedding_consistency_loss(
    first: FootAnatomyOutput | FootAnatomyFeatures,
    second: FootAnatomyOutput | FootAnatomyFeatures,
) -> torch.Tensor:
    """Cosine consistency for two augmentations of the same unlabeled foot."""

    first_embedding = first.embedding
    second_embedding = second.embedding
    if first_embedding.shape != second_embedding.shape:
        raise ValueError("Paired anatomy embeddings must have identical shapes")
    return (1.0 - F.cosine_similarity(first_embedding, second_embedding, dim=1)).mean()


def foot_anatomy_multitask_loss(
    output: FootAnatomyOutput,
    targets: FootAnatomyTargets | None = None,
    *,
    weights: FootAnatomyLossWeights | None = None,
    paired_output: FootAnatomyOutput | FootAnatomyFeatures | None = None,
) -> FootAnatomyLossOutput:
    """Compute only the losses whose labels/signals are present.

    A zero, differentiable total is returned for an entirely unlabeled batch.
    In practice such a batch should provide either a masked reconstruction or
    a paired augmented view.
    """

    targets = targets or FootAnatomyTargets()
    weights = weights or FootAnatomyLossWeights()
    components: dict[str, torch.Tensor] = {}

    if targets.foot_mask is not None and weights.foot > 0:
        components["foot"] = _binary_segmentation_loss(output.foot_logits, targets.foot_mask)
    if targets.part_mask is not None and weights.parts > 0:
        components["parts"] = _part_segmentation_loss(output.part_logits, targets.part_mask)
    if (
        targets.landmarks is not None or targets.landmark_heatmaps is not None
    ) and weights.landmarks > 0:
        components["landmarks"] = _landmark_loss(output, targets)
    if targets.laterality is not None and weights.laterality > 0:
        components["laterality"] = _class_loss(
            output.laterality_logits, targets.laterality, name="laterality"
        )
    if targets.view is not None and weights.view > 0:
        components["view"] = _class_loss(output.view_logits, targets.view, name="view")
    if targets.quality is not None and weights.quality > 0:
        quality = targets.quality.to(
            device=output.quality_logit.device, dtype=output.quality_logit.dtype
        ).reshape(-1)
        if quality.shape != output.quality_logit.shape:
            raise ValueError("quality labels and output batch sizes differ")
        valid = torch.isfinite(quality) & (quality >= 0)
        if bool(valid.any()):
            components["quality"] = F.binary_cross_entropy_with_logits(
                output.quality_logit[valid], quality[valid].clamp(0.0, 1.0)
            )
    if targets.reconstruction_image is not None and weights.reconstruction > 0:
        if output.reconstruction is None or output.masked_pixel_mask is None:
            raise ValueError(
                "Reconstruction targets require forward(..., patch_mask=..., "
                "return_reconstruction=True)"
            )
        target_image = targets.reconstruction_image.to(
            device=output.reconstruction.device, dtype=output.reconstruction.dtype
        )
        if target_image.shape != output.reconstruction.shape:
            target_image = F.interpolate(
                target_image,
                size=output.reconstruction.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        mask = F.interpolate(
            output.masked_pixel_mask.to(dtype=output.reconstruction.dtype),
            size=output.reconstruction.shape[-2:],
            mode="nearest",
        )
        denominator = (mask.sum() * output.reconstruction.shape[1]).clamp_min(1.0)
        components["reconstruction"] = (
            F.smooth_l1_loss(output.reconstruction, target_image, reduction="none") * mask
        ).sum() / denominator
    if paired_output is not None and weights.consistency > 0:
        components["consistency"] = anatomy_embedding_consistency_loss(output, paired_output)

    weight_lookup = asdict(weights)
    if components:
        total = sum(
            component * float(weight_lookup[name])
            for name, component in components.items()
        )
    else:
        total = output.embedding.sum() * 0.0
    return FootAnatomyLossOutput(
        total=total,
        components=components,
        active_tasks=tuple(components),
    )


def save_foot_anatomy_checkpoint(
    path: str | Path,
    model: FootAnatomyModel,
    *,
    epoch: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Save a self-describing anatomy checkpoint for later feature transfer."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "foot_anatomy_pretraining",
        "version": FOOT_ANATOMY_CHECKPOINT_VERSION,
        "spec": model.config_dict(),
        "model_state": model.state_dict(),
        "backbone_state": model.backbone_state_dict(),
        "encoder_channels": tuple(model.encoder_channels),
        "projected_feature_channels": model.projected_feature_channels,
        "epoch": None if epoch is None else int(epoch),
        "metrics": dict(metrics or {}),
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _load_payload(
    source: str | Path | Mapping[str, Any],
    *,
    map_location: str | torch.device | None,
) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - PyTorch before weights_only
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("Foot-anatomy checkpoint must contain a mapping")
    return payload


def load_foot_anatomy_checkpoint(
    model: FootAnatomyModel,
    source: str | Path | Mapping[str, Any],
    *,
    strict: bool = True,
    map_location: str | torch.device | None = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load weights and return non-tensor training metadata."""

    payload = _load_payload(source, map_location=map_location)
    if payload.get("format") != "foot_anatomy_pretraining":
        raise ValueError("Not a foot_anatomy_pretraining checkpoint")
    version = int(payload.get("version", -1))
    if version != FOOT_ANATOMY_CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported foot-anatomy checkpoint version: {version}")
    saved_spec = FootAnatomySpec.from_dict(payload["spec"])
    if strict and saved_spec != model.spec:
        raise ValueError("Checkpoint FootAnatomySpec does not match the target model")
    incompatible = model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return {
        "epoch": payload.get("epoch"),
        "metrics": dict(payload.get("metrics") or {}),
        "extra": dict(payload.get("extra") or {}),
        "missing_keys": tuple(incompatible.missing_keys),
        "unexpected_keys": tuple(incompatible.unexpected_keys),
        "spec": saved_spec,
    }


def build_foot_anatomy_from_checkpoint(
    source: str | Path | Mapping[str, Any],
    *,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> tuple[FootAnatomyModel, dict[str, Any]]:
    """Reconstruct a model directly from its serialized specification."""

    payload = _load_payload(source, map_location=map_location)
    if "spec" not in payload:
        raise ValueError("Foot-anatomy checkpoint is missing its spec")
    spec = FootAnatomySpec.from_dict(payload["spec"])
    model = FootAnatomyModel(spec, pretrained=False)
    metadata = load_foot_anatomy_checkpoint(
        model, payload, strict=strict, map_location=map_location
    )
    return model, metadata


def extract_foot_anatomy_backbone_state(
    source: str | Path | Mapping[str, Any],
    *,
    map_location: str | torch.device | None = "cpu",
) -> tuple[dict[str, torch.Tensor], FootAnatomySpec]:
    """Read only the transferable backbone weights from an anatomy checkpoint."""

    payload = _load_payload(source, map_location=map_location)
    if payload.get("format") != "foot_anatomy_pretraining":
        raise ValueError("Not a foot_anatomy_pretraining checkpoint")
    if "backbone_state" not in payload:
        raise ValueError("Checkpoint does not contain a transferable backbone_state")
    state = payload["backbone_state"]
    if not isinstance(state, Mapping):
        raise ValueError("backbone_state must be a mapping")
    return dict(state), FootAnatomySpec.from_dict(payload["spec"])


def load_foot_anatomy_backbone_into(
    target_backbone: nn.Module,
    source: str | Path | Mapping[str, Any],
    *,
    strict: bool = False,
    map_location: str | torch.device | None = "cpu",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Initialize a downstream visual backbone from foot pretraining.

    ``strict=False`` is the intended setting when transferring a timm
    ``features_only`` Swin encoder into a classification Swin, because the
    latter owns an additional classification/head surface.  Missing and
    unexpected keys are returned for explicit logging.
    """

    state, _ = extract_foot_anatomy_backbone_state(source, map_location=map_location)
    incompatible = target_backbone.load_state_dict(state, strict=strict)
    return tuple(incompatible.missing_keys), tuple(incompatible.unexpected_keys)


__all__ = [
    "FOOT_ANATOMY_CHECKPOINT_VERSION",
    "FOOT_LANDMARK_NAMES",
    "FOOT_LATERALITY_NAMES",
    "FOOT_PART_NAMES",
    "FOOT_VIEW_NAMES",
    "FootAnatomyFeatures",
    "FootAnatomyLossOutput",
    "FootAnatomyLossWeights",
    "FootAnatomyModel",
    "FootAnatomyOutput",
    "FootAnatomyPretrainer",
    "FootAnatomySpec",
    "FootAnatomyTargets",
    "anatomy_embedding_consistency_loss",
    "build_foot_anatomy_from_checkpoint",
    "foot_anatomy_multitask_loss",
    "extract_foot_anatomy_backbone_state",
    "landmark_soft_argmax",
    "load_foot_anatomy_checkpoint",
    "load_foot_anatomy_backbone_into",
    "make_patch_mask",
    "save_foot_anatomy_checkpoint",
]
