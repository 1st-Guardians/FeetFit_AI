"""Single-pass wound SegFormer inference with the project's copied checkpoint."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageOps
import torch

from app.core.config import PROJECT_ROOT, settings
from app.core.weights import weights
from app.wound.preprocessing import REDNESS_POLICY, reduce_redness
from app.wound.segmentation import (
    Geometry,
    PREPROCESSING_POLICY,
    choose_device,
    image_tensor,
    load_checkpoint,
    native_logits,
)


PIPELINE_VERSION = "wound-segformer-b0-redness-single-pass-v2"
MODEL_NAME = "wound_segformer_b0"


class WoundAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class WoundPrediction:
    probability: np.ndarray
    mask: np.ndarray
    # Retained for existing consumers; a single pass has one probability map.
    coarse_probability: np.ndarray
    metrics: dict


class WoundAnalyzer:
    def __init__(self) -> None:
        self.checkpoint_path = (PROJECT_ROOT / Path(weights.wound_segmentation)).resolve()
        if not self.checkpoint_path.is_relative_to(PROJECT_ROOT.resolve()):
            raise WoundAnalysisError("Wound checkpoint must be copied inside this project.")
        if not self.checkpoint_path.is_file():
            raise WoundAnalysisError(f"Local wound checkpoint not found: {self.checkpoint_path}")
        try:
            self.device = choose_device(settings.analysis_device)
            self.model, checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        except (ValueError, KeyError, RuntimeError) as error:
            raise WoundAnalysisError(f"Cannot load the local wound SegFormer: {error}") from error
        self.input_size = int(checkpoint["size"])
        self.threshold = float(checkpoint["threshold"])
        self.redness_reduction = float(settings.wound_redness_reduction)
        self.checkpoint_epoch = int(checkpoint.get("epoch", -1))
        self.weight_keys_converted = bool(checkpoint["inference_weight_keys_converted"])
        self.checkpoint_sha256 = sha256(self.checkpoint_path.read_bytes()).hexdigest()
        self._inference_lock = threading.Lock()

    def _predict_probability(self, photo: Image.Image) -> tuple[np.ndarray, Geometry]:
        tensor, geometry = image_tensor(photo, self.input_size)
        with torch.inference_mode():
            logits = self.model(pixel_values=tensor.unsqueeze(0).to(self.device)).logits
            # Match the supplied CLI: restore logits, then sigmoid in native space.
            probability = native_logits(logits, [geometry], self.input_size)[0].sigmoid()[0]
            probability = probability.cpu().numpy()
        return np.ascontiguousarray(probability, dtype=np.float32), geometry

    def analyze(self, photo: Image.Image) -> WoundPrediction:
        photo = ImageOps.exif_transpose(photo).convert("RGB")
        model_photo = Image.fromarray(reduce_redness(
            np.asarray(photo), "scale", self.redness_reduction,
        ))
        with self._inference_lock:
            probability, geometry = self._predict_probability(model_photo)
        if probability.shape != (photo.height, photo.width) or not np.isfinite(probability).all():
            raise WoundAnalysisError("Wound model returned invalid native-space probabilities.")
        if np.any(probability < 0) or np.any(probability > 1):
            raise WoundAnalysisError("Wound probabilities must be in [0, 1].")
        mask = probability >= self.threshold
        return WoundPrediction(probability, mask, probability, {
            "wound_pipeline_version": PIPELINE_VERSION,
            "wound_model": MODEL_NAME,
            "wound_checkpoint": str(self.checkpoint_path),
            "wound_checkpoint_sha256": self.checkpoint_sha256,
            "wound_checkpoint_epoch": self.checkpoint_epoch,
            "wound_device": str(self.device),
            "wound_input_size": self.input_size,
            "wound_threshold": self.threshold,
            "wound_threshold_source": "checkpoint",
            "wound_weight_keys_converted": self.weight_keys_converted,
            "wound_preprocessing_policy": PREPROCESSING_POLICY,
            "wound_color_preprocessing_policy": REDNESS_POLICY,
            "wound_redness_reduction": self.redness_reduction,
            "wound_red_chroma_gain": 1 - self.redness_reduction,
            "wound_color_preprocessing_applied": self.redness_reduction > 0,
            "wound_color_preprocessing_scope": "wound_model_input_only_before_letterbox",
            "wound_preprocessing_verified_against_training": False,
            "wound_input_geometry": asdict(geometry),
            "wound_raw_pixels": int(mask.sum()),
            "wound_inference_passes": 1,
            "wound_roi_xyxy": None,
            "wound_coordinate_space": "exif_oriented_original",
            "wound_postprocessing": "threshold_only",
            "wound_status": "detected" if mask.any() else "no_detection",
        })


_analyzer: WoundAnalyzer | None = None
_analyzer_lock = threading.Lock()


def get_wound_analyzer() -> WoundAnalyzer:
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = WoundAnalyzer()
    return _analyzer
