from pathlib import Path

from app.core.config import settings


class WeightPaths:
    """Central place for all local model weight file paths."""

    root: Path = settings.weights_dir
    foot_segmentation: Path = root / "foot_seg_yolo11n_best.pt"
    tinea_pedis_segmentation: Path = root / "tinea_segformer_b2_teacher_student_best.pt"
    wound_segmentation: Path = settings.wound_segmentation_checkpoint
    sam_checkpoint: Path = root / "sam_vit_b_01ec64.pth"
    hallux_landmark: Path = settings.hallux_landmark_weights
    plantar_foot_segmentation: Path = root / "foot_seg_yolo11n_best.pt"
    plantar_contact_segformer: Path = settings.plantar_contact_segformer_checkpoint
    plantar_full_sam: Path = settings.plantar_full_sam_checkpoint
    plantar_sam: Path = settings.plantar_sam_checkpoint


weights = WeightPaths()
