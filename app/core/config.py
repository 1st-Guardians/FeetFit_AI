from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, AnyHttpUrl, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ChArUco's two_marker_foot_length profile is vendored as the active
# measurement contract. These are foot-board dimensions, not the 7x10 camera
# calibration board dimensions.
CHARUCO_FOOT_RIG_PROFILE_COMMIT = (
    "36b4db11682b160941ff66b2761907c04f0ad4f1"
)
CHARUCO_FOOT_REQUIRED_MARKER_IDS = (1, 4)
CHARUCO_FOOT_MARKER_SIZE_MM = 20.0
CHARUCO_FOOT_MARKER_EDGE_GAP_MM = 175.0
# Legacy uniform correction; active per-foot offsets are configured below.
CHARUCO_FOOT_HEEL_OFFSET_MM = 114.0
CHARUCO_FOOT_BOARD_WIDTH_MM = 365.0
CHARUCO_FOOT_MARKER_CENTER_X_MM = 181.0
CHARUCO_FOOT_PIXELS_PER_MM = 4.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Feetfit AI Server"
    app_version: str = "0.1.0"
    # Opt-in local capture of rejected dorsal marker images, bounded to 12.
    photo_capture_diagnostics_enabled: bool = False
    public_base_url: AnyHttpUrl | None = None
    weights_dir: Path = Field(default=PROJECT_ROOT / "weights")
    sam_source_dir: Path = Field(
        default=PROJECT_ROOT
    )
    backend_base_url: str = "http://127.0.0.1:8080"
    tinea_report_endpoint: str = "http://127.0.0.1:8080/api/reports/tina-pedis"
    hallux_valgus_report_endpoint: str = "http://127.0.0.1:8080/api/reports/hallux-valgus"
    plantar_footprint_report_endpoint: str = (
        "http://127.0.0.1:8080/api/reports/daily-foot-analysis/plantar-footprint"
    )
    pressure_heatmap_report_endpoint: str = (
        "http://127.0.0.1:8080/api/reports/daily-foot-analysis/pressure-heatmap"
    )
    pressure_distribution_report_endpoint: str = (
        "http://127.0.0.1:8080/api/reports/daily-foot-analysis/pressure"
    )
    pressure_balance_report_endpoint: str = (
        "http://127.0.0.1:8080/api/reports/daily-foot-analysis/balance"
    )
    daily_metrics_report_endpoint: str = (
        "http://127.0.0.1:8080/api/reports/daily-foot-analysis/metrics"
    )
    environment_report_endpoint: str = (
        "http://127.0.0.1:8080/api/reports/daily-foot-analysis/environment"
    )
    summary_report_endpoint: str = "http://127.0.0.1:8080/api/reports/summary"
    measurement_status_endpoint_template: str = (
        "http://127.0.0.1:8080/api/measurement-sessions/"
        "{measurement_session_id}/status"
    )
    measurement_failure_notification_timeout_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=15.0,
    )
    shoe_recommendation_endpoint: str = Field(
        default="http://127.0.0.1:8080/api/shoes/recommendations",
        validation_alias=AliasChoices(
            "FEETFIT_SERVER_RECOMMENDATION_ENDPOINT",
            "SHOE_RECOMMENDATION_ENDPOINT",
        ),
    )
    shoe_summary_save_endpoint_template: str = Field(
        default="http://127.0.0.1:8080/api/shoes/{shoe_id}/summaries",
        validation_alias=AliasChoices(
            "FEETFIT_SERVER_SUMMARY_SAVE_ENDPOINT_TEMPLATE",
            "SHOE_SUMMARY_SAVE_ENDPOINT_TEMPLATE",
        ),
    )
    shoe_recommendation_context_endpoint: str = Field(
        default="http://127.0.0.1:8080/api/internal/shoe-analysis/recommendation-context",
        validation_alias=AliasChoices(
            "FEETFIT_SERVER_RECOMMENDATION_CONTEXT_ENDPOINT",
            "SHOE_RECOMMENDATION_CONTEXT_ENDPOINT",
        ),
    )
    shoe_summary_context_endpoint_template: str = Field(
        default=(
            "http://127.0.0.1:8080/api/internal/shoe-analysis/shoes/"
            "{shoe_id}/recommendation-summary-context"
        ),
        validation_alias=AliasChoices(
            "FEETFIT_SERVER_SUMMARY_CONTEXT_ENDPOINT_TEMPLATE",
            "SHOE_SUMMARY_CONTEXT_ENDPOINT_TEMPLATE",
        ),
    )
    shoe_characteristics_endpoint_template: str = Field(
        default="http://127.0.0.1:8080/api/shoes/{shoe_id}/characteristics",
        validation_alias=AliasChoices(
            "FEETFIT_SERVER_CHARACTERISTICS_ENDPOINT_TEMPLATE",
            "SHOE_CHARACTERISTICS_ENDPOINT_TEMPLATE",
        ),
    )
    shoe_recommendation_context_page_size: int = Field(default=200, ge=1, le=200)
    feetfit_server_internal_api_key: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices(
            "FEETFIT_SERVER_INTERNAL_API_KEY",
            "INTERNAL_API_KEY",
        ),
    )
    report_proxy_timeout_seconds: float = Field(default=60.0, gt=0, le=3600)
    feetfit_server_callback_timeout_seconds: float = Field(default=900.0, ge=900.0)
    tinea_handoff_ttl_seconds: int = Field(default=1800, ge=1)
    tinea_handoff_max_sessions: int = Field(default=16, ge=1, le=256)

    openai_api_key: str | None = Field(default=None, repr=False)
    openai_report_model: str = "gpt-6-astra"
    openai_report_trust_env: bool = False
    openai_report_reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    openai_report_min_output_tokens: int = Field(default=8000, ge=1024, le=128000)
    openai_report_max_output_tokens: int = Field(default=16000, ge=1024, le=128000)
    # Standalone/smaller report-text calls remain bounded. The combined full-photo
    # call explicitly disables read/write timeouts because it can submit many
    # high-detail images; an overall request/retry deadline still bounds it.
    openai_report_timeout_seconds: float = Field(default=180.0, ge=5.0, le=600.0)
    # Fail before the overall deadline when a TCP/TLS connection cannot be
    # established or the client pool is exhausted.
    openai_report_connect_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    openai_report_pool_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    openai_report_max_attempts: int = Field(default=5, ge=1, le=5)
    openai_report_retry_max_delay_seconds: float = Field(default=30.0, ge=0, le=60)
    openai_report_full_photo_timeout_seconds: float = Field(default=240.0, ge=30, le=600)
    openai_report_fallback_enabled: bool = True
    openai_report_text_enabled: bool = True
    openai_foot_type_text_enabled: bool = True
    foot_type_pressure_balance_tolerance_percent: float = Field(
        default=5.0, ge=0, le=25
    )
    openai_report_include_images: bool = True

    # The active plantar-contact path first extracts the complete foot with
    # YOLO, then runs the dedicated fine-tuned SegFormer-B0 checkpoint on a
    # white-background foot crop.  The legacy decoder/SAM paths remain
    # configurable only for source and test compatibility.
    plantar_contact_segformer_checkpoint: Path = Field(
        default=(
            PROJECT_ROOT
            / "weights"
            / "plantar_contact_segformer_b0_real_finetune_brightness2_best.pt"
        )
    )
    plantar_contact_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    plantar_contact_crop_padding_ratio: float = Field(
        default=0.06,
        ge=0.0,
        le=0.50,
    )
    plantar_full_sam_checkpoint: Path = Field(
        default=PROJECT_ROOT / "weights" / "plantar_full_sam_v4_decoder.pt"
    )
    plantar_sam_checkpoint: Path = Field(
        default=PROJECT_ROOT / "weights" / "sam_vit_b_01ec64.pth"
    )
    plantar_output_dir: Path = Field(
        default=PROJECT_ROOT / "storage" / "pressed-regions"
    )
    pressure_heatmap_output_dir: Path = Field(
        default=PROJECT_ROOT / "storage" / "pressure-heatmaps"
    )
    # Keep the temporary reference frame active until the hardware is
    # calibrated.  Switching this to "hardware" makes the pressure POST use
    # the two validated arrays from the request without changing its schema.
    pressure_sensor_input_mode: Literal["fixed", "hardware"] = "hardware"
    # Provisional measurement-presence gate, not a diagnostic balance cutoff.
    pressure_min_foot_load_percent: float = Field(default=5.0, ge=0.0, lt=50.0)
    pressure_presence_sensor_peak_percent: float = Field(default=5.0, gt=0.0, le=100.0)
    plantar_canvas_width: int = Field(default=768, ge=256, le=4096)
    plantar_canvas_height: int = Field(default=1536, ge=512, le=8192)
    plantar_canvas_margin_ratio: float = Field(default=0.06, ge=0.0, le=0.25)
    plantar_inference_device: Literal["auto", "cpu", "cuda"] = "auto"
    # Legacy decoder/SAM selection limits; inactive in SegFormer production.
    plantar_sam_min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    plantar_max_contact_ratio: float = Field(default=0.70, gt=0.0, le=1.0)
    plantar_min_orientation_confidence: float = Field(default=0.10, ge=0.0, le=1.0)
    plantar_foot_roi_confidence: float = Field(default=0.20, ge=0.0, le=1.0)
    plantar_foot_roi_image_size: int = Field(default=960, ge=256, le=4096)
    plantar_foot_roi_min_component_area: int = Field(default=1200, ge=1)
    # Retained for legacy geometry helpers. The active model crop uses the
    # dedicated 0 px / plantar_contact_crop_padding_ratio contract above.
    plantar_foot_roi_padding: int = Field(default=28, ge=0, le=1024)
    plantar_foot_roi_padding_ratio: float = Field(default=0.08, ge=0.0, le=0.50)
    plantar_transform_tta: bool = True
    plantar_foot_roi_target_compatible_mode: bool = True

    # Legacy GPT-assisted pressed-region settings remain temporarily readable so
    # existing deployments/tests do not break, but the plantar API route does not
    # import or consume them.
    openai_pressed_region_model: Literal["gpt-5.4"] = "gpt-5.4"
    openai_pressed_region_reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh"
    ] = "low"
    openai_pressed_region_image_detail: Literal["original"] = "original"
    openai_pressed_region_max_output_tokens: int = Field(
        default=25000, ge=4096, le=128000
    )
    openai_pressed_region_timeout_seconds: float = Field(
        default=180.0, ge=5.0, le=600.0
    )
    pressed_region_output_dir: Path = Field(
        default=PROJECT_ROOT / "storage" / "pressed-regions"
    )
    pressed_region_overlay_alpha: float = 0.45
    pressed_region_canvas_width: int = Field(default=768, ge=256, le=4096)
    pressed_region_canvas_height: int = Field(default=1536, ge=512, le=8192)
    pressed_region_canvas_margin_ratio: float = Field(default=0.06, ge=0.0, le=0.25)
    pressed_region_reference_dataset_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "reference-images"
    )
    pressed_region_reference_annotation_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "reference-masks"
    )
    pressed_region_reference_max_pairs: int = Field(default=3, ge=0, le=6)
    # Normalized scan-panel location used only when full-frame YOLO finds no sole.
    pressed_region_panel_center_x_ratio: float = Field(default=0.50, ge=0.0, le=1.0)
    pressed_region_panel_center_y_ratio: float = Field(default=0.44, ge=0.0, le=1.0)
    pressed_region_panel_crop_width_ratio: float = Field(default=0.36, ge=0.15, le=1.0)
    pressed_region_panel_crop_height_ratio: float = Field(default=0.36, ge=0.15, le=1.0)
    pressed_region_max_heel_forefoot_ratio: float = Field(
        default=0.90, ge=0.55, le=1.0
    )
    pressed_region_max_endpoint_width_ratio: float = Field(
        default=0.80, ge=0.40, le=1.0
    )
    pressed_region_seed_radius_ratio: float = 0.012
    pressed_region_support_dilation_ratio: float = 0.06
    pressed_region_grabcut_iterations: int = 5
    pressed_region_min_component_ratio: float = 0.0005
    pressed_region_max_hole_ratio: float = 0.001
    pressed_region_max_region_ratio: float = 0.90
    pressed_region_max_total_ratio: float = 0.92
    pressed_region_min_seed_expansion_ratio: float = 1.10
    pressed_region_max_support_fill_ratio: float = 0.95
    pressed_region_prior_dominance_min_area_ratio: float = 0.08
    pressed_region_prior_dominance_min_coarse_retention: float = 0.95
    pressed_region_prior_dominance_min_support_fill: float = 0.85
    pressed_region_min_stability_iou: float = 0.35
    pressed_region_min_boundary_support: float = 0.12
    pressed_region_min_feature_separation: float = 0.12
    pressed_region_min_quality_score: float = 0.50
    pressed_region_glare_value_threshold: int = 248
    pressed_region_glare_saturation_threshold: int = 32
    pressed_region_glare_local_contrast: float = 10.0

    shoe_embedding_model_name: str = "BAAI/bge-m3"
    shoe_embedding_local_files_only: bool = True
    shoe_embedding_device: str = "auto"
    shoe_review_embedding_cache_path: Path = Field(default=PROJECT_ROOT / ".cache" / "shoe_review_embeddings.npz")
    shoe_release_embedding_model_after_batch: bool = True
    shoe_recommendation_batch_lock_timeout_seconds: float = Field(default=1.0, ge=0, le=30)
    shoe_max_candidate_reviews_per_reason: int = Field(default=40, ge=1, le=200)
    shoe_reviews_per_reason: int = Field(default=3, ge=1, le=3)
    shoe_review_semantic_min_score: float = Field(default=0.42, ge=-1, le=1)

    # Phase D scoring policy. These are intentionally configurable and remain
    # TEMPORARY_HEURISTIC / NOT_CLINICALLY_VALIDATED until a validated policy replaces them.
    shoe_risk_low_min_score: float = Field(default=75.0, ge=0, le=100)
    shoe_risk_medium_min_score: float = Field(default=50.0, ge=0, le=100)
    shoe_forefoot_area_weight: float = Field(default=0.40, gt=0)
    shoe_heel_area_weight: float = Field(default=0.30, gt=0)
    shoe_insole_area_weight: float = Field(default=0.30, gt=0)
    shoe_forefoot_width_ratio_neutral: float = Field(default=0.36, gt=0)
    shoe_forefoot_width_ratio_high: float = Field(default=0.43, gt=0)
    shoe_hallux_angle_neutral_degree: float = Field(default=10.0, ge=0)
    shoe_hallux_angle_high_degree: float = Field(default=30.0, gt=0)
    shoe_forefoot_width_allowance_mm: float = Field(default=5.0, ge=0)
    shoe_forefoot_width_extra_allowance_mm: float = Field(default=4.0, ge=0)
    shoe_forefoot_width_excess_free_mm: float = Field(default=8.0, ge=0)
    shoe_forefoot_width_shortfall_penalty_per_mm: float = Field(default=9.0, ge=0)
    shoe_forefoot_width_excess_penalty_per_mm: float = Field(default=2.0, ge=0)
    shoe_pressure_imbalance_neutral_percent: float = Field(default=5.0, ge=0)
    shoe_pressure_imbalance_high_percent: float = Field(default=20.0, gt=0)
    shoe_balance_neutral_score: float = Field(default=85.0, ge=0, le=100)
    shoe_balance_high_risk_score: float = Field(default=50.0, ge=0, le=100)
    shoe_rearfoot_pressure_neutral_percent: float = Field(default=50.0, ge=0, le=100)
    shoe_rearfoot_pressure_high_percent: float = Field(default=75.0, ge=0, le=100)
    shoe_forefoot_pressure_neutral_percent: float = Field(default=50.0, ge=0, le=100)
    shoe_forefoot_pressure_high_percent: float = Field(default=75.0, ge=0, le=100)
    shoe_skin_safety_neutral_score: float = Field(default=85.0, ge=0, le=100)
    shoe_skin_safety_high_risk_score: float = Field(default=45.0, ge=0, le=100)
    shoe_humidity_neutral_percent: float = Field(default=50.0, ge=0, le=100)
    shoe_humidity_high_percent: float = Field(default=75.0, ge=0, le=100)
    shoe_fungal_safety_neutral_score: float = Field(default=85.0, ge=0, le=100)
    shoe_fungal_safety_high_risk_score: float = Field(default=45.0, ge=0, le=100)
    shoe_metric_target_tolerance: float = Field(default=0.75, gt=0, le=1)
    shoe_forefoot_width_component_weight: float = Field(default=0.55, ge=0)
    shoe_forefoot_toebox_component_weight: float = Field(default=0.45, ge=0)
    shoe_heel_hold_component_weight: float = Field(default=0.40, ge=0)
    shoe_heel_shock_component_weight: float = Field(default=0.35, ge=0)
    shoe_heel_energy_component_weight: float = Field(default=0.15, ge=0)
    shoe_heel_cushion_component_weight: float = Field(default=0.10, ge=0)
    shoe_insole_breathability_component_weight: float = Field(default=0.45, ge=0)
    shoe_insole_cushion_component_weight: float = Field(default=0.30, ge=0)
    shoe_insole_shock_component_weight: float = Field(default=0.25, ge=0)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_temperature: float = Field(default=0.0, ge=0, le=2)
    ollama_request_timeout_seconds: float = Field(default=120.0, gt=0, le=3600)
    ollama_num_gpu: int = Field(default=-1, ge=-1)
    ollama_cpu_fallback_enabled: bool = True
    ollama_max_concurrency: int = Field(default=1, ge=1, le=4)
    ollama_queue_timeout_seconds: float = Field(default=5.0, gt=0, le=120)
    ollama_direct_summary_deadline_seconds: float = Field(
        default=8.5, gt=0, le=30
    )
    ollama_keep_alive: str = "30m"
    ollama_prewarm_on_startup: bool = False

    max_upload_size_bytes: int = 10 * 1024 * 1024
    combined_image_max_height: int = 1600
    combined_image_gap_pixels: int = 16

    analysis_device: str = "auto"
    foot_confidence: float = 0.25
    foot_min_component_area: int = 2000
    foot_iou_duplicate_threshold: float = 0.6
    # Legacy SAM setting; the fungal Student always uses its checkpoint threshold.
    fungal_threshold: float | None = None
    # Temporary test switch; the fungal class continues to use the same model.
    tinea_inflammation_enabled: bool = True
    inflammation_threshold: float | None = Field(default=0.90, ge=0, le=1)
    wound_segmentation_enabled: bool = True
    wound_segmentation_checkpoint: Path = PROJECT_ROOT / "weights" / "wound_segformer_experiment_c_best.pt"
    # Input size and final threshold are read from the wound checkpoint.
    wound_redness_reduction: float = Field(default=0.2, ge=0, le=1)
    # Legacy SAM inference controls; ignored by the binary fungal Student.
    tinea_sliding_window_enabled: bool = False
    tinea_sliding_window_tile_size: int = 768
    tinea_sliding_window_overlap: float = 0.3
    tinea_sliding_window_padding: int = 32
    tinea_sliding_window_max_tiles: int = 12
    tinea_preprocess_enhance_enabled: bool = False
    tinea_preprocess_contrast_gain: float = 1.05
    tinea_preprocess_clahe_clip_limit: float = 1.6
    tinea_preprocess_red_saturation_gain: float = 1.10
    tinea_preprocess_red_value_gain: float = 1.02
    score_hard_weight: float = 0.7
    score_sensitivity: float = 7.0

    suspicion_line_thickness: int = 3
    min_suspicion_area: int = 18
    circle_scale: float = 1.08
    min_circle_radius: int = 5
    max_circle_radius: int = 42
    max_dot_area_for_suspicion_map: int = 450
    inner_confidence_percentile: float = 78.0
    inner_min_area_ratio: float = 0.18
    photo_cutout_background: bool = True
    photo_cutout_padding: int = 28

    hallux_model_source_dir: Path = Field(default=PROJECT_ROOT / "vendor" / "hallux")
    hallux_landmark_weights: Path = Field(
        default=PROJECT_ROOT / "weights" / "hallux_landmark_optional.pt"
    )
    hallux_img_size: int = 384
    hallux_seg_conf: float = 0.25
    hallux_crop_padding: int = 28
    hallux_visual_lower_segment_ratio: float = 0.85
    hallux_visual_lower_segment_foot_ratio: float = 0.24

    # The ID 1/4 geometry is implemented locally from the pinned ChArUco
    # contract. ArUco-marker-code supplies only detector/lens runtime modules;
    # the camera-specific NPZ is selected independently.
    aruco_source_dir: Path = Field(default=PROJECT_ROOT / "vendor" / "aruco")
    aruco_camera_calibration: Path | None = PROJECT_ROOT / "weights" / "camera_calibration.npz"
    aruco_foot_segmentation_weights: Path | None = None
    aruco_dictionary: str = "DICT_4X4_50"
    aruco_expected_image_width: int = 1280
    aruco_expected_image_height: int = 720
    aruco_allow_calibration_resize: bool = False
    aruco_undistort_balance: float = 0.0
    aruco_max_camera_rms_px: float = Field(default=1.0, gt=0.0)
    # ID 1/4 base profile from ChArUco commit 36b4db1, with the active heel correction above.
    aruco_marker_size_mm: float = Field(
        default=CHARUCO_FOOT_MARKER_SIZE_MM,
        gt=0.0,
    )
    aruco_marker_edge_gap_mm: float = Field(
        default=CHARUCO_FOOT_MARKER_EDGE_GAP_MM,
        gt=0.0,
    )
    # Retained for legacy configuration; per-foot fields control measurements.
    aruco_heel_offset_mm: float = Field(
        default=CHARUCO_FOOT_HEEL_OFFSET_MM,
        gt=0.0,
    )
    aruco_left_heel_offset_mm: float = Field(default=114.0, gt=0.0)
    aruco_right_heel_offset_mm: float = Field(default=117.0, gt=0.0)
    aruco_board_width_mm: float = Field(
        default=CHARUCO_FOOT_BOARD_WIDTH_MM,
        gt=0.0,
    )
    aruco_marker_center_x_mm: float = Field(
        default=CHARUCO_FOOT_MARKER_CENTER_X_MM,
        gt=0.0,
    )
    aruco_pixels_per_mm: float = Field(
        default=CHARUCO_FOOT_PIXELS_PER_MM,
        gt=0.0,
    )
    aruco_minimum_marker_side_px: float = Field(default=30.0, gt=0.0)
    aruco_maximum_fit_rmse_mm: float = Field(default=2.0, gt=0.0)
    aruco_maximum_loocv_mm: float = Field(default=5.0, gt=0.0)
    aruco_minimum_mask_area_mm2: float = Field(default=2_000.0, gt=0.0)
    aruco_minimum_foot_length_mm: float = Field(default=180.0, gt=0.0)
    aruco_maximum_foot_length_mm: float = Field(default=320.0, gt=0.0)
    aruco_foot_length_offset_mm: float = Field(default=0.0, ge=0.0, le=20.0)
    aruco_yolo_confidence: float = 0.25
    aruco_yolo_mask_threshold: float = 0.5
    aruco_yolo_image_size: int = 1280
    aruco_yolo_device: str = ""
    aruco_forefoot_start_fraction: float = 0.32
    aruco_forefoot_end_fraction: float = 0.36
    aruco_crop_padding_mm: float = 8.0
    # ArUco's image_left/image_right names are board bays, not anatomy.  Set
    # which anatomical side is placed in the canonical image-left bay.
    aruco_image_left_anatomical_side: str = "left"

    @model_validator(mode="after")
    def resolve_portable_paths_and_endpoints(self):
        provided_fields = set(self.model_fields_set)
        # Relative environment paths always resolve from the repository root.
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, Path):
                setattr(self, name, value.expanduser() if value.is_absolute()
                        else (PROJECT_ROOT / value).resolve())
        checkpoint_files = {
            "wound_segmentation_checkpoint": "wound_segformer_experiment_c_best.pt",
            "plantar_contact_segformer_checkpoint": "plantar_contact_segformer_b0_real_finetune_brightness2_best.pt",
            "plantar_full_sam_checkpoint": "plantar_full_sam_v4_decoder.pt",
            "plantar_sam_checkpoint": "sam_vit_b_01ec64.pth",
            "hallux_landmark_weights": "hallux_landmark_optional.pt",
            "aruco_camera_calibration": "camera_calibration.npz",
        }
        for name, filename in checkpoint_files.items():
            if name not in provided_fields:
                setattr(self, name, self.weights_dir / filename)
        base_url = self.backend_base_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("BACKEND_BASE_URL must be an HTTP(S) URL")
        for name in type(self).model_fields:
            value = getattr(self, name)
            if name not in provided_fields and isinstance(value, str) and value.startswith("http://127.0.0.1:8080/"):
                setattr(self, name, base_url + value[len("http://127.0.0.1:8080"):])
        if self.openai_report_max_output_tokens < self.openai_report_min_output_tokens:
            raise ValueError("OpenAI maximum output tokens must be at least the minimum")
        return self

    @model_validator(mode="after")
    def validate_shoe_fit_policy(self):
        if self.shoe_risk_low_min_score <= self.shoe_risk_medium_min_score:
            raise ValueError("shoe risk LOW threshold must be greater than MEDIUM threshold")
        increasing_pairs = (
            (self.shoe_forefoot_width_ratio_neutral, self.shoe_forefoot_width_ratio_high),
            (self.shoe_hallux_angle_neutral_degree, self.shoe_hallux_angle_high_degree),
            (self.shoe_pressure_imbalance_neutral_percent, self.shoe_pressure_imbalance_high_percent),
            (self.shoe_rearfoot_pressure_neutral_percent, self.shoe_rearfoot_pressure_high_percent),
            (self.shoe_forefoot_pressure_neutral_percent, self.shoe_forefoot_pressure_high_percent),
            (self.shoe_humidity_neutral_percent, self.shoe_humidity_high_percent),
        )
        if any(high <= neutral for neutral, high in increasing_pairs):
            raise ValueError("shoe fit high thresholds must be greater than neutral thresholds")
        decreasing_pairs = (
            (self.shoe_balance_neutral_score, self.shoe_balance_high_risk_score),
            (self.shoe_skin_safety_neutral_score, self.shoe_skin_safety_high_risk_score),
            (self.shoe_fungal_safety_neutral_score, self.shoe_fungal_safety_high_risk_score),
        )
        if any(high_risk >= neutral for neutral, high_risk in decreasing_pairs):
            raise ValueError("shoe safety high-risk thresholds must be below neutral thresholds")
        weight_groups = (
            (
                self.shoe_forefoot_area_weight,
                self.shoe_heel_area_weight,
                self.shoe_insole_area_weight,
            ),
            (
                self.shoe_forefoot_width_component_weight,
                self.shoe_forefoot_toebox_component_weight,
            ),
            (
                self.shoe_heel_hold_component_weight,
                self.shoe_heel_shock_component_weight,
                self.shoe_heel_energy_component_weight,
                self.shoe_heel_cushion_component_weight,
            ),
            (
                self.shoe_insole_breathability_component_weight,
                self.shoe_insole_cushion_component_weight,
                self.shoe_insole_shock_component_weight,
            ),
        )
        if any(sum(group) <= 0 for group in weight_groups):
            raise ValueError("each shoe fit weight group must have a positive sum")
        return self


settings = Settings()
