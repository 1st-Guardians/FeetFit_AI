from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from app.services import plantar_full_sam_analysis as service
from app.services.tinea_analysis import TineaAnalysisResult


def _synthetic_sole() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = 190, 120
    mask = np.zeros((height, width), np.uint8)
    # Wider/lobed toe and forefoot end at the top.
    cv2.ellipse(mask, (60, 55), (42, 29), 0, 0, 360, 1, -1)
    for x, radius in ((27, 9), (45, 10), (63, 11), (82, 9), (98, 7)):
        cv2.circle(mask, (x, 19), radius, 1, -1)
        cv2.line(mask, (x, 27), (x, 39), 1, max(3, radius // 2))
    # Narrow lateral midfoot bridge and heel.
    cv2.rectangle(mask, (30, 60), (54, 139), 1, -1)
    cv2.ellipse(mask, (51, 150), (25, 31), 0, 0, 360, 1, -1)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    image = np.full((height, width, 3), 180, np.uint8)
    image[mask == 0] = 245
    # Asymmetric red marker proves canonicalization rotates but never mirrors.
    cv2.circle(image, (37, 91), 5, (255, 0, 0), -1)
    alpha = np.full((height, width), 255, np.uint8)
    return image, mask > 0, alpha


class PlantarOrientationTests(unittest.TestCase):
    def test_quarter_turns_all_finish_toes_up_without_mirroring(self) -> None:
        original_image, original_mask, original_alpha = _synthetic_sole()
        rotations = (
            lambda value: value,
            lambda value: cv2.rotate(value, cv2.ROTATE_90_CLOCKWISE),
            lambda value: cv2.rotate(value, cv2.ROTATE_180),
            lambda value: cv2.rotate(value, cv2.ROTATE_90_COUNTERCLOCKWISE),
        )
        marker_sides: list[int] = []
        for transform in rotations:
            result = service._canonicalize_sole(
                transform(original_image),
                transform(original_mask.astype(np.uint8)) > 0,
                transform(original_alpha),
            )
            x1, y1, x2, y2 = service._bbox(result.sole_mask)
            tight = result.sole_mask[y1:y2, x1:x2]
            self.assertGreater(tight.shape[0], tight.shape[1])
            self.assertGreaterEqual(service._toe_end_margin(tight), 0.0)
            self.assertFalse(result.mirrored)

            red = (
                (result.image_rgb[..., 0] > 235)
                & (result.image_rgb[..., 1] < 40)
                & result.sole_mask
            )
            self.assertTrue(red.any())
            marker_x = float(np.where(red)[1].mean())
            sole_x = float(np.where(result.sole_mask)[1].mean())
            marker_sides.append(-1 if marker_x < sole_x else 1)
        self.assertEqual(marker_sides, [-1, -1, -1, -1])

    def test_invalid_round_blob_fails_closed(self) -> None:
        image = np.full((80, 80, 3), 180, np.uint8)
        mask = np.zeros((80, 80), np.uint8)
        cv2.circle(mask, (40, 40), 27, 1, -1)
        with self.assertRaises(service.PlantarFullSamAnalysisError):
            service._canonicalize_sole(
                image,
                mask,
                np.full((80, 80), 255, np.uint8),
            )

    def test_low_toe_heel_margin_fails_closed(self) -> None:
        image, mask, alpha = _synthetic_sole()
        orientation = service._canonicalize_sole(image, mask, alpha)
        with (
            patch.object(service.settings, "plantar_min_orientation_confidence", 0.10),
            self.assertRaises(service.PlantarFullSamAnalysisError),
        ):
            service._require_unambiguous_orientation(
                replace(orientation, confidence=0.05)
            )

    def test_cardinal_crop_restores_capture_orientation_without_continuous_warp(self) -> None:
        image, mask, _ = _synthetic_sole()
        source_alpha = np.full(mask.shape, 255, np.uint8)
        roi = service.FootRoiResult(
            found=True,
            mask=mask.astype(np.uint8),
            bbox=(0, 0, 1, 1),
            crop_image=np.zeros((1, 1, 3), np.uint8),
            crop_mask=np.ones((1, 1), np.uint8),
            confidence=0.9,
            instance_count=1,
            cardinal_rotation_degrees_clockwise=90,
            cardinal_view_image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
            cardinal_view_mask=cv2.rotate(mask.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE),
        )
        with (
            patch.object(service.settings, "plantar_foot_roi_padding", 0),
            patch.object(service.settings, "plantar_foot_roi_padding_ratio", 0.0),
        ):
            orientation, crop_image, crop_mask, crop_alpha, crop_bbox = (
                service._canonical_cardinal_crop(roi, source_alpha)
            )
        self.assertEqual(orientation.image_rgb.shape, image.shape)
        np.testing.assert_array_equal(orientation.image_rgb, image)
        np.testing.assert_array_equal(orientation.sole_mask, mask)
        self.assertEqual(orientation.rotation_degrees, 0.0)
        self.assertEqual(orientation.operations, (
            "restore_detector_view_by_270_clockwise",
            "preserve_preprocessed_capture_orientation",
        ))
        self.assertFalse(any("rotate_bound" in item for item in orientation.operations))
        self.assertFalse(orientation.mirrored)
        self.assertGreater(service._toe_end_margin(crop_mask), 0.0)
        self.assertEqual(crop_image.shape[:2], crop_mask.shape)
        self.assertEqual(crop_alpha.shape, crop_mask.shape)
        self.assertEqual(crop_bbox, service._bbox(mask))


class PlantarTransparentRenderTests(unittest.TestCase):
    def test_antialiased_display_keeps_quantitative_contact_mask_binary(self) -> None:
        image = np.full((20, 10, 3), (120, 100, 80), dtype=np.uint8)
        sole = np.ones((20, 10), dtype=bool)
        contact = np.zeros_like(sole)
        contact[:, :5] = True
        alpha = np.full(sole.shape, 255, dtype=np.uint8)
        with (
            patch.object(service.settings, "plantar_canvas_width", 100),
            patch.object(service.settings, "plantar_canvas_height", 200),
            patch.object(service.settings, "plantar_canvas_margin_ratio", 0.0),
        ):
            rgba, _, contact_canvas, _ = service._fit_to_canvas(
                image,
                sole,
                contact,
                alpha,
            )

        expected = cv2.resize(
            contact.astype(np.uint8),
            (100, 200),
            interpolation=cv2.INTER_NEAREST,
        ) * 255
        np.testing.assert_array_equal(contact_canvas, expected)
        self.assertEqual(set(np.unique(contact_canvas)), {0, 255})

        row = rgba[100, :, :3]
        full_fill = np.asarray(
            (1.0 - service.CONTACT_OVERLAY_ALPHA) * image[0, 0]
            + service.CONTACT_OVERLAY_ALPHA
            * np.asarray(service.CONTACT_OVERLAY_COLOR_RGB),
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(row[20], full_fill)
        np.testing.assert_array_equal(row[80], image[0, 0])
        # The accepted region is a uniform fill; only its narrow geometric
        # outline may be anti-aliased.
        np.testing.assert_array_equal(row[20], row[35])
        np.testing.assert_array_equal(row[80], row[65])

    def test_contact_overlay_is_translucent_pastel_sky_blue(self) -> None:
        image = np.full((100, 50, 3), (120, 100, 80), dtype=np.uint8)
        sole = np.ones((100, 50), dtype=bool)
        contact = np.zeros_like(sole)
        contact[20:80, 10:40] = True
        alpha = np.full(sole.shape, 255, dtype=np.uint8)
        with (
            patch.object(service.settings, "plantar_canvas_width", 50),
            patch.object(service.settings, "plantar_canvas_height", 100),
            patch.object(service.settings, "plantar_canvas_margin_ratio", 0.0),
        ):
            rgba, _, contact_canvas, _ = service._fit_to_canvas(
                image,
                sole,
                contact,
                alpha,
            )

        expected_fill = np.asarray(
            (1.0 - service.CONTACT_OVERLAY_ALPHA) * image[50, 25]
            + service.CONTACT_OVERLAY_ALPHA
            * np.asarray(service.CONTACT_OVERLAY_COLOR_RGB),
            dtype=np.uint8,
        )
        # A point well inside the mask receives only the translucent fill, so
        # its exact blend proves that the real photographed pixel remains in
        # the published visualization.
        np.testing.assert_array_equal(rgba[50, 25, :3], expected_fill)
        np.testing.assert_array_equal(rgba[10, 25, :3], image[10, 25])
        self.assertEqual(int(rgba[50, 25, 3]), 255)
        self.assertGreater(int(rgba[50, 25, 2]), int(image[50, 25, 2]))
        self.assertTrue(contact_canvas[50, 25])
        boundary = rgba[20, 25, :3]
        self.assertFalse(
            np.array_equal(
                boundary,
                np.asarray(service.CONTACT_OUTLINE_COLOR_RGB, dtype=np.uint8),
            )
        )
        self.assertFalse(np.array_equal(boundary, image[20, 25]))

    def test_probability_does_not_change_binary_region_opacity(self) -> None:
        image = np.full((90, 90, 3), (120, 100, 80), dtype=np.uint8)
        sole = np.ones((90, 90), dtype=bool)
        contact = np.zeros_like(sole)
        contact[10:40, 10:40] = True
        contact[50:80, 50:80] = True
        probability = np.zeros_like(sole, dtype=np.float32)
        probability[10:40, 10:40] = 0.80
        probability[50:80, 50:80] = 0.99
        tinea = _authoritative_tinea_result(contact, probability)

        coverage = service._fungal_contact_display_coverage(
            tinea,
            sole,
            contact,
        )
        rendered = service._render_contact_overlay(
            image,
            sole,
            contact,
            contact_coverage=coverage,
        )

        low_delta = float(
            np.linalg.norm(rendered[25, 25].astype(float) - image[25, 25])
        )
        high_delta = float(
            np.linalg.norm(rendered[65, 65].astype(float) - image[65, 65])
        )
        self.assertAlmostEqual(high_delta, low_delta)
        self.assertGreater(low_delta, 0.0)
        np.testing.assert_array_equal(rendered[45, 45], image[45, 45])
        np.testing.assert_array_equal(rendered[9, 25], image[9, 25])
        self.assertFalse(np.any(coverage[~contact]))

        self.assertTrue(np.all(coverage[contact] == 1.0))
        np.testing.assert_array_equal(rendered[49, 65], image[49, 65])

    def test_supplied_low_coverage_cannot_fade_an_accepted_region(self) -> None:
        image = np.full((96, 96, 3), (120, 100, 80), dtype=np.uint8)
        sole = np.ones((96, 96), dtype=bool)
        contact = np.zeros_like(sole)
        contact[16:80, 16:80] = True
        contact_before = contact.copy()
        coverage = np.zeros_like(sole, dtype=np.float32)
        coverage[contact] = 0.01

        rendered = service._render_contact_overlay(
            image,
            sole,
            contact,
            contact_coverage=coverage,
        )

        center_delta = float(
            np.linalg.norm(rendered[48, 48].astype(float) - image[48, 48])
        )
        boundary_delta = float(
            np.linalg.norm(rendered[16, 48].astype(float) - image[16, 48])
        )
        self.assertGreater(center_delta, 0.0)
        self.assertGreater(boundary_delta, center_delta)
        np.testing.assert_array_equal(rendered[15, 48], image[15, 48])
        np.testing.assert_array_equal(contact, contact_before)

    def test_inner_outline_does_not_leak_into_internal_holes_or_empty_masks(self) -> None:
        image = np.full((96, 96, 3), (120, 100, 80), dtype=np.uint8)
        sole = np.ones((96, 96), dtype=bool)
        contact = np.zeros_like(sole)
        contact[12:84, 12:84] = True
        contact[36:60, 36:60] = False
        coverage = contact.astype(np.float32) * 0.16

        rendered = service._render_contact_overlay(
            image,
            sole,
            contact,
            contact_coverage=coverage,
        )
        self.assertFalse(np.array_equal(rendered[35, 48], image[35, 48]))
        np.testing.assert_array_equal(rendered[36, 48], image[36, 48])
        np.testing.assert_array_equal(rendered[48, 48], image[48, 48])

        empty = np.zeros_like(contact)
        empty_rendered = service._render_contact_overlay(
            image,
            sole,
            empty,
            contact_coverage=empty.astype(np.float32),
        )
        np.testing.assert_array_equal(empty_rendered, image)

    def test_canvas_is_rgba_and_hidden_rgb_is_zero(self) -> None:
        image, sole, alpha = _synthetic_sole()
        contact = np.zeros_like(sole)
        contact[40:75, 25:90] = sole[40:75, 25:90]
        with (
            patch.object(service.settings, "plantar_canvas_width", 128),
            patch.object(service.settings, "plantar_canvas_height", 256),
            patch.object(service.settings, "plantar_canvas_margin_ratio", 0.08),
        ):
            rgba, sole_canvas, contact_canvas, tinea_input = service._fit_to_canvas(
                image,
                sole,
                contact,
                alpha,
            )
        self.assertEqual(rgba.shape, (256, 128, 4))
        self.assertEqual(sole_canvas.shape, (256, 128))
        self.assertEqual(contact_canvas.shape, (256, 128))
        self.assertEqual(tinea_input.shape, (256, 128, 3))
        transparent = rgba[..., 3] == 0
        self.assertTrue(transparent.any())
        self.assertFalse(np.any(rgba[..., :3][transparent]))
        self.assertTrue(np.any(rgba[..., 3] == 255))
        self.assertTrue(np.any((rgba[..., 3] > 0) & (rgba[..., 3] < 255)))
        self.assertFalse(np.any((rgba[..., 3] > 0) & (sole_canvas == 0)))
        self.assertTrue(np.any(contact_canvas > 0))
        self.assertFalse(np.any((contact_canvas > 0) & (sole_canvas == 0)))
        self.assertTrue(np.all(tinea_input[sole_canvas == 0] == 255))
        self.assertTrue(
            np.any(
                tinea_input[contact_canvas > 0]
                != rgba[..., :3][contact_canvas > 0]
            )
        )

    def test_transparent_pair_combination_keeps_gap_transparent(self) -> None:
        left = np.zeros((30, 12, 4), np.uint8)
        right = np.zeros((30, 12, 4), np.uint8)
        left[5:25, 3:9] = (220, 30, 20, 255)
        right[5:25, 3:9] = (20, 30, 220, 255)
        with (
            patch.object(
                service.Image,
                "open",
                side_effect=[
                    Image.fromarray(left, mode="RGBA"),
                    Image.fromarray(right, mode="RGBA"),
                ],
            ),
            patch.object(service.settings, "combined_image_max_height", 30),
            patch.object(service.settings, "combined_image_gap_pixels", 6),
        ):
            encoded = service.combine_transparent_visualizations(
                Path("left.png"),
                Path("right.png"),
            )
        decoded = cv2.imdecode(
            np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED
        )
        self.assertEqual(decoded.shape, (30, 30, 4))
        gap = decoded[:, 12:18]
        self.assertFalse(np.any(gap[..., 3]))
        self.assertFalse(np.any(gap[..., :3]))


def _authoritative_tinea_result(
    mask: np.ndarray,
    probability: np.ndarray | None = None,
) -> TineaAnalysisResult:
    if probability is None:
        probability = np.where(mask, 0.99, 0.0).astype(np.float32)
    return TineaAnalysisResult(
        suspicion_map_png=b"map",
        photo_overlay_png=b"overlay",
        original_filename="sole.png",
        fungal_safety_score=70,
        skin_reaction_safety_score=100,
        metrics={
            "fungal_threshold": 0.78,
            "fungal_signal_detected": bool(np.any(mask)),
        },
        clean_original_png=b"clean",
        authoritative_fungal_mask=np.ascontiguousarray(mask, dtype=bool),
        authoritative_fungal_probability=np.ascontiguousarray(
            probability,
            dtype=np.float32,
        ),
    )


def _contact_prediction(
    mask: np.ndarray,
    probability: np.ndarray | None = None,
) -> service._PressedContactPrediction:
    if probability is None:
        probability = np.where(mask, 0.99, 0.0).astype(np.float32)
    return service._PressedContactPrediction(
        mask=np.ascontiguousarray(mask, dtype=bool),
        probability=np.ascontiguousarray(probability, dtype=np.float32),
        audit={
            "source": "unit-test-contact-model",
            "modelInferenceExecuted": True,
            "postprocessing": {},
        },
    )


def _prepared_canvas() -> service._PreparedPlantarCanvas:
    height, width = 16, 10
    sole = np.zeros((height, width), dtype=np.uint8)
    sole[2:14, 2:8] = 255
    tinea_input = np.full((height, width, 3), 255, dtype=np.uint8)
    tinea_input[sole > 0] = (90, 120, 150)
    clean_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    clean_rgba[sole > 0, :3] = tinea_input[sole > 0]
    clean_rgba[sole > 0, 3] = 255
    return service._PreparedPlantarCanvas(
        side="left",
        clean_rgba=clean_rgba,
        contact_display_rgba=clean_rgba.copy(),
        tinea_input_rgb=tinea_input,
        sole_canvas=sole,
        contact_model_input_rgb=np.ascontiguousarray(tinea_input[2:14, 2:8]),
        contact_model_sole=np.ascontiguousarray(sole[2:14, 2:8]),
        contact_source_shape=(height, width),
        contact_source_bbox=(2, 2, 8, 14),
        contact_display_rotation_degrees_clockwise=0,
        contact_display_crop_bbox=(2, 2, 8, 14),
        audit_base={
            "source": {"format": "PNG", "byte_count": 3, "sha256": "source"},
            "foot_roi": {"confidence": 0.91},
            "orientation": {"confidence": 0.88, "mirrored": False},
            "display_crop_bbox_xyxy_in_model_crop": [0, 0, width, height],
        },
        yolo_confidence=0.91,
        orientation_confidence=0.88,
    )


class PlantarSegFormerContactProductionTests(unittest.TestCase):
    @staticmethod
    def _analyzer() -> service.LocalPlantarFullSamAnalyzer:
        analyzer = service.LocalPlantarFullSamAnalyzer.__new__(
            service.LocalPlantarFullSamAnalyzer
        )
        analyzer._inference_lock = threading.RLock()
        analyzer.detector = object()
        analyzer.checkpoint_metadata = {
            "contact_model": "segformer_b0_real_finetune",
            "plantar_decoder_loaded": False,
            "plantar_sam_loaded": False,
        }
        return analyzer

    def test_constructor_loads_yolo_and_keeps_contact_model_lazy(self) -> None:
        detector = object()
        contact_checkpoint = service.weights.plantar_contact_segformer

        class DetectorOnlyWeights:
            plantar_foot_segmentation = service.weights.plantar_foot_segmentation
            plantar_contact_segformer = contact_checkpoint

            @property
            def tinea_pedis_segmentation(self):
                raise AssertionError("the tinea checkpoint must not be read")

            @property
            def plantar_full_sam(self):
                raise AssertionError("legacy decoder checkpoint must not be read")

            @property
            def plantar_sam(self):
                raise AssertionError("legacy SAM checkpoint must not be read")

        with (
            patch.object(
                service,
                "get_plantar_foot_roi_detector",
                return_value=detector,
            ) as get_detector,
            patch.object(service, "weights", DetectorOnlyWeights()),
            patch.object(service, "_sha256_file", return_value="yolo-sha"),
            patch.object(
                service.LocalPlantarFullSamAnalyzer,
                "_load_models",
                side_effect=AssertionError("legacy model loader must not run"),
            ) as legacy_loader,
            patch.object(
                service.LocalPlantarFullSamAnalyzer,
                "_segment_contact",
                side_effect=AssertionError("legacy contact path must not run"),
            ) as segment_contact,
            patch.object(
                service.LocalPlantarFullSamAnalyzer,
                "_refine_sole_alpha",
                side_effect=AssertionError("legacy sole SAM path must not run"),
            ) as refine_sole,
        ):
            with patch.object(service.Path, "is_file", return_value=True), patch.object(service, "_sha256_file", return_value="test-only-checkpoint-hash"):
                analyzer = service.LocalPlantarFullSamAnalyzer()

        get_detector.assert_called_once_with()
        legacy_loader.assert_not_called()
        segment_contact.assert_not_called()
        refine_sole.assert_not_called()
        self.assertIs(analyzer.detector, detector)
        self.assertEqual(
            analyzer.checkpoint_metadata["contact_model"],
            "segformer_b0_real_finetune",
        )
        self.assertEqual(
            analyzer.checkpoint_metadata["contact_model_loading"],
            "lazy_local_singleton",
        )
        self.assertEqual(analyzer.checkpoint_metadata["contact_output_channels"], 1)
        self.assertEqual(analyzer.checkpoint_metadata["contact_threshold"], 0.5)
        self.assertFalse(analyzer.checkpoint_metadata["plantar_decoder_loaded"])
        self.assertFalse(analyzer.checkpoint_metadata["plantar_sam_loaded"])
        self.assertFalse(hasattr(analyzer, "segmenter"))
        self.assertFalse(hasattr(analyzer, "predictor"))

    def test_prepare_canvas_bypasses_legacy_contact_and_sole_sam(self) -> None:
        analyzer = self._analyzer()
        image, sole, alpha = _synthetic_sole()
        orientation = service.OrientationResult(
            image_rgb=image,
            sole_mask=sole,
            source_alpha=alpha,
            rotation_degrees=0.0,
            toe_end_margin=0.2,
            confidence=0.8,
            operations=("detector_rotate_0_clockwise",),
        )
        roi = SimpleNamespace(
            found=True,
            confidence=0.93,
            detection_pass="identity",
            tta_attempted_transforms=("identity",),
            instance_count=1,
            bbox=(0, 0, image.shape[1], image.shape[0]),
            mask=sole,
            crop_image=image,
            crop_mask=sole,
            original_shape=image.shape[:2],
            cardinal_rotation_degrees_clockwise=0,
            selection_audit={"selected": "identity"},
        )
        with (
            patch.object(
                service,
                "_decode_image",
                return_value=(image, alpha, {"format": "PNG"}),
            ),
            patch.object(
                service, "prepare_plantar_foot_input",
                return_value=service.PreparedPlantarFootInput(image_rgb=image, roi=roi),
            ),
            patch.object(
                service,
                "_canonical_cardinal_crop",
                return_value=(
                    orientation,
                    image,
                    sole,
                    alpha,
                    (0, 0, image.shape[1], image.shape[0]),
                ),
            ),
            patch.object(
                analyzer,
                "_segment_contact",
                side_effect=AssertionError("legacy contact path must not run"),
            ) as segment_contact,
            patch.object(
                analyzer,
                "_refine_sole_alpha",
                side_effect=AssertionError("legacy sole SAM path must not run"),
            ) as refine_sole,
            patch.object(service.settings, "plantar_canvas_width", 128),
            patch.object(service.settings, "plantar_canvas_height", 256),
            patch.object(service.settings, "plantar_canvas_margin_ratio", 0.06),
        ):
            prepared = analyzer._prepare_canvas(b"png", "left")

        segment_contact.assert_not_called()
        refine_sole.assert_not_called()
        self.assertEqual(prepared.sole_canvas.shape, (256, 128))
        self.assertTrue(np.any(prepared.sole_canvas > 0))
        self.assertTrue(
            np.all(prepared.tinea_input_rgb[prepared.sole_canvas == 0] == 255)
        )
        self.assertFalse(
            np.any(
                prepared.clean_rgba[..., :3][prepared.clean_rgba[..., 3] == 0]
            )
        )

    def test_segformer_prediction_uses_white_foot_crop_and_thresholds_at_half(
        self,
    ) -> None:
        prepared = _prepared_canvas()
        crop_probability = np.full((12, 6), 0.49, dtype=np.float32)
        crop_probability[3:7, 2:5] = 0.50

        class FakeSegmenter:
            device = service.torch.device("cpu")

            def __init__(self) -> None:
                self.received: np.ndarray | None = None
                self.calls = 0

            def predict(self, images, device):
                self.received = np.asarray(list(images)[0]).copy()
                value = crop_probability if self.calls == 0 else crop_probability[:, ::-1].copy()
                self.calls += 1
                return SimpleNamespace(
                    original_probabilities=(
                        service.torch.from_numpy(value),
                    )
                )

        segmenter = FakeSegmenter()
        with (
            patch.object(
                service,
                "get_plantar_contact_segmenter",
                return_value=(segmenter, {"architecture": "segformer_b0"}),
            ) as get_segmenter,
            patch.object(service.settings, "plantar_contact_crop_padding_ratio", 0.0),
            patch.object(service.settings, "plantar_contact_threshold", 0.5),
            patch.object(service.settings, "plantar_canvas_width", 10),
            patch.object(service.settings, "plantar_canvas_height", 16),
            patch.object(service.settings, "plantar_canvas_margin_ratio", 0.125),
        ):
            prediction = service.LocalPlantarFullSamAnalyzer._predict_pressed_contact(
                prepared
            )

        get_segmenter.assert_called_once_with()
        self.assertEqual(segmenter.calls, 2)
        self.assertIsNotNone(segmenter.received)
        self.assertEqual(segmenter.received.shape, (12, 6, 3))
        self.assertTrue(np.all(segmenter.received == (90, 120, 150)))
        self.assertFalse(np.any(prediction.mask[:2]))
        self.assertTrue(np.all(prediction.mask[5:9, 4:7]))
        self.assertFalse(np.any(prediction.probability[prepared.sole_canvas == 0]))
        self.assertEqual(
            prediction.audit["postprocessing"]["policy"],
            "identity_hflip_binary_intersection_then_clip_to_yolo",
        )
        self.assertFalse(
            prediction.audit["postprocessing"]["morphologyApplied"]
        )

    def test_segformer_source_probability_rotates_only_for_saved_display(self) -> None:
        source_probability = np.zeros((4, 6), dtype=np.float32)
        source_probability[1, 2] = 0.9
        model_input = np.full((4, 6, 3), 120, dtype=np.uint8)
        source_sole = np.ones((4, 6), dtype=np.uint8) * 255
        display_sole = np.ones((6, 4), dtype=np.uint8) * 255
        prepared = service._PreparedPlantarCanvas(
            side="left",
            clean_rgba=np.full((6, 4, 4), 255, dtype=np.uint8),
            contact_display_rgba=np.dstack((
                np.full((6, 4, 3), 120, dtype=np.uint8),
                np.full((6, 4), 255, dtype=np.uint8),
            )),
            tinea_input_rgb=np.full((6, 4, 3), 120, dtype=np.uint8),
            sole_canvas=display_sole,
            contact_model_input_rgb=model_input,
            contact_model_sole=source_sole,
            contact_source_shape=(4, 6),
            contact_source_bbox=(0, 0, 6, 4),
            contact_display_rotation_degrees_clockwise=90,
            contact_display_crop_bbox=(0, 0, 4, 6),
            audit_base={
                "source": {},
                "foot_roi": {},
                "orientation": {},
                "display_crop_bbox_xyxy_in_model_crop": [0, 0, 4, 6],
            },
            yolo_confidence=0.9,
            orientation_confidence=0.9,
        )

        class FakeSegmenter:
            device = service.torch.device("cpu")
            calls = 0

            def predict(self, images, device):
                value = source_probability if self.calls == 0 else source_probability[:, ::-1].copy()
                self.calls += 1
                return SimpleNamespace(
                    original_probabilities=(
                        service.torch.from_numpy(value),
                    )
                )

        with (
            patch.object(
                service,
                "get_plantar_contact_segmenter",
                return_value=(FakeSegmenter(), {}),
            ),
            patch.object(service.settings, "plantar_contact_threshold", 0.5),
            patch.object(service.settings, "plantar_canvas_width", 4),
            patch.object(service.settings, "plantar_canvas_height", 6),
            patch.object(service.settings, "plantar_canvas_margin_ratio", 0.0),
        ):
            prediction = service.LocalPlantarFullSamAnalyzer._predict_pressed_contact(
                prepared
            )

        expected = service._rotate_quarter_turn_clockwise(source_probability, 90)
        np.testing.assert_allclose(prediction.probability, expected)
        np.testing.assert_array_equal(prediction.mask, expected >= 0.5)

    def test_hysteresis_keeps_only_low_threshold_regions_joined_to_a_seed(self) -> None:
        sole = np.ones((140, 120), dtype=bool)
        probability = np.zeros(sole.shape, dtype=np.float32)
        probability[15:50, 15:50] = 0.65
        probability[25:35, 25:35] = 0.77
        probability[75:110, 70:105] = 0.65

        recovered = service._hysteresis_contact_mask(probability, sole)

        self.assertTrue(recovered[30, 30])
        self.assertFalse(recovered[90, 85])
        self.assertEqual(service._component_count(recovered), 1)

    def test_glare_rejection_removes_a_whole_reflection_component(self) -> None:
        sole = np.ones((140, 120), dtype=bool)
        contact = np.zeros(sole.shape, dtype=bool)
        contact[15:50, 15:50] = True
        contact[75:110, 70:105] = True
        glare = np.zeros_like(contact)
        glare[15:50, 15:35] = True

        cleaned, rejected = service._reject_glare_contact_components(
            contact,
            glare,
            sole,
        )

        self.assertEqual(rejected, 1)
        self.assertFalse(cleaned[30, 30])
        self.assertTrue(cleaned[90, 85])
        self.assertEqual(service._component_count(cleaned), 1)

    def test_analyze_uses_contact_only_prediction_and_writes_binary_mask(self) -> None:
        analyzer = self._analyzer()
        prepared = _prepared_canvas()
        contact = np.zeros_like(prepared.sole_canvas, dtype=bool)
        contact[4:9, 3:7] = True
        prediction = _contact_prediction(contact)

        output_dir = Path("unit-test-fungal-footprint")
        written_pngs: dict[Path, np.ndarray] = {}

        def capture_png(path: Path, array: np.ndarray) -> None:
            written_pngs[Path(path)] = np.asarray(array).copy()

        with (
            patch.object(analyzer, "_prepare_canvas", return_value=prepared),
            patch.object(
                analyzer,
                "_predict_pressed_contact",
                return_value=prediction,
            ) as predict_contact,
            patch.object(service, "_write_png", side_effect=capture_png),
            patch.object(service, "_sha256_file", return_value="artifact-sha"),
            patch.object(Path, "mkdir", autospec=True),
            patch.object(Path, "write_text", autospec=True) as write_text,
        ):
            result = analyzer.analyze(b"source", "left", output_dir)

        predict_contact.assert_called_once_with(prepared)

        np.testing.assert_array_equal(
            written_pngs[result.contact_mask_path],
            contact.astype(np.uint8) * 255,
        )
        self.assertEqual(result.region_count, 1)
        self.assertGreater(result.contact_area_ratio, 0.0)
        self.assertIsNone(result.tinea_analysis_result)
        audit = json.loads(write_text.call_args.args[1])
        self.assertEqual(
            audit["pipeline_version"],
            "local-segformer-b0-contact-v2",
        )
        self.assertEqual(
            audit["segmentation"]["source"],
            "dedicated_segformer_b0_contact_model",
        )
        self.assertEqual(audit["segmentation"]["class_index"], 0)
        self.assertEqual(audit["segmentation"]["contact_threshold"], 0.5)
        self.assertEqual(
            audit["segmentation"]["display_confidence"]["source"],
            "binary_contact_mask_only",
        )
        self.assertFalse(
            audit["segmentation"]["display_confidence"][
                "modelConfidenceAffectsDisplayOpacity"
            ]
        )
        self.assertFalse(
            audit["sole_alpha_refinement"]["provenance"]["plantar_sam_used"]
        )

    def test_empty_pressed_contact_mask_is_a_valid_result(self) -> None:
        analyzer = self._analyzer()
        prepared = _prepared_canvas()
        empty = np.zeros_like(prepared.sole_canvas, dtype=bool)
        written_pngs: dict[Path, np.ndarray] = {}

        def capture_png(path: Path, array: np.ndarray) -> None:
            written_pngs[Path(path)] = np.asarray(array).copy()

        with (
            patch.object(service, "_write_png", side_effect=capture_png),
            patch.object(service, "_sha256_file", return_value="artifact-sha"),
            patch.object(Path, "mkdir", autospec=True),
            patch.object(Path, "write_text", autospec=True),
        ):
            result = analyzer._finalize_canvas(
                prepared,
                _contact_prediction(empty),
                Path("unit-test-empty-fungal-footprint"),
            )
        self.assertFalse(np.any(written_pngs[result.contact_mask_path]))
        self.assertEqual(result.region_count, 0)
        self.assertEqual(result.contact_area_ratio, 0.0)

    def test_authoritative_mask_outside_yolo_support_fails_closed(self) -> None:
        prepared = _prepared_canvas()
        outside = np.zeros_like(prepared.sole_canvas, dtype=bool)
        outside[0, 0] = True
        with self.assertRaisesRegex(
            service.PlantarFullSamAnalysisError,
            "outside the YOLO plantar support",
        ):
            service.LocalPlantarFullSamAnalyzer._fungal_contact_mask(
                _authoritative_tinea_result(outside),
                prepared.sole_canvas,
            )

    def test_probability_outside_authoritative_mask_fails_closed(self) -> None:
        prepared = _prepared_canvas()
        fungal = np.zeros_like(prepared.sole_canvas, dtype=bool)
        fungal[3:8, 3:7] = True
        probability = np.where(fungal, 0.9, 0.0).astype(np.float32)
        probability[2, 2] = 0.99
        with self.assertRaisesRegex(
            service.PlantarFullSamAnalysisError,
            "outside the authoritative mask",
        ):
            service._fungal_contact_display_coverage(
                _authoritative_tinea_result(fungal, probability),
                prepared.sole_canvas,
                fungal,
            )


class PreparedPlantarHandoffTests(unittest.TestCase):
    def test_cached_roi_preserves_original_pixels_and_rotates_source_alpha_once(self) -> None:
        analyzer = PlantarSegFormerContactProductionTests._analyzer()
        upright, sole, alpha = _synthetic_sole()
        alpha[20:30, 30:40] = 128
        original_rgb = cv2.rotate(upright, cv2.ROTATE_90_COUNTERCLOCKWISE)
        original_alpha = cv2.rotate(alpha, cv2.ROTATE_90_COUNTERCLOCKWISE)
        bbox = (0, 0, upright.shape[1], upright.shape[0])
        roi = service.FootRoiResult(
            found=True, mask=sole.astype(np.uint8), bbox=bbox,
            crop_image=upright, crop_mask=sole.astype(np.uint8),
            confidence=0.95, instance_count=1,
            cardinal_view_image=upright, cardinal_view_mask=sole.astype(np.uint8),
            cardinal_bbox=bbox, cardinal_crop_image=upright,
            cardinal_crop_mask=sole.astype(np.uint8),
        )
        original_bytes = b"original-upload"
        cached = service.PreparedPlantarFootInput(
            image_rgb=upright, roi=roi,
            rotation_degrees_clockwise=90, brightness_gain=1.0,
            audit={"source_sha256": service._sha256_bytes(original_bytes)},
        )
        with (
            patch.object(
                service, "_decode_image",
                return_value=(original_rgb, original_alpha, {"format": "PNG"}),
            ),
            patch.object(
                service, "extract_plantar_foot_roi",
                side_effect=AssertionError("accepted YOLO ROI must be reused"),
            ),
            patch.object(
                service, "prepare_plantar_foot_input",
                side_effect=AssertionError("accepted original pixels must be reused"),
            ),
            patch.object(
                service, "_canonical_cardinal_crop",
                wraps=service._canonical_cardinal_crop,
            ) as canonical,
            patch.object(service.settings, "plantar_canvas_width", 128),
            patch.object(service.settings, "plantar_canvas_height", 256),
        ):
            prepared = analyzer._prepare_canvas(
                original_bytes, "right", prepared_input=cached,
            )
        self.assertIs(canonical.call_args.args[0], roi)
        np.testing.assert_array_equal(canonical.call_args.args[1], alpha)
        np.testing.assert_array_equal(prepared.contact_model_input_rgb[sole], upright[sole])
        np.testing.assert_array_equal(prepared.contact_display_rgba, prepared.clean_rgba)
        self.assertEqual(prepared.audit_base["contact_preprocessing"]["brightness_gain"], 1.0)
        self.assertEqual(prepared.audit_base["contact_preprocessing"]["contrast_factor"], 1.0)
        self.assertEqual(prepared.audit_base["contact_preprocessing"]["source"], "original_upload_rotated_crop")
        self.assertTrue(np.all(prepared.contact_model_input_rgb[~sole] == 255))
        self.assertEqual(prepared.contact_source_shape, upright.shape[:2])
        self.assertEqual(prepared.audit_base["source"]["sha256"], service._sha256_bytes(original_bytes))

    def test_cached_roi_for_another_upload_is_rejected(self) -> None:
        analyzer = PlantarSegFormerContactProductionTests._analyzer()
        image, _, alpha = _synthetic_sole()
        cached = service.PreparedPlantarFootInput(
            image_rgb=image, roi=Mock(),
            audit={"source_sha256": service._sha256_bytes(b"another-upload")},
        )
        with (
            patch.object(service, "_decode_image", return_value=(image, alpha, {})),
            patch.object(service, "extract_plantar_foot_roi") as extract,
            self.assertRaisesRegex(service.PlantarFullSamAnalysisError, "original upload"),
        ):
            analyzer._prepare_canvas(b"actual-upload", "left", prepared_input=cached)
        extract.assert_not_called()


class PlantarLocalPolicyTests(unittest.TestCase):
    def test_sole_alpha_refinement_restores_every_contact_pixel(self) -> None:
        image, sole, _ = _synthetic_sole()
        contact = np.zeros_like(sole)
        contact[40:75, 25:90] = sole[40:75, 25:90]
        proposed = sole.copy()
        missing = np.argwhere(contact)[0]
        proposed[tuple(missing)] = False
        fake_result = SimpleNamespace(
            mask=proposed,
            selected_variant="box_contact",
            fallback_used=False,
            audit={"provenance": {"local_only": True, "gpt_used": False}},
        )
        analyzer = service.LocalPlantarFullSamAnalyzer.__new__(
            service.LocalPlantarFullSamAnalyzer
        )
        analyzer.predictor = object()
        with patch.object(
            service,
            "refine_sole_alpha_with_sam",
            return_value=fake_result,
        ):
            result, audit = analyzer._refine_sole_alpha(
                image,
                sole,
                contact,
            )
        self.assertTrue(np.all(result[contact]))
        self.assertEqual(audit["contact_pixels_restored_after_sam"], 1)
        self.assertTrue(audit["all_contact_pixels_preserved"])
        self.assertTrue(audit["contained_in_yolo"])

    def test_eligible_conservative_candidate_precedes_higher_color_rank(self) -> None:
        values = [
            {
                "eligible": True,
                "rank_score": 0.70,
                "sam_selected_score": 0.85,
                "candidate": {"method": "base", "base_retention": 1.0, "number": 1},
            },
            {
                "eligible": True,
                "rank_score": 0.99,
                "sam_selected_score": 0.99,
                "candidate": {
                    "method": "hsv_seeded_intersection",
                    "base_retention": 0.8,
                    "number": 2,
                },
            },
        ]
        self.assertEqual(service._select_evaluation(values), 0)

    def test_all_ineligible_candidates_fail_closed(self) -> None:
        values = [
            {
                "eligible": False,
                "eligibility_reasons": ["sam_score_below_minimum"],
                "rank_score": 0.99,
                "sam_selected_score": 0.99,
                "candidate": {"method": "base", "base_retention": 1.0, "number": 1},
            }
        ]
        with self.assertRaises(service.PlantarFullSamAnalysisError):
            service._select_evaluation(values)

    def test_near_tied_color_fallback_prefers_joint_s35(self) -> None:
        values = [
            {
                "eligible": True,
                "rank_score": 0.910,
                "sam_selected_score": 0.94,
                "candidate": {
                    "method": "lab_seeded_intersection",
                    "strength": 0.50,
                    "base_retention": 0.7,
                    "number": 6,
                },
            },
            {
                "eligible": True,
                "rank_score": 0.904,
                "sam_selected_score": 0.93,
                "candidate": {
                    "method": "lab_hsv_intersection",
                    "strength": 0.35,
                    "base_retention": 0.8,
                    "number": 5,
                },
            },
        ]
        self.assertEqual(service._select_evaluation(values), 1)

    @staticmethod
    def _dummy_refinement(area_ratio: float) -> SimpleNamespace:
        score = SimpleNamespace(
            candidate_iou=0.90,
            area_similarity=0.95,
            to_dict=lambda: {"candidate_iou": 0.90, "area_similarity": 0.95},
        )
        qc = SimpleNamespace(
            area_ratio=area_ratio,
            outside_support_ratio=0.0,
            positive_prompt_recall=1.0,
            component_count=1,
            largest_component_ratio=1.0,
            hole_area_ratio=0.0,
            passed=False,
            to_dict=lambda: {"area_ratio": area_ratio},
        )
        return SimpleNamespace(
            qc=qc,
            scores=(score,),
            selected_index=0,
            selected_score=0.95,
            mask=np.ones((8, 8), np.uint8),
        )

    def test_conservative_small_refinement_headroom_is_eligible(self) -> None:
        candidate = service.MaskCandidate(
            number=2,
            name="morphology_open",
            mask=np.ones((8, 8), np.uint8),
            method="morphology_open",
            strength=None,
            area_pixels=42,
            area_ratio=0.6524,
            base_retention=0.99,
        )
        row = service._candidate_evaluation(
            candidate,
            self._dummy_refinement(0.7015),
            minimum_sam_score=0.55,
            maximum_contact_ratio=0.70,
        )
        self.assertTrue(row["eligible"])
        self.assertTrue(row["conservative_hysteresis_applied"])

    def test_conservative_large_expansion_is_rejected(self) -> None:
        candidate = service.MaskCandidate(
            number=1,
            name="base",
            mask=np.ones((8, 8), np.uint8),
            method="base",
            strength=None,
            area_pixels=32,
            area_ratio=0.60,
            base_retention=1.0,
        )
        row = service._candidate_evaluation(
            candidate,
            self._dummy_refinement(0.74),
            minimum_sam_score=0.55,
            maximum_contact_ratio=0.70,
        )
        self.assertFalse(row["eligible"])
        self.assertIn(
            "conservative_refinement_expansion_above_maximum",
            row["eligibility_reasons"],
        )

    def test_service_has_no_openai_http_or_gpt_execution_path(self) -> None:
        source = inspect.getsource(service).lower()
        self.assertNotIn("import httpx", source)
        self.assertNotIn("import requests", source)
        self.assertNotIn("from openai", source)
        self.assertNotIn("api.openai.com", source)
        # Negative provenance flags are intentionally present in the audit.
        self.assertIn('"gpt_used": false', source)
        self.assertIn('"openai_used": false', source)


if __name__ == "__main__":
    unittest.main()
