"""Opt-in, bounded local copies of original marker-failure captures.

This diagnostic never sends network requests or changes measurement state.
Only its own regular files are removed during retention; links, reparse points,
unknown files and unrelated directories are never followed or deleted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
from uuid import uuid4

from app.services.photo_failure_guidance import marker_occlusion_explanation


_DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / "storage/runtime/marker-capture-diagnostics"
_ARTIFACT_TYPE = "feetfit-marker-capture-diagnostic-v1"
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_CAPTURES = 12
_LOCK = threading.Lock()
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}\Z")
_DIRECTORY_NAME = re.compile(r"capture_\d{20}_[0-9a-f]{8}_s[1-9]\d{0,18}_a[0-3]_[A-Za-z0-9_-]{1,32}\Z")
_IMAGE_FILES = frozenset({"dorsal_original.jpg", "dorsal_original.png"})
_CAPTURE_FILES = _IMAGE_FILES | {"metadata.json"}
_MARKER_CODES = frozenset({
    "DORSAL_MARKERS_INSUFFICIENT", "DORSAL_ARUCO_GEOMETRY_INVALID", "DORSAL_MARKER_OCCLUDED_BY_FOOT",
})
_ENUM_FIELDS = {
    "diagnosticCode": _MARKER_CODES,
    "markerOcclusionStatus": frozenset({"foot_occlusion", "no_foot_overlap", "unknown", "markers_visible"}),
    "markerOcclusionMethod": frozenset({"calibrated_single_marker_projection_yolo_overlap", "visible_partial_marker_corners_yolo_overlap"}),
    "measurementStatus": frozenset({"FAILED", "WAITING_FOR_RECAPTURE", "WAITING_FOR_ENVIRONMENT"}),
    "failureReason": frozenset({"INVALID_CAPTURE_DATA", "CAMERA_ERROR", "AI_SERVER_ERROR"}),
    "stage": frozenset({"capture-preflight", "dorsal-measurement-preflight", "capture-status-transition"}),
    "imageField": frozenset({"dorsalFootImage"}),
}
_INT_FIELDS = {
    "actualWidth": (1, 20_000), "actualHeight": (1, 20_000),
    "expectedWidth": (1, 20_000), "expectedHeight": (1, 20_000),
    "detectedMarkerCount": (0, 2), "requiredMarkerCount": (0, 2),
    "detectedFootCount": (0, 2), "requiredFootCount": (0, 2),
    "markerVisibleCornerCount": (0, 4), "markerCoveredCornerCount": (0, 4),
    "markerPartialCandidateCount": (0, 1_000_000), "markerPartialEdgeCount": (0, 1_000_000),
    "markerPartialCoveredCandidateCount": (0, 1_000_000), "markerPartialPatternCandidateCount": (0, 1_000_000),
    "photoRecaptureCount": (0, 3), "maxPhotoRecaptures": (0, 3),
    "statusCode": (100, 599), "backendStatusCode": (100, 599),
}
_BOOL_FIELDS = frozenset({
    "markerOcclusionDetected", "markerOcclusionFootDetected", "statusCallbackConfirmed",
    "sameSessionRecaptureAllowed", "recaptureRequired", "mayStartPressure", "shouldDisconnect",
    "detectedMarkerId1", "detectedMarkerId4", "missingMarkerId1", "missingMarkerId4",
})
_RATIO_FIELDS = frozenset({"markerOcclusionOverlapRatio", "markerPatternScore"})


def _is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _safe_directory(path: Path, *, parent: Path | None = None) -> bool:
    try:
        if _is_link(path) or not stat.S_ISDIR(path.lstat().st_mode):
            return False
        resolved = path.resolve(strict=True)
        return resolved == path and (parent is None or resolved.parent == parent)
    except OSError:
        return False


def _prepare_base(directory: Path) -> Path | None:
    absolute = directory.absolute()
    # Check existing ancestors before mkdir/resolve, including Windows junctions.
    for ancestor in (*reversed(absolute.parents), absolute):
        try:
            if _is_link(ancestor) or not stat.S_ISDIR(ancestor.lstat().st_mode):
                return None
        except FileNotFoundError:
            continue
    absolute.mkdir(parents=True, exist_ok=True)
    resolved = absolute.resolve(strict=True)
    return resolved if _safe_directory(resolved) else None


def _safe_regular_file(path: Path, parent: Path) -> bool:
    try:
        return (not _is_link(path) and stat.S_ISREG(path.lstat().st_mode)
                and path.resolve(strict=True).parent == parent)
    except OSError:
        return False


def _safe_details(details: dict) -> dict:
    output = {}
    for key, allowed in _ENUM_FIELDS.items():
        value = details.get(key)
        if isinstance(value, str) and value in allowed:
            output[key] = value
    for key in ("markerOcclusionReason", "markerProjectionReason", "markerPartialCheckReason"):
        reason = details.get(key)
        if isinstance(reason, str) and marker_occlusion_explanation(reason) is not None:
            output[key] = reason
    for key, (minimum, maximum) in _INT_FIELDS.items():
        value = details.get(key)
        if type(value) is int and minimum <= value <= maximum:
            output[key] = value
    for key in _BOOL_FIELDS:
        value = details.get(key)
        if type(value) is bool:
            output[key] = value
    for key in _RATIO_FIELDS:
        value = details.get(key)
        if type(value) in (float, int) and 0 <= value <= 1 and math.isfinite(value):
            output[key] = float(value)
    return output


def _owned_capture(directory: Path, base: Path) -> bool:
    if not _DIRECTORY_NAME.fullmatch(directory.name) or not _safe_directory(directory, parent=base):
        return False
    metadata = directory / "metadata.json"
    if not _safe_regular_file(metadata, directory) or metadata.stat().st_size > 16_384:
        return False
    try:
        data = json.loads(metadata.read_text(encoding="utf-8"))
        return (isinstance(data, dict) and data.get("artifactType") == _ARTIFACT_TYPE
                and data.get("requestDirectory") == directory.name
                and data.get("imageFile") in _IMAGE_FILES)
    except (OSError, ValueError, TypeError):
        return False


def _remove_capture(directory: Path, base: Path) -> bool:
    """Remove only known regular files, then the empty direct child directory."""
    try:
        if (not _safe_directory(base) or not _safe_directory(directory, parent=base)
                or not _DIRECTORY_NAME.fullmatch(directory.name)):
            return False
        entries = tuple(directory.iterdir())
        if any(entry.name not in _CAPTURE_FILES or not _safe_regular_file(entry, directory) for entry in entries):
            return False
        # All resolved targets have been checked before the first deletion.
        for entry in entries:
            if not _safe_regular_file(entry, directory):
                return False
            entry.unlink()
        directory.rmdir()
        return True
    except OSError:
        return False


def _prune(base: Path, current: Path) -> bool:
    owned = sorted((entry for entry in base.iterdir() if _owned_capture(entry, base)), key=lambda entry: entry.name)
    excess = len(owned) - _MAX_CAPTURES
    for directory in owned:
        if excess <= 0:
            break
        if directory != current and _remove_capture(directory, base):
            excess -= 1
    return excess <= 0


def _write_exclusive(path: Path, contents: bytes) -> None:
    with path.open("xb") as output:
        output.write(contents)


def save_marker_capture_diagnostic(
    dorsal_bytes: bytes,
    *,
    measurement_session_id: int,
    photo_capture_attempt: int,
    request_id: str,
    diagnostic_details: dict,
    enabled: bool = False,
    _directory: Path | None = None,
) -> str | None:
    """Save exact input bytes locally when explicitly enabled; never raise.

    ``_directory`` is a test seam. Production uses the fixed repository runtime
    directory. Optional debugging failures must not change the API outcome.
    """
    if enabled is not True:
        return None
    base = target = pointer_temporary = None
    created_target = False
    published = False
    try:
        if (type(measurement_session_id) is not int or not 0 < measurement_session_id < 2**63
                or type(photo_capture_attempt) is not int or not 0 <= photo_capture_attempt <= 3
                or not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id)
                or not isinstance(dorsal_bytes, bytes) or not 0 < len(dorsal_bytes) <= _MAX_IMAGE_BYTES
                or not isinstance(diagnostic_details, dict)):
            return None
        if dorsal_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            image_file = "dorsal_original.png"
        elif dorsal_bytes.startswith(b"\xff\xd8\xff"):
            image_file = "dorsal_original.jpg"
        else:
            return None
        details = _safe_details(diagnostic_details)
        if details.get("diagnosticCode") not in _MARKER_CODES:
            return None
        token = re.sub(r"[^A-Za-z0-9_-]", "_", request_id)[:32]
        name = f"capture_{time.time_ns():020d}_{uuid4().hex[:8]}_s{measurement_session_id}_a{photo_capture_attempt}_{token}"
        with _LOCK:
            base = _prepare_base(Path(_directory) if _directory is not None else _DEFAULT_DIRECTORY)
            if base is None:
                return None
            target = base / name
            target.mkdir(exist_ok=False)
            created_target = True
            if not _safe_directory(target, parent=base):
                return None
            metadata = {
                "artifactType": _ARTIFACT_TYPE,
                "requestDirectory": name,
                "measurementSessionId": measurement_session_id,
                "photoCaptureAttempt": photo_capture_attempt,
                "requestId": request_id,
                "createdAtUtc": datetime.now(timezone.utc).isoformat(),
                "imageFile": image_file,
                "imageSha256": sha256(dorsal_bytes).hexdigest(),
                "imageSizeBytes": len(dorsal_bytes),
                "diagnostics": details,
            }
            _write_exclusive(target / image_file, dorsal_bytes)
            _write_exclusive(target / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))
            pointer = base / "latest.json"
            if pointer.exists() or pointer.is_symlink():
                if not _safe_regular_file(pointer, base):
                    raise OSError("Unsafe diagnostic pointer")
            pointer_temporary = base / f".latest-{uuid4().hex}.tmp"
            pointer_data = {"artifactType": _ARTIFACT_TYPE, "requestDirectory": name, "imageFile": image_file,
                            "measurementSessionId": measurement_session_id, "photoCaptureAttempt": photo_capture_attempt}
            _write_exclusive(pointer_temporary, json.dumps(pointer_data, indent=2).encode("utf-8"))
            if not _prune(base, target):
                raise OSError("Diagnostic retention unavailable")
            os.replace(pointer_temporary, pointer)
            pointer_temporary = None
            published = True
            return str(target)
    except Exception:
        return None
    finally:
        if pointer_temporary is not None and base is not None and _safe_regular_file(pointer_temporary, base):
            try:
                pointer_temporary.unlink()
            except OSError:
                pass
        # A successfully published directory is retained. Failed, newly created
        # directories are cleaned without touching any pre-existing directory.
        if not published and created_target and target is not None and base is not None:
            _remove_capture(target, base)
