"""Short-lived, process-local handoff for dorsal tinea analysis.

The calibrated top-photo request and the plantar-photo request arrive at
different times.  This module keeps only the already-computed dorsal analysis
artifacts long enough for the plantar request to build the final four-image
report.  Authorization credentials are never retained: a process-random HMAC
fingerprint is used solely to prevent one bearer credential from consuming a
different credential's staged result.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from hashlib import sha256
import hmac
import secrets
import threading
import time

from app.core.config import settings
from app.services.tinea_analysis import TineaAnalysisResult


@dataclass(frozen=True)
class DorsalTineaHandoff:
    session_id: int
    left_result: TineaAnalysisResult
    right_result: TineaAnalysisResult
    combined_suspicion_map_png: bytes
    combined_photo_overlay_png: bytes
    combined_clean_original_png: bytes
    staged_at_monotonic: float
    generation_id: str = field(default_factory=lambda: secrets.token_hex(16))


_FINGERPRINT_KEY = secrets.token_bytes(32)
_lock = threading.RLock()
_entries: OrderedDict[tuple[int, bytes], DorsalTineaHandoff] = OrderedDict()


def _key(session_id: int, authorization: str) -> tuple[int, bytes]:
    normalized_session_id = int(session_id)
    if normalized_session_id < 1:
        raise ValueError("session_id must be positive.")
    if not isinstance(authorization, str) or not authorization.strip():
        raise ValueError("authorization must not be empty.")
    fingerprint = hmac.new(
        _FINGERPRINT_KEY,
        authorization.encode("utf-8"),
        sha256,
    ).digest()
    return normalized_session_id, fingerprint


def _purge_expired(now: float) -> None:
    ttl = float(settings.tinea_handoff_ttl_seconds)
    expired = [
        key
        for key, value in _entries.items()
        if now - value.staged_at_monotonic >= ttl
    ]
    for key in expired:
        _entries.pop(key, None)


def stage_dorsal_tinea(
    session_id: int,
    authorization: str,
    left_result: TineaAnalysisResult,
    right_result: TineaAnalysisResult,
    combined_map: bytes,
    combined_overlay: bytes,
    combined_clean_original: bytes,
) -> DorsalTineaHandoff:
    """Stage one dorsal result without retaining the bearer credential."""

    entry_key = _key(session_id, authorization)
    now = time.monotonic()
    handoff = DorsalTineaHandoff(
        session_id=int(session_id),
        left_result=left_result,
        right_result=right_result,
        combined_suspicion_map_png=bytes(combined_map),
        combined_photo_overlay_png=bytes(combined_overlay),
        combined_clean_original_png=bytes(combined_clean_original),
        staged_at_monotonic=now,
    )
    with _lock:
        _purge_expired(now)
        _entries.pop(entry_key, None)
        _entries[entry_key] = handoff
        maximum = int(settings.tinea_handoff_max_sessions)
        while len(_entries) > maximum:
            _entries.popitem(last=False)
    return handoff


def get_dorsal_tinea(
    session_id: int,
    authorization: str,
) -> DorsalTineaHandoff | None:
    """Return a matching unexpired handoff without consuming it."""

    entry_key = _key(session_id, authorization)
    now = time.monotonic()
    with _lock:
        _purge_expired(now)
        return _entries.get(entry_key)


def delete_dorsal_tinea(
    session_id: int,
    authorization: str,
    generation_id: str,
) -> bool:
    """Delete only the generation that the completed request consumed.

    A newer top-photo request may replace the same session while an older
    plantar request is in flight.  Comparing generations prevents that older
    request from deleting the newly staged result.
    """

    entry_key = _key(session_id, authorization)
    with _lock:
        current = _entries.get(entry_key)
        if current is None or current.generation_id != generation_id:
            return False
        _entries.pop(entry_key, None)
        return True


def _clear_for_tests() -> None:
    with _lock:
        _entries.clear()


__all__ = [
    "DorsalTineaHandoff",
    "delete_dorsal_tinea",
    "get_dorsal_tinea",
    "stage_dorsal_tinea",
]
