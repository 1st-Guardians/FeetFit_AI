"""Remember consumed pressure attempts without storing tokens or sensor values."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3


def claim_pressure_attempt(*, registry_path: Path, measurement_session_id: int,
                           pressure_measurement_attempt: int) -> bool:
    """Atomically consume a session/attempt pair exactly once.

    Receipts survive process restarts. An unconfirmed callback or failed job
    must not replay the same pressure frame or its state transitions.
    The backend status PATCH validates which attempt is current.
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(registry_path, timeout=5)) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS pressure_attempt_receipts "
            "(session_id TEXT NOT NULL, attempt INTEGER NOT NULL, PRIMARY KEY(session_id, attempt))"
        )
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            receipt = connection.execute(
                "INSERT OR IGNORE INTO pressure_attempt_receipts (session_id, attempt) VALUES (?, ?)",
                (str(measurement_session_id), pressure_measurement_attempt),
            )
            claimed = receipt.rowcount == 1
    return claimed
