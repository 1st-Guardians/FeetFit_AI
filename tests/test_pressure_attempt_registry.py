from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest
from uuid import uuid4

from app.services.pressure_attempt_registry import claim_pressure_attempt


class PressureAttemptRegistryTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(".build") / f"pressure-registry-test-{uuid4().hex}.sqlite3"
        self.addCleanup(self.path.unlink, missing_ok=True)

    def claim(self, session=9, attempt=1):
        return claim_pressure_attempt(registry_path=self.path, measurement_session_id=session,
                                      pressure_measurement_attempt=attempt)

    def test_exact_receipts_survive_connection_reopening_without_guessing_current_attempt(self):
        self.assertTrue(self.claim(attempt=1))
        self.assertFalse(self.claim(attempt=1))
        self.assertTrue(self.claim(attempt=99))
        self.assertTrue(self.claim(attempt=2))
        self.assertFalse(self.claim(attempt=2))
        self.assertTrue(self.claim(session=10, attempt=0))

    def test_concurrent_duplicate_receipts_accept_exactly_once(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(lambda _: self.claim(), range(8)))
        self.assertEqual(sum(outcomes), 1)
        self.assertFalse(self.claim())


if __name__ == "__main__":
    unittest.main()
