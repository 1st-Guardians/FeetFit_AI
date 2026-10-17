from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.routes import shoe_ingestion
from app.core.config import settings
from app.core.ingestion_security import require_shoe_ingestion_admin_key
from app.schemas.shoe_ingestion import (
    VERIFIED_INGESTION_ACTION,
    VERIFIED_INGESTION_MANIFEST_SHA256,
    ShoeIngestionJobResponse,
    ShoeIngestionJobStatus,
)
from app.services.shoe import verified_ingestion_jobs as jobs


ADMIN_KEY = "admin-" + "a" * 40
SERVER_KEY = "server-" + "b" * 40


class _FakeProcess:
    pid = os.getpid()

    def __init__(self) -> None:
        self.release = threading.Event()

    def wait(self) -> int:
        self.release.wait()
        return 0


class _ExitedProcess:
    pid = 999_999_999

    def __init__(self, exit_code: int = 2) -> None:
        self.exit_code = exit_code

    def wait(self) -> int:
        return self.exit_code


class _NoopThread:
    def __init__(self, **_: object) -> None:
        pass

    def start(self) -> None:
        pass


class ShoeIngestionSecurityTest(unittest.TestCase):
    def test_feature_flag_fails_closed(self):
        with patch.object(settings, "shoe_ingestion_enabled", False):
            with self.assertRaises(HTTPException) as raised:
                require_shoe_ingestion_admin_key(ADMIN_KEY)
        self.assertEqual(raised.exception.status_code, 503)

    def test_short_or_reused_admin_key_fails_closed(self):
        with (
            patch.object(settings, "shoe_ingestion_enabled", True),
            patch.object(settings, "feetfit_server_internal_api_key", SERVER_KEY),
            patch.object(settings, "shoe_ingestion_admin_api_key", "x" * 31),
        ):
            with self.assertRaises(HTTPException) as short:
                require_shoe_ingestion_admin_key("x" * 31)
        self.assertEqual(short.exception.status_code, 503)

        with (
            patch.object(settings, "shoe_ingestion_enabled", True),
            patch.object(settings, "feetfit_server_internal_api_key", SERVER_KEY),
            patch.object(settings, "shoe_ingestion_admin_api_key", SERVER_KEY),
        ):
            with self.assertRaises(HTTPException) as reused:
                require_shoe_ingestion_admin_key(SERVER_KEY)
        self.assertEqual(reused.exception.status_code, 503)

    def test_missing_and_wrong_admin_keys_share_forbidden_response(self):
        with (
            patch.object(settings, "shoe_ingestion_enabled", True),
            patch.object(settings, "feetfit_server_internal_api_key", SERVER_KEY),
            patch.object(settings, "shoe_ingestion_admin_api_key", ADMIN_KEY),
        ):
            for provided in (None, "wrong-key"):
                with self.subTest(provided=provided):
                    with self.assertRaises(HTTPException) as raised:
                        require_shoe_ingestion_admin_key(provided)
                    self.assertEqual(raised.exception.status_code, 403)
                    self.assertEqual(
                        raised.exception.detail,
                        "Forbidden shoe ingestion request.",
                    )
            self.assertIsNone(require_shoe_ingestion_admin_key(ADMIN_KEY))


class ShoeIngestionJobServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.crawler = self.root / "shoe_crawler"
        self.artifacts = self.root / "jobs"
        self.crawler.mkdir()
        (self.crawler / jobs.WORKER_SCRIPT).write_text("# test worker\n", "utf-8")
        python_path = (
            self.crawler / "venv" / "Scripts" / "python.exe"
            if os.name == "nt"
            else self.crawler / "venv" / "bin" / "python"
        )
        python_path.parent.mkdir(parents=True)
        python_path.write_bytes(b"test executable")
        self.python_path = python_path
        self.config = jobs.ShoeIngestionJobConfig(
            crawler_root=self.crawler,
            artifact_root=self.artifacts,
            server_base_url="https://server.example:8443",
            server_internal_api_key=SERVER_KEY,
            target_label="production",
        )
        self.service = jobs.ShoeIngestionJobService(self.config)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _start(self) -> ShoeIngestionJobResponse:
        return self.service.start_job(
            action=VERIFIED_INGESTION_ACTION,
            target="production",
            manifest_sha256=VERIFIED_INGESTION_MANIFEST_SHA256,
        )

    def test_exact_fixed_argv_and_env_only_secret(self):
        process = _FakeProcess()
        with (
            patch.object(jobs.subprocess, "Popen", return_value=process) as popen,
            patch.object(jobs.threading, "Thread", _NoopThread),
        ):
            response = self._start()

        argv = popen.call_args.args[0]
        job_dir = self.artifacts.resolve() / response.job_id
        self.assertEqual(
            argv,
            [
                str(self.python_path.resolve()),
                str((self.crawler / jobs.WORKER_SCRIPT).resolve()),
                "--execute",
                "--output-dir",
                str(job_dir),
            ],
        )
        kwargs = popen.call_args.kwargs
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(kwargs["cwd"], str(self.crawler.resolve()))
        self.assertNotIn(SERVER_KEY, " ".join(argv))
        self.assertNotIn("https://server.example:8443", " ".join(argv))
        self.assertEqual(
            kwargs["env"]["FEETFIT_SERVER_INTERNAL_API_KEY"], SERVER_KEY
        )
        self.assertEqual(
            kwargs["env"]["FEETFIT_SERVER_BASE_URL"],
            "https://server.example:8443",
        )
        state_text = (job_dir / jobs.API_JOB_FILE).read_text("utf-8")
        self.assertNotIn(SERVER_KEY, state_text)
        self.assertNotIn("server.example", state_text)
        self.assertEqual(response.status, ShoeIngestionJobStatus.QUEUED)

    def test_confirmation_mismatch_is_rejected_before_launch(self):
        for field, value in (
            ("action", "ANY_OTHER_ACTION"),
            ("target", "staging"),
            ("manifest_sha256", "0" * 64),
        ):
            values = {
                "action": VERIFIED_INGESTION_ACTION,
                "target": "production",
                "manifest_sha256": VERIFIED_INGESTION_MANIFEST_SHA256,
            }
            values[field] = value
            with self.subTest(field=field):
                with patch.object(jobs.subprocess, "Popen") as popen:
                    with self.assertRaises(jobs.ShoeIngestionJobConfirmationError):
                        self.service.start_job(**values)
                    popen.assert_not_called()

    def test_public_plain_http_target_is_rejected(self):
        service = jobs.ShoeIngestionJobService(
            jobs.ShoeIngestionJobConfig(
                crawler_root=self.crawler,
                artifact_root=self.artifacts,
                server_base_url="http://34.209.169.111",
                server_internal_api_key=SERVER_KEY,
                target_label="production",
            )
        )
        with patch.object(jobs.subprocess, "Popen") as popen:
            with self.assertRaises(jobs.ShoeIngestionJobConfigurationError):
                service.start_job(
                    action=VERIFIED_INGESTION_ACTION,
                    target="production",
                    manifest_sha256=VERIFIED_INGESTION_MANIFEST_SHA256,
                )
            popen.assert_not_called()

    def test_manual_commit_check_keeps_single_writer_lock(self):
        with (
            patch.object(jobs.subprocess, "Popen", return_value=_FakeProcess()),
            patch.object(jobs.threading, "Thread", _NoopThread),
        ):
            first = self._start()
        first_dir = self.artifacts / first.job_id
        (first_dir / jobs.WORKER_JOB_FILE).write_text(
            json.dumps(
                {
                    "status": "NEEDS_MANUAL_COMMIT_CHECK",
                    "stage": "CANARY",
                    "failure": {
                        "reason": "REQUEST_COMMIT_OUTCOME_UNCERTAIN",
                        "databaseCommitVerificationRequired": True,
                    },
                }
            ),
            "utf-8",
        )
        with self.assertRaises(jobs.ShoeIngestionJobAlreadyRunningError) as raised:
            self._start()
        self.assertEqual(raised.exception.job_id, first.job_id)
        self.assertTrue((self.artifacts / jobs.ACTIVE_LOCK_DIRECTORY).is_dir())

    def test_nested_worker_failure_reason_is_the_only_exposed_error(self):
        with (
            patch.object(jobs.subprocess, "Popen", return_value=_FakeProcess()),
            patch.object(jobs.threading, "Thread", _NoopThread),
        ):
            started = self._start()
        job_dir = self.artifacts / started.job_id
        (job_dir / jobs.WORKER_JOB_FILE).write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "stage": "CANARY",
                    "failure": {
                        "reason": "CANARY_DID_NOT_COMPLETE_VERIFIED_REPLAY",
                        "errorType": "SecretBearingInternalException",
                    },
                    "progress": {"verifiedShoeCount": 338},
                }
            ),
            "utf-8",
        )
        status_result = self.service.get_job(started.job_id)
        self.assertEqual(status_result.status, ShoeIngestionJobStatus.FAILED)
        self.assertEqual(
            status_result.failure_code,
            "CANARY_DID_NOT_COMPLETE_VERIFIED_REPLAY",
        )
        self.assertNotIn("SecretBearing", status_result.model_dump_json())

    def test_api_manual_state_overrides_stale_running_worker_state(self):
        with (
            patch.object(jobs.subprocess, "Popen", return_value=_FakeProcess()),
            patch.object(jobs.threading, "Thread", _NoopThread),
        ):
            started = self._start()
        job_dir = self.artifacts / started.job_id
        metadata_path = job_dir / jobs.API_JOB_FILE
        metadata = json.loads(metadata_path.read_text("utf-8"))
        metadata.update(
            {
                "status": "NEEDS_MANUAL_COMMIT_CHECK",
                "failureCode": "WORKER_EXITED_WITHOUT_TERMINAL_STATE",
            }
        )
        metadata_path.write_text(json.dumps(metadata), "utf-8")
        (job_dir / jobs.WORKER_JOB_FILE).write_text(
            json.dumps({"status": "FULL_RUNNING", "stage": "FULL"}),
            "utf-8",
        )

        status_result = self.service.get_job(started.job_id)
        self.assertEqual(
            status_result.status,
            ShoeIngestionJobStatus.NEEDS_MANUAL_COMMIT_CHECK,
        )
        self.assertEqual(
            status_result.failure_code,
            "WORKER_EXITED_WITHOUT_TERMINAL_STATE",
        )

    def test_post_launch_tracking_failure_does_not_release_lock(self):
        real_atomic_write = jobs._atomic_json_write
        call_count = 0

        def fail_first_post_launch_write(path, document):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                raise OSError("simulated metadata failure")
            return real_atomic_write(path, document)

        with (
            patch.object(
                jobs.subprocess, "Popen", return_value=_FakeProcess()
            ) as popen,
            patch.object(jobs, "_atomic_json_write", fail_first_post_launch_write),
        ):
            with self.assertRaises(jobs.ShoeIngestionJobPostLaunchError) as raised:
                self._start()
            with self.assertRaises(jobs.ShoeIngestionJobAlreadyRunningError):
                self._start()
        owner = json.loads(
            (
                self.artifacts
                / jobs.ACTIVE_LOCK_DIRECTORY
                / "owner.json"
            ).read_text("utf-8")
        )
        self.assertEqual(owner["jobId"], raised.exception.job_id)
        metadata = json.loads(
            (
                self.artifacts
                / raised.exception.job_id
                / jobs.API_JOB_FILE
            ).read_text("utf-8")
        )
        self.assertNotIn("workerPid", metadata)
        self.assertEqual(popen.call_count, 1)

    def test_post_launch_recovery_state_records_pid_when_possible(self):
        real_atomic_write = jobs._atomic_json_write
        call_count = 0
        process = _FakeProcess()

        def fail_once(path, document):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise OSError("simulated first metadata failure")
            return real_atomic_write(path, document)

        with (
            patch.object(jobs.subprocess, "Popen", return_value=process),
            patch.object(jobs, "_atomic_json_write", fail_once),
        ):
            with self.assertRaises(jobs.ShoeIngestionJobPostLaunchError) as raised:
                self._start()
        metadata = json.loads(
            (
                self.artifacts
                / raised.exception.job_id
                / jobs.API_JOB_FILE
            ).read_text("utf-8")
        )
        self.assertEqual(metadata["workerPid"], process.pid)
        self.assertEqual(metadata["status"], "NEEDS_MANUAL_COMMIT_CHECK")
        self.assertTrue((self.artifacts / jobs.ACTIVE_LOCK_DIRECTORY).is_dir())

    def test_lock_lifecycle_guard_prevents_owner_check_rename_aba_race(self):
        with (
            patch.object(jobs.subprocess, "Popen", return_value=_FakeProcess()),
            patch.object(jobs.threading, "Thread", _NoopThread),
        ):
            started = self._start()
        job_dir = self.artifacts / started.job_id
        (job_dir / jobs.WORKER_JOB_FILE).write_text(
            json.dumps({"status": "COMPLETED", "stage": "COMPLETE"}),
            "utf-8",
        )

        owner_path = (
            self.artifacts.resolve()
            / jobs.ACTIVE_LOCK_DIRECTORY
            / "owner.json"
        )
        owner_read = threading.Event()
        allow_old_release = threading.Event()
        new_acquired = threading.Event()
        failures: list[BaseException] = []
        real_read = jobs._safe_json_read

        def pause_old_releaser(path, *, required):
            value = real_read(path, required=required)
            if (
                threading.current_thread().name == "old-release"
                and path == owner_path
                and not owner_read.is_set()
            ):
                owner_read.set()
                if not allow_old_release.wait(5):
                    raise AssertionError("test release gate timed out")
            return value

        new_job_id = str(uuid4())

        def release_old() -> None:
            try:
                self.service._release_active_lock(
                    self.artifacts.resolve(), started.job_id
                )
            except BaseException as exc:
                failures.append(exc)

        def acquire_new() -> None:
            try:
                self.service._acquire_active_lock(
                    self.artifacts.resolve(), new_job_id
                )
                new_acquired.set()
            except BaseException as exc:
                failures.append(exc)

        with patch.object(jobs, "_safe_json_read", pause_old_releaser):
            old_thread = threading.Thread(target=release_old, name="old-release")
            new_thread = threading.Thread(target=acquire_new, name="new-acquire")
            old_thread.start()
            self.assertTrue(owner_read.wait(2))
            new_thread.start()
            self.assertFalse(new_acquired.wait(0.1))
            allow_old_release.set()
            old_thread.join(5)
            new_thread.join(5)

        self.assertFalse(old_thread.is_alive())
        self.assertFalse(new_thread.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(new_acquired.is_set())
        owner = json.loads(owner_path.read_text("utf-8"))
        self.assertEqual(owner["jobId"], new_job_id)

    def test_worker_exit_before_ingestion_marks_failed_and_releases_lock(self):
        with (
            patch.object(jobs.subprocess, "Popen", return_value=_FakeProcess()),
            patch.object(jobs.threading, "Thread", _NoopThread),
        ):
            started = self._start()
        job_dir = self.artifacts / started.job_id

        self.service._monitor_worker(
            self.artifacts.resolve(),
            started.job_id,
            job_dir,
            _ExitedProcess(),
        )

        result = self.service.get_job(started.job_id)
        self.assertEqual(result.status, ShoeIngestionJobStatus.FAILED)
        self.assertEqual(result.failure_code, "WORKER_EXITED_BEFORE_INGESTION")
        self.assertFalse((self.artifacts / jobs.ACTIVE_LOCK_DIRECTORY).exists())

    def test_worker_exit_during_canary_requires_manual_check_and_keeps_lock(self):
        with (
            patch.object(jobs.subprocess, "Popen", return_value=_FakeProcess()),
            patch.object(jobs.threading, "Thread", _NoopThread),
        ):
            started = self._start()
        job_dir = self.artifacts / started.job_id
        (job_dir / jobs.WORKER_JOB_FILE).write_text(
            json.dumps({"status": "CANARY_RUNNING", "stage": "CANARY"}),
            "utf-8",
        )

        self.service._monitor_worker(
            self.artifacts.resolve(),
            started.job_id,
            job_dir,
            _ExitedProcess(),
        )

        result = self.service.get_job(started.job_id)
        self.assertEqual(
            result.status,
            ShoeIngestionJobStatus.NEEDS_MANUAL_COMMIT_CHECK,
        )
        self.assertEqual(
            result.failure_code,
            "WORKER_EXITED_WITHOUT_TERMINAL_STATE",
        )
        self.assertTrue((self.artifacts / jobs.ACTIVE_LOCK_DIRECTORY).is_dir())

    @unittest.skipUnless(os.name == "nt", "Windows-specific non-destructive probe")
    def test_windows_pid_probe_never_calls_os_kill(self):
        with patch.object(jobs.os, "kill", side_effect=AssertionError("must not signal")):
            self.assertTrue(jobs._pid_is_running(os.getpid()))

    def test_posix_pid_probe_only_treats_missing_process_as_dead(self):
        cases = (
            (ProcessLookupError(), False),
            (PermissionError(), True),
            (OSError("inconclusive"), True),
        )
        for error, expected in cases:
            with self.subTest(error=type(error).__name__), patch.object(
                jobs.os, "name", "posix"
            ), patch.object(jobs.os, "kill", side_effect=error):
                self.assertIs(expected, jobs._pid_is_running(123456))

    @unittest.skipUnless(os.name == "nt", "Windows-specific error classification")
    def test_windows_pid_probe_only_treats_invalid_parameter_as_dead(self):
        import ctypes

        kernel32 = MagicMock()
        kernel32.OpenProcess.return_value = 0
        for error_code, expected in ((87, False), (5, True), (6, True), (8, True)):
            with self.subTest(error_code=error_code), patch.object(
                ctypes, "WinDLL", return_value=kernel32
            ), patch.object(ctypes, "get_last_error", return_value=error_code):
                self.assertIs(expected, jobs._pid_is_running(123456))


class _FakeJobService:
    def __init__(self) -> None:
        self.last_start: dict[str, str] | None = None
        self.response = ShoeIngestionJobResponse(
            jobId="11111111-1111-4111-8111-111111111111",
            status="QUEUED",
            stage="LAUNCH",
            target="production",
            manifestSha256=VERIFIED_INGESTION_MANIFEST_SHA256,
            statusUrl=(
                "/api/admin/shoe-ingestion/jobs/"
                "11111111-1111-4111-8111-111111111111"
            ),
            createdAt="2026-09-04T00:00:00+00:00",
        )

    def start_job(self, **kwargs: str) -> ShoeIngestionJobResponse:
        self.last_start = kwargs
        return self.response

    def get_job(self, job_id: str) -> ShoeIngestionJobResponse:
        if job_id != self.response.job_id:
            raise jobs.ShoeIngestionJobNotFoundError()
        return self.response


class ShoeIngestionRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_service = _FakeJobService()
        application = FastAPI()
        application.include_router(
            shoe_ingestion.router, prefix="/api/admin/shoe-ingestion"
        )
        application.dependency_overrides[
            jobs.get_shoe_ingestion_job_service
        ] = lambda: self.fake_service
        self.client = TestClient(application)
        self.setting_patches = [
            patch.object(settings, "shoe_ingestion_enabled", True),
            patch.object(settings, "shoe_ingestion_admin_api_key", ADMIN_KEY),
            patch.object(settings, "feetfit_server_internal_api_key", SERVER_KEY),
        ]
        for setting_patch in self.setting_patches:
            setting_patch.start()

    def tearDown(self) -> None:
        for setting_patch in reversed(self.setting_patches):
            setting_patch.stop()

    def test_post_returns_202_location_and_never_accepts_connection_input(self):
        response = self.client.post(
            "/api/admin/shoe-ingestion/jobs",
            headers={"X-Shoe-Ingestion-Admin-Key": ADMIN_KEY},
            json={
                "action": VERIFIED_INGESTION_ACTION,
                "target": "production",
                "manifestSha256": VERIFIED_INGESTION_MANIFEST_SHA256,
            },
        )
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.headers["location"], self.fake_service.response.status_url)
        self.assertEqual(response.json()["expected"]["shoes"], 338)
        self.assertEqual(
            self.fake_service.last_start,
            {
                "action": VERIFIED_INGESTION_ACTION,
                "target": "production",
                "manifest_sha256": VERIFIED_INGESTION_MANIFEST_SHA256,
            },
        )

        rejected = self.client.post(
            "/api/admin/shoe-ingestion/jobs",
            headers={"X-Shoe-Ingestion-Admin-Key": ADMIN_KEY},
            json={
                "action": VERIFIED_INGESTION_ACTION,
                "target": "production",
                "manifestSha256": VERIFIED_INGESTION_MANIFEST_SHA256,
                "serverBaseUrl": "https://attacker.example",
                "internalApiKey": "attacker-controlled",
            },
        )
        self.assertEqual(rejected.status_code, 422)

    def test_missing_admin_key_is_forbidden_and_get_is_protected(self):
        body = {
            "action": VERIFIED_INGESTION_ACTION,
            "target": "production",
            "manifestSha256": VERIFIED_INGESTION_MANIFEST_SHA256,
        }
        self.assertEqual(
            self.client.post("/api/admin/shoe-ingestion/jobs", json=body).status_code,
            403,
        )
        get_response = self.client.get(
            self.fake_service.response.status_url,
            headers={"X-Shoe-Ingestion-Admin-Key": ADMIN_KEY},
        )
        self.assertEqual(get_response.status_code, 200)

    def test_openapi_uses_separate_admin_api_key_scheme(self):
        document = self.client.get("/openapi.json").json()
        schemes = document["components"]["securitySchemes"]
        self.assertEqual(
            schemes["신발 DB 적재 관리자 키"],
            {
                "type": "apiKey",
                "description": (
                    "검증된 신발 카탈로그 적재 작업 전용 관리자 키입니다. "
                    "Feetfit_Server 내부 API 키와 다른 값을 사용하세요."
                ),
                "in": "header",
                "name": "X-Shoe-Ingestion-Admin-Key",
            },
        )
        self.assertEqual(
            document["components"]["schemas"]["ShoeIngestionJobStartRequest"][
                "example"
            ],
            {
                "action": VERIFIED_INGESTION_ACTION,
                "target": "production",
                "manifestSha256": VERIFIED_INGESTION_MANIFEST_SHA256,
            },
        )


if __name__ == "__main__":
    unittest.main()
