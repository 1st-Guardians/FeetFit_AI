"""Durable launcher for the crawler-owned verified ingestion reconciler.

The HTTP process owns only admission control and process launch.  The crawler
worker owns validation, ingestion, resume semantics, and its durable
``job-state.json``.  Connection credentials are passed through a minimal child
environment and are never written to argv or API-owned state.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from app.core.config import settings
from app.schemas.shoe_ingestion import (
    VERIFIED_INGESTION_ACTION,
    VERIFIED_INGESTION_MANIFEST_SHA256,
    ShoeIngestionExpectedCounts,
    ShoeIngestionJobResponse,
    ShoeIngestionJobStatus,
)


API_JOB_FORMAT = "feetfit-ai-shoe-ingestion-job"
API_JOB_VERSION = 1
API_JOB_FILE = "api-job.json"
WORKER_JOB_FILE = "job-state.json"
WORKER_SCRIPT = "verified_ingestion_reconcile.py"
ACTIVE_LOCK_DIRECTORY = ".active-job"
ACTIVE_LOCK_GUARD_FILE = ".active-job.guard"
MAX_STATE_BYTES = 2 * 1024 * 1024
LOCK_GUARD_TIMEOUT_SECONDS = 10.0

_WORKER_STATUSES = {
    "VALIDATING": ShoeIngestionJobStatus.VALIDATING,
    "CANARY_RUNNING": ShoeIngestionJobStatus.CANARY_RUNNING,
    "FULL_RUNNING": ShoeIngestionJobStatus.FULL_RUNNING,
    "COMPLETED": ShoeIngestionJobStatus.COMPLETED,
    "FAILED": ShoeIngestionJobStatus.FAILED,
    "NEEDS_MANUAL_COMMIT_CHECK": (
        ShoeIngestionJobStatus.NEEDS_MANUAL_COMMIT_CHECK
    ),
}
_RECYCLABLE_TERMINAL_STATUSES = {"COMPLETED", "FAILED"}
_SAFE_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_SAFE_PROGRESS_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_LOCK_LIFECYCLE_MUTEX = threading.RLock()


class PopenProcess(Protocol):
    pid: int

    def wait(self) -> int: ...


class ShoeIngestionJobError(RuntimeError):
    pass


class ShoeIngestionJobConfigurationError(ShoeIngestionJobError):
    pass


class ShoeIngestionJobConfirmationError(ShoeIngestionJobError):
    pass


class ShoeIngestionJobNotFoundError(ShoeIngestionJobError):
    pass


class ShoeIngestionJobAlreadyRunningError(ShoeIngestionJobError):
    def __init__(self, job_id: str | None) -> None:
        super().__init__("A verified shoe ingestion job is already active.")
        self.job_id = job_id


class ShoeIngestionJobPostLaunchError(ShoeIngestionJobError):
    def __init__(self, job_id: str) -> None:
        super().__init__(
            "The ingestion worker started, but API-side tracking setup failed."
        )
        self.job_id = job_id


@dataclass(frozen=True, slots=True)
class ShoeIngestionJobConfig:
    crawler_root: Path
    artifact_root: Path
    server_base_url: str
    server_internal_api_key: str
    target_label: str

    @classmethod
    def from_settings(cls) -> "ShoeIngestionJobConfig":
        return cls(
            crawler_root=settings.shoe_ingestion_crawler_root,
            artifact_root=settings.shoe_ingestion_artifact_root,
            server_base_url=settings.feetfit_server_base_url,
            server_internal_api_key=settings.feetfit_server_internal_api_key,
            target_label=settings.shoe_ingestion_target_label,
        )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_json_read(path: Path, *, required: bool) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise ShoeIngestionJobError("Required ingestion job state is missing.")
        return None
    if path.is_symlink() or not path.is_file():
        raise ShoeIngestionJobError("Ingestion job state path is unsafe.")
    if path.stat().st_size > MAX_STATE_BYTES:
        raise ShoeIngestionJobError("Ingestion job state is unexpectedly large.")
    try:
        value = json.loads(path.read_text("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShoeIngestionJobError("Ingestion job state is invalid.") from exc
    if not isinstance(value, dict):
        raise ShoeIngestionJobError("Ingestion job state root is invalid.")
    return value


def _atomic_json_write(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists() and path.is_symlink():
        raise ShoeIngestionJobError("Refusing to replace a state symlink.")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _normalized_safe_target(value: str) -> str:
    raw = value.strip()
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise ShoeIngestionJobConfigurationError(
            "FEETFIT_SERVER_BASE_URL is invalid."
        ) from exc
    scheme = parsed.scheme.casefold()
    if (
        scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ShoeIngestionJobConfigurationError(
            "FEETFIT_SERVER_BASE_URL must be an origin without credentials, path, query, or fragment."
        )

    hostname = parsed.hostname.casefold().rstrip(".")
    is_loopback = hostname == "localhost"
    try:
        is_loopback = is_loopback or ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        pass
    if scheme == "http" and not is_loopback:
        raise ShoeIngestionJobConfigurationError(
            "Verified ingestion requires HTTPS, or HTTP through a loopback tunnel."
        )

    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 443 if scheme == "https" else 80
    port_suffix = "" if port is None or port == default_port else f":{port}"
    return f"{scheme}://{rendered_host}{port_suffix}"


def _pid_is_running(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        # ``os.kill(pid, 0)`` is not a harmless POSIX-style probe on Windows.
        # Query a synchronization handle and poll it without signalling the
        # child process.
        import ctypes
        from ctypes import wintypes

        synchronize = 0x00100000
        wait_object_0 = 0x00000000
        wait_timeout = 0x00000102
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            # ERROR_INVALID_PARAMETER is the documented missing-PID result.
            # Access denial and every other probe failure are inconclusive, so
            # fail safe and keep the writer lock.
            return ctypes.get_last_error() != 87
        try:
            result = kernel32.WaitForSingleObject(handle, 0)
            if result == wait_timeout:
                return True
            if result == wait_object_0:
                return False
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # An unknown probe failure does not prove that the process stopped.
        return True
    return True


@contextmanager
def _active_lock_lifecycle_guard(artifact_root: Path) -> Iterator[None]:
    """Serialize owner-check/retire/create as one cross-process operation."""

    guard_path = artifact_root / ACTIVE_LOCK_GUARD_FILE
    with _LOCK_LIFECYCLE_MUTEX:
        if guard_path.is_symlink():
            raise ShoeIngestionJobError("Active ingestion guard path is unsafe.")
        handle = guard_path.open("a+b")
        locked = False
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            deadline = time.monotonic() + LOCK_GUARD_TIMEOUT_SECONDS
            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        locked = True
                        break
                    except OSError as exc:
                        if time.monotonic() >= deadline:
                            raise ShoeIngestionJobError(
                                "Timed out acquiring the ingestion lifecycle guard."
                            ) from exc
                        time.sleep(0.01)
            else:
                import fcntl

                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locked = True
                        break
                    except BlockingIOError as exc:
                        if time.monotonic() >= deadline:
                            raise ShoeIngestionJobError(
                                "Timed out acquiring the ingestion lifecycle guard."
                            ) from exc
                        time.sleep(0.01)
            yield
        finally:
            if locked:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


class ShoeIngestionJobService:
    def __init__(self, config: ShoeIngestionJobConfig) -> None:
        self._config = config

    def start_job(
        self,
        *,
        action: str,
        target: str,
        manifest_sha256: str,
    ) -> ShoeIngestionJobResponse:
        if (
            action != VERIFIED_INGESTION_ACTION
            or target != self._config.target_label.strip()
            or manifest_sha256 != VERIFIED_INGESTION_MANIFEST_SHA256
        ):
            raise ShoeIngestionJobConfirmationError(
                "The action, target, or verified manifest confirmation does not match configuration."
            )

        artifact_root = self._artifact_root()
        crawler_root, python_executable, worker_script = self._worker_paths()
        target_origin = _normalized_safe_target(self._config.server_base_url)
        internal_key = self._config.server_internal_api_key.strip()
        if not internal_key:
            raise ShoeIngestionJobConfigurationError(
                "FEETFIT_SERVER_INTERNAL_API_KEY is not configured."
            )

        job_id = str(uuid4())
        self._acquire_active_lock(artifact_root, job_id)
        job_dir = artifact_root / job_id
        metadata_path = job_dir / API_JOB_FILE
        process: PopenProcess | None = None
        try:
            job_dir.mkdir(parents=False, exist_ok=False)
            created_at = _now()
            metadata: dict[str, Any] = {
                "format": API_JOB_FORMAT,
                "version": API_JOB_VERSION,
                "jobId": job_id,
                "status": ShoeIngestionJobStatus.QUEUED.value,
                "stage": "LAUNCH",
                "createdAt": created_at,
                "updatedAt": created_at,
                "target": self._config.target_label.strip(),
                "manifestSha256": VERIFIED_INGESTION_MANIFEST_SHA256,
                "expected": ShoeIngestionExpectedCounts().model_dump(by_alias=True),
            }
            _atomic_json_write(metadata_path, metadata)

            argv = [
                str(python_executable),
                str(worker_script),
                "--execute",
                "--output-dir",
                str(job_dir),
            ]
            log_handle = (job_dir / "worker.log").open("ab", buffering=0)
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=str(crawler_root),
                    env=self._worker_environment(target_origin, internal_key),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    close_fds=True,
                    creationflags=(
                        getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        if os.name == "nt"
                        else 0
                    ),
                    start_new_session=os.name != "nt",
                )
            finally:
                log_handle.close()

            metadata["workerPid"] = process.pid
            metadata["stage"] = "PROCESS_STARTED"
            metadata["updatedAt"] = _now()
            _atomic_json_write(metadata_path, metadata)
            monitor = threading.Thread(
                target=self._monitor_worker,
                args=(artifact_root, job_id, job_dir, process),
                name=f"shoe-ingestion-{job_id[:8]}",
                daemon=True,
            )
            monitor.start()
        except Exception as exc:
            post_launch = process is not None
            if metadata_path.exists():
                try:
                    metadata = _safe_json_read(metadata_path, required=True) or {}
                    if post_launch:
                        metadata["workerPid"] = process.pid
                    metadata.update(
                        {
                            "status": (
                                ShoeIngestionJobStatus.NEEDS_MANUAL_COMMIT_CHECK.value
                                if post_launch
                                else ShoeIngestionJobStatus.FAILED.value
                            ),
                            "stage": (
                                "PROCESS_STARTED" if post_launch else "LAUNCH"
                            ),
                            "failureCode": (
                                "POST_LAUNCH_TRACKING_FAILED"
                                if post_launch
                                else "WORKER_LAUNCH_FAILED"
                            ),
                            "updatedAt": _now(),
                        }
                    )
                    if not post_launch:
                        metadata["completedAt"] = _now()
                    _atomic_json_write(metadata_path, metadata)
                except Exception:
                    pass
            if post_launch:
                raise ShoeIngestionJobPostLaunchError(job_id) from exc
            self._release_active_lock(artifact_root, job_id)
            if isinstance(exc, ShoeIngestionJobError):
                raise
            raise ShoeIngestionJobError(
                "Verified shoe ingestion worker could not be launched."
            ) from exc

        return self.get_job(job_id)

    def get_job(self, job_id: str) -> ShoeIngestionJobResponse:
        try:
            canonical_job_id = str(UUID(job_id))
        except (ValueError, AttributeError) as exc:
            raise ShoeIngestionJobNotFoundError("Ingestion job was not found.") from exc
        artifact_root = self._artifact_root()
        job_dir = artifact_root / canonical_job_id
        if job_dir.parent != artifact_root or not job_dir.is_dir() or job_dir.is_symlink():
            raise ShoeIngestionJobNotFoundError("Ingestion job was not found.")
        metadata = _safe_json_read(job_dir / API_JOB_FILE, required=True)
        if (
            metadata is None
            or metadata.get("format") != API_JOB_FORMAT
            or metadata.get("version") != API_JOB_VERSION
            or metadata.get("jobId") != canonical_job_id
        ):
            raise ShoeIngestionJobNotFoundError("Ingestion job was not found.")
        worker = _safe_json_read(job_dir / WORKER_JOB_FILE, required=False)
        return self._snapshot(metadata, worker)

    def _artifact_root(self) -> Path:
        supplied = self._config.artifact_root.expanduser()
        if supplied.exists() and supplied.is_symlink():
            raise ShoeIngestionJobConfigurationError(
                "Shoe ingestion artifact root must not be a symlink."
            )
        supplied.mkdir(parents=True, exist_ok=True)
        root = supplied.resolve()
        if not root.is_dir():
            raise ShoeIngestionJobConfigurationError(
                "Shoe ingestion artifact root is not a directory."
            )
        return root

    def _worker_paths(self) -> tuple[Path, Path, Path]:
        supplied = self._config.crawler_root.expanduser()
        if supplied.is_symlink():
            raise ShoeIngestionJobConfigurationError(
                "Shoe crawler root must not be a symlink."
            )
        root = supplied.resolve()
        if not root.is_dir():
            raise ShoeIngestionJobConfigurationError(
                "Configured shoe crawler root does not exist."
            )
        worker = root / WORKER_SCRIPT
        python_executable = (
            root / "venv" / "Scripts" / "python.exe"
            if os.name == "nt"
            else root / "venv" / "bin" / "python"
        )
        if (
            worker.is_symlink()
            or not worker.is_file()
            or worker.resolve().parent != root
            or python_executable.is_symlink()
            or not python_executable.is_file()
        ):
            raise ShoeIngestionJobConfigurationError(
                "Verified crawler worker or crawler virtualenv Python is missing or unsafe."
            )
        return root, python_executable.resolve(), worker.resolve()

    @staticmethod
    def _worker_environment(target_origin: str, internal_key: str) -> dict[str, str]:
        environment: dict[str, str] = {}
        for name in (
            "SystemRoot",
            "WINDIR",
            "COMSPEC",
            "PATH",
            "PATHEXT",
            "TEMP",
            "TMP",
            "TMPDIR",
            "LANG",
            "LC_ALL",
            "SSL_CERT_FILE",
            "REQUESTS_CA_BUNDLE",
        ):
            value = os.environ.get(name)
            if value:
                environment[name] = value
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "FEETFIT_SERVER_BASE_URL": target_origin,
                "FEETFIT_SERVER_INTERNAL_API_KEY": internal_key,
            }
        )
        return environment

    def _acquire_active_lock(self, artifact_root: Path, job_id: str) -> None:
        with _active_lock_lifecycle_guard(artifact_root):
            lock_dir = artifact_root / ACTIVE_LOCK_DIRECTORY
            while True:
                if lock_dir.is_symlink():
                    raise ShoeIngestionJobError(
                        "Active ingestion lock path is unsafe."
                    )
                try:
                    lock_dir.mkdir()
                except FileExistsError:
                    owner = _safe_json_read(
                        lock_dir / "owner.json", required=False
                    )
                    active_job_id = (
                        owner.get("jobId")
                        if isinstance(owner, dict)
                        and isinstance(owner.get("jobId"), str)
                        else None
                    )
                    if active_job_id and self._is_recyclable_terminal(
                        artifact_root, active_job_id
                    ):
                        if self._release_active_lock_unlocked(
                            artifact_root, active_job_id
                        ):
                            continue
                    raise ShoeIngestionJobAlreadyRunningError(active_job_id)
                try:
                    _atomic_json_write(
                        lock_dir / "owner.json",
                        {"jobId": job_id, "createdAt": _now()},
                    )
                except Exception:
                    try:
                        lock_dir.rmdir()
                    except OSError:
                        pass
                    raise
                return

    @staticmethod
    def _is_recyclable_terminal(artifact_root: Path, job_id: str) -> bool:
        try:
            UUID(job_id)
        except ValueError:
            return False
        job_dir = artifact_root / job_id
        worker = _safe_json_read(job_dir / WORKER_JOB_FILE, required=False)
        if worker and worker.get("status") in _RECYCLABLE_TERMINAL_STATUSES:
            return True
        metadata = _safe_json_read(job_dir / API_JOB_FILE, required=False)
        if metadata and metadata.get("status") == "FAILED":
            return True
        # The worker creates job-state.json before any POST.  A dead child with
        # no state, or one still in VALIDATING, therefore cannot have mutated
        # Feetfit_Server and is safe to replace.
        if (
            metadata
            and "workerPid" in metadata
            and not _pid_is_running(metadata.get("workerPid"))
        ):
            return worker is None or worker.get("status") == "VALIDATING"
        return False

    @staticmethod
    def _release_active_lock(artifact_root: Path, job_id: str) -> bool:
        with _active_lock_lifecycle_guard(artifact_root):
            return ShoeIngestionJobService._release_active_lock_unlocked(
                artifact_root, job_id
            )

    @staticmethod
    def _release_active_lock_unlocked(artifact_root: Path, job_id: str) -> bool:
        lock_dir = artifact_root / ACTIVE_LOCK_DIRECTORY
        owner = _safe_json_read(lock_dir / "owner.json", required=False)
        if not owner or owner.get("jobId") != job_id:
            return False
        retired = artifact_root / f".retired-active-job-{uuid4()}"
        try:
            lock_dir.rename(retired)
        except (FileNotFoundError, FileExistsError, PermissionError):
            return False
        try:
            (retired / "owner.json").unlink(missing_ok=True)
            retired.rmdir()
        except OSError:
            pass
        return True

    def _monitor_worker(
        self,
        artifact_root: Path,
        job_id: str,
        job_dir: Path,
        process: PopenProcess,
    ) -> None:
        exit_code = process.wait()
        metadata_path = job_dir / API_JOB_FILE
        try:
            metadata = _safe_json_read(metadata_path, required=True) or {}
            worker = _safe_json_read(job_dir / WORKER_JOB_FILE, required=False)
            worker_status = worker.get("status") if worker else None
            metadata["workerExitCode"] = exit_code
            metadata["workerExitedAt"] = _now()
            metadata["updatedAt"] = _now()
            if worker_status is None or worker_status == "VALIDATING":
                metadata["status"] = ShoeIngestionJobStatus.FAILED.value
                metadata["stage"] = "VALIDATION"
                metadata["failureCode"] = "WORKER_EXITED_BEFORE_INGESTION"
                metadata["completedAt"] = _now()
            elif worker_status not in {
                "COMPLETED",
                "FAILED",
                "NEEDS_MANUAL_COMMIT_CHECK",
            }:
                metadata["status"] = (
                    ShoeIngestionJobStatus.NEEDS_MANUAL_COMMIT_CHECK.value
                )
                metadata["stage"] = "PROCESS_EXIT"
                metadata["failureCode"] = (
                    "WORKER_EXITED_WITHOUT_TERMINAL_STATE"
                )
            _atomic_json_write(metadata_path, metadata)
            if worker_status in _RECYCLABLE_TERMINAL_STATUSES or worker_status in {
                None,
                "VALIDATING",
            }:
                self._release_active_lock(artifact_root, job_id)
        except Exception:
            # A corrupt/missing terminal state is deliberately left locked.  A
            # second catalog writer must not be admitted after an uncertain exit.
            return

    @staticmethod
    def _snapshot(
        metadata: Mapping[str, Any], worker: Mapping[str, Any] | None
    ) -> ShoeIngestionJobResponse:
        raw_worker_status = worker.get("status") if worker else None
        status_value = _WORKER_STATUSES.get(str(raw_worker_status))
        failure_code = ShoeIngestionJobService._safe_failure_code(worker)
        worker_pid_alive = _pid_is_running(metadata.get("workerPid"))
        if (
            metadata.get("status")
            == ShoeIngestionJobStatus.NEEDS_MANUAL_COMMIT_CHECK.value
            and raw_worker_status not in _RECYCLABLE_TERMINAL_STATUSES
        ):
            status_value = ShoeIngestionJobStatus.NEEDS_MANUAL_COMMIT_CHECK
            candidate = metadata.get("failureCode")
            if isinstance(candidate, str) and _SAFE_CODE.fullmatch(candidate):
                failure_code = candidate
        elif raw_worker_status == "VALIDATING" and not worker_pid_alive:
            status_value = ShoeIngestionJobStatus.FAILED
            failure_code = "WORKER_EXITED_BEFORE_INGESTION"
        elif raw_worker_status in {"CANARY_RUNNING", "FULL_RUNNING"} and not worker_pid_alive:
            status_value = ShoeIngestionJobStatus.NEEDS_MANUAL_COMMIT_CHECK
            failure_code = "WORKER_PROCESS_NOT_RUNNING"
        if status_value is None:
            raw_api_status = str(metadata.get("status") or "")
            try:
                status_value = ShoeIngestionJobStatus(raw_api_status)
            except ValueError:
                status_value = ShoeIngestionJobStatus.QUEUED
            if (
                status_value == ShoeIngestionJobStatus.QUEUED
                and not worker_pid_alive
                and "workerPid" in metadata
            ):
                status_value = ShoeIngestionJobStatus.NEEDS_MANUAL_COMMIT_CHECK
                failure_code = "WORKER_PROCESS_NOT_RUNNING"

        stage = worker.get("stage") if worker else metadata.get("stage")
        if not isinstance(stage, str) or not _SAFE_CODE.fullmatch(stage):
            stage = None
        if failure_code is None:
            candidate = metadata.get("failureCode")
            if isinstance(candidate, str) and _SAFE_CODE.fullmatch(candidate):
                failure_code = candidate

        progress = ShoeIngestionJobService._safe_progress(worker)
        completed_at = (
            worker.get("completedAt") if worker else metadata.get("completedAt")
        )
        updated_at = worker.get("updatedAt") if worker else metadata.get("updatedAt")
        return ShoeIngestionJobResponse(
            jobId=str(metadata["jobId"]),
            status=status_value,
            stage=stage,
            target=str(metadata["target"]),
            manifestSha256=str(metadata["manifestSha256"]),
            statusUrl=f"/api/admin/shoe-ingestion/jobs/{metadata['jobId']}",
            expected=ShoeIngestionExpectedCounts.model_validate(metadata["expected"]),
            progress=progress,
            createdAt=str(metadata["createdAt"]),
            updatedAt=updated_at if isinstance(updated_at, str) else None,
            completedAt=completed_at if isinstance(completed_at, str) else None,
            failureCode=failure_code,
        )

    @staticmethod
    def _safe_failure_code(worker: Mapping[str, Any] | None) -> str | None:
        if not worker:
            return None
        for key in ("failureCode", "errorCode"):
            value = worker.get(key)
            if isinstance(value, str) and _SAFE_CODE.fullmatch(value):
                return value
        failure = worker.get("failure")
        if isinstance(failure, Mapping):
            reason = failure.get("reason")
            if isinstance(reason, str) and _SAFE_CODE.fullmatch(reason):
                return reason
        return None

    @staticmethod
    def _safe_progress(worker: Mapping[str, Any] | None) -> dict[str, int]:
        if not worker:
            return {}
        result: dict[str, int] = {}

        def collect(value: object, prefix: str = "") -> None:
            if not isinstance(value, Mapping) or len(result) >= 32:
                return
            for raw_key, item in value.items():
                if not isinstance(raw_key, str):
                    continue
                key = f"{prefix}.{raw_key}" if prefix else raw_key
                if not _SAFE_PROGRESS_KEY.fullmatch(key):
                    continue
                if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                    result[key] = item
                elif isinstance(item, Mapping) and key.count(".") < 2:
                    collect(item, key)

        for container_name in ("counts", "progress", "summary"):
            collect(worker.get(container_name), container_name)
        for key, value in worker.items():
            if (
                isinstance(key, str)
                and key.lower().endswith("count")
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                and _SAFE_PROGRESS_KEY.fullmatch(key)
            ):
                result[key] = value
        return dict(sorted(result.items()))


def get_shoe_ingestion_job_service() -> ShoeIngestionJobService:
    return ShoeIngestionJobService(ShoeIngestionJobConfig.from_settings())


__all__ = [
    "ShoeIngestionJobAlreadyRunningError",
    "ShoeIngestionJobConfig",
    "ShoeIngestionJobConfigurationError",
    "ShoeIngestionJobConfirmationError",
    "ShoeIngestionJobError",
    "ShoeIngestionJobNotFoundError",
    "ShoeIngestionJobPostLaunchError",
    "ShoeIngestionJobService",
    "get_shoe_ingestion_job_service",
]
