from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

from dotenv import load_dotenv, set_key
from pyngrok.conf import get_default


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "storage" / "runtime" / "ngrok"
STATE_PATH = RUNTIME_DIR / "state.json"
STOP_PATH = RUNTIME_DIR / "stop.request"
LOCK_PATH = RUNTIME_DIR / "agent.lock"
OUT_LOG = RUNTIME_DIR / "stdout.log"
ERR_LOG = RUNTIME_DIR / "stderr.log"
INSPECTOR_URL = "http://127.0.0.1:4040/api/tunnels"


def tunnel_settings(*, new_session: bool = False) -> tuple[int, str | None]:
    load_dotenv(PROJECT_ROOT / ".env")
    port = int(os.getenv("NGROK_PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("NGROK_PORT must be between 1 and 65535.")
    if new_session:
        return port, None
    domain = os.getenv("NGROK_DOMAIN", "").strip()
    if not domain:
        raise ValueError("Set NGROK_DOMAIN to your assigned static ngrok domain in .env.")
    url = domain if "://" in domain else f"https://{domain}"
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.port
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise ValueError("NGROK_DOMAIN must be a hostname or an HTTPS URL without a path.")
    return port, f"https://{parts.hostname}"


def read_tunnels() -> list[dict]:
    # The inspector is local; global proxy settings must not intercept it.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(INSPECTOR_URL, timeout=1) as response:
            return json.load(response).get("tunnels", [])
    except (OSError, URLError, ValueError):
        return []


def matching_tunnel(port: int, url: str) -> dict | None:
    for tunnel in read_tunnels():
        if (
            tunnel.get("public_url", "").rstrip("/") == url
            and tunnel.get("config", {}).get("addr") == f"http://127.0.0.1:{port}"
        ):
            return tunnel
    return None


@contextmanager
def agent_lock():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_tunnel(port: int, url: str | None) -> int:
    # This process owns only ngrok, never the local API server.
    if not os.getenv("NGROK_AUTHTOKEN"):
        raise ValueError("Set NGROK_AUTHTOKEN in .env before starting ngrok.")
    binary = Path(os.getenv("NGROK_BIN") or get_default().ngrok_path)
    if not binary.is_file():
        raise ValueError("ngrok binary not found. Install ngrok or set NGROK_BIN in .env.")
    with agent_lock():
        if read_tunnels():
            raise ValueError("Another ngrok agent is using port 4040. Check it before starting.")
        STOP_PATH.unlink(missing_ok=True)
        command = [str(binary), "http", f"http://127.0.0.1:{port}"]
        if url is not None:
            command.append(f"--url={url}")
        command.extend(["--log=stdout", "--log-format=json"])
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, creationflags=flags,
            stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr,
        )
        try:
            state = {"pid": os.getpid(), "ngrok_pid": process.pid, "url": url, "port": port}
            STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
            print(f"ngrok URL: {url or 'waiting for a new session URL'}", flush=True)
            print(f"Upstream: http://127.0.0.1:{port}", flush=True)
            print("The local server can be stopped/restarted independently.", flush=True)
            while process.poll() is None:
                if STOP_PATH.exists():
                    return 0
                if url is None:
                    for tunnel in read_tunnels():
                        assigned_url = tunnel.get("public_url", "")
                        if (
                            assigned_url.startswith("https://")
                            and tunnel.get("config", {}).get("addr") == f"http://127.0.0.1:{port}"
                        ):
                            # Only --new-session opts in to replacing the saved URL.
                            set_key(str(PROJECT_ROOT / ".env"), "NGROK_DOMAIN", urlsplit(assigned_url).hostname, quote_mode="never")
                            set_key(str(PROJECT_ROOT / ".env"), "PUBLIC_BASE_URL", assigned_url, quote_mode="never")
                            url = assigned_url
                            state["url"] = url
                            STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
                            print(f"Session URL saved in .env: {url}", flush=True)
                            print("Keep ngrok running to preserve this URL when restarting the local API.", flush=True)
                            break
                time.sleep(0.5)
            print(f"ngrok exited with code {process.returncode}.", flush=True)
            return process.returncode or 0
        except KeyboardInterrupt:
            return 0
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            STATE_PATH.unlink(missing_ok=True)
            STOP_PATH.unlink(missing_ok=True)


def start_background(port: int, url: str | None) -> int:
    if url is None and STATE_PATH.exists():
        try:
            existing = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            existing_url = existing.get("url") if existing.get("port") == port else None
            if existing_url and matching_tunnel(port, existing_url):
                print(f"ngrok is already running: {existing_url}")
                return 0
        except (OSError, ValueError):
            pass
    if url is not None and matching_tunnel(port, url):
        print(f"ngrok is already running: {url}")
        return 0
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    # Check the lock before truncating logs belonging to a running agent.
    with agent_lock():
        pass
    with OUT_LOG.open("w", encoding="utf-8") as out, ERR_LOG.open("w", encoding="utf-8") as err:
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        command = [sys.executable, "-u", str(Path(__file__).resolve())]
        if url is None:
            command.append("--new-session")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            creationflags=flags, start_new_session=os.name != "nt",
        )
    for _ in range(40):
        if url is None and STATE_PATH.exists():
            try:
                state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                if state.get("pid") == process.pid:
                    url = state.get("url")
            except (OSError, ValueError):
                pass
        if url is not None and matching_tunnel(port, url):
            print(f"ngrok is running in the background: {url}")
            print(f"Swagger: {url}/docs")
            print("Start/stop the local API with run_local_full_sam_api.cmd and Ctrl+C.")
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.5)
    # Stop only the helper we just created if it cannot establish the requested URL.
    if process.poll() is None:
        STOP_PATH.touch()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
    print(f"Could not connect the requested ngrok URL: {url or 'new session'}")
    print("Check your assigned static domain at https://dashboard.ngrok.com/domains")
    print(f"Logs: {OUT_LOG} and {ERR_LOG}")
    return 1


def stop_background() -> int:
    if not STATE_PATH.exists():
        print("No managed ngrok agent is running.")
        return 0
    STOP_PATH.touch()
    for _ in range(20):
        if not STATE_PATH.exists():
            print("ngrok stopped. The local API server was left running.")
            return 0
        time.sleep(0.5)
    print(f"ngrok did not acknowledge the stop request. Check {ERR_LOG}.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep ngrok running separately from the foreground local API.")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--background", action="store_true")
    actions.add_argument("--stop", action="store_true")
    actions.add_argument("--status", action="store_true")
    parser.add_argument(
        "--new-session", action="store_true",
        help="Explicitly create one new session URL and save it in .env; keep ngrok running to preserve it.",
    )
    args = parser.parse_args()
    try:
        if args.new_session and (args.stop or args.status):
            raise ValueError("--new-session can only be used to start ngrok.")
        if args.stop:
            return stop_background()
        port, url = tunnel_settings(new_session=args.new_session)
        if args.status:
            connected = matching_tunnel(port, url) is not None
            print(f"ngrok: {'connected' if connected else 'not connected'}")
            print(f"Saved URL: {url}")
            print(f"Upstream: http://127.0.0.1:{port}")
            print(f"Logs: {RUNTIME_DIR}")
            return 0 if connected else 1
        if args.background:
            return start_background(port, url)
        return run_tunnel(port, url)
    except (OSError, ValueError) as exc:
        print(f"ngrok: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
