from __future__ import annotations

import os
import argparse
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"port must be an integer: {value!r}") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _wait_for_port(host: str, port: int, process: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"FastAPI exited before opening port {port} (exit={process.returncode})")
        if _is_port_open(host, port):
            return
        time.sleep(0.25)
    raise RuntimeError(f"FastAPI did not open port {port} within {timeout:.0f} seconds")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start the FeetFit FastAPI service and expose it through the configured ngrok CLI."
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=_port(os.getenv("NGROK_PORT", "8000")),
        help="local FastAPI/ngrok port (default: NGROK_PORT or 8000)",
    )
    args = parser.parse_args()
    port = args.port
    ngrok_executable = shutil.which("ngrok")
    if not ngrok_executable:
        parser.error("ngrok CLI was not found on PATH; install it and run 'ngrok config add-authtoken <TOKEN>'")

    server = None
    started_server = False
    if _is_port_open("127.0.0.1", port):
        print(f"Port {port} is already in use. Reusing the existing local server.")
    else:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ],
            cwd=PROJECT_ROOT,
        )
        started_server = True

    tunnel_process = None
    try:
        if server is not None:
            _wait_for_port("127.0.0.1", port, server)

        print()
        print("FastAPI server is ready. Starting ngrok...")
        print(f"Local docs:  http://127.0.0.1:{port}/docs")
        print("Use the HTTPS Forwarding URL printed by ngrok. Press Ctrl+C to stop.")
        print()
        tunnel_process = subprocess.Popen([ngrok_executable, "http", str(port)])
        return tunnel_process.wait()
    except KeyboardInterrupt:
        print("\nStopping server and ngrok tunnel...")
        return 0
    finally:
        if tunnel_process is not None and tunnel_process.poll() is None:
            tunnel_process.terminate()
            try:
                tunnel_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tunnel_process.kill()
        if started_server and server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
