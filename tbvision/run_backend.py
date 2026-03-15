"""Launch the TBVision FastAPI app using the uvicorn CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from multiprocessing import cpu_count


def main() -> None:
    """Run uvicorn using the same mechanism as `python -m uvicorn`."""

    host = os.getenv("TBVISION_HOST", "0.0.0.0")
    port = os.getenv("TBVISION_PORT", "8000")
    reload_flag = os.getenv("TBVISION_RELOAD", "false").lower() == "true"
    log_level = os.getenv("TBVISION_LOG_LEVEL", "info")

    workers = (
        "1"
        if reload_flag
        else os.getenv("TBVISION_WORKERS", str(max(1, cpu_count() - 1)))
    )

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "tbvision.main:app",
        "--host",
        host,
        "--port",
        port,
        "--log-level",
        log_level,
        "--proxy-headers",
        "--forwarded-allow-ips",
        "*",
    ]

    if reload_flag:
        cmd.append("--reload")
    else:
        cmd.extend(["--workers", workers])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
