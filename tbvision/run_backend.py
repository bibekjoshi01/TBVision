"""Launch the TBVision FastAPI app with production-ready defaults."""

from __future__ import annotations

import os
from multiprocessing import cpu_count

import uvicorn


def main() -> None:
    """Run uvicorn with sensible defaults for local production."""

    host = os.getenv("TBVISION_HOST", "0.0.0.0")
    port = int(os.getenv("TBVISION_PORT", 8000))
    reload_flag = os.getenv("TBVISION_RELOAD", "false").lower() == "true"
    workers = (
        1
        if reload_flag
        else int(os.getenv("TBVISION_WORKERS", max(1, cpu_count() - 1)))
    )
    log_level = os.getenv("TBVISION_LOG_LEVEL", "info")

    uvicorn.run(
        "tbvision.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=reload_flag,
        loop="auto",
        http="auto",
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
