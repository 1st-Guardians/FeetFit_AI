from __future__ import annotations

import logging
import sys


CONSOLE_LOG_FORMAT = "%(levelname)s: %(message)s"


def configure_console_logging(level: int = logging.INFO) -> None:
    """PowerShell에서 애플리케이션 INFO 로그가 바로 보이도록 설정한다.

    Uvicorn의 access/error logger 설정은 그대로 두고, root logger에 handler가
    없는 일반 실행 환경에서만 FeetFit용 console handler를 추가한다.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(CONSOLE_LOG_FORMAT))
        root_logger.addHandler(console_handler)

    logging.getLogger("app").setLevel(level)
