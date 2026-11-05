"""Application logging with conservative secret redaction."""

import logging
import re
import sys

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "correlation_id=%(correlation_id)s | %(message)s"
)
_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|access[_-]?key|secret|password|token)"
    r"(\s*[=:]\s*)([^\s,;]+)"
)


class RedactingFilter(logging.Filter):
    """Redact common credential assignments from rendered messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        rendered = record.getMessage()
        record.msg = _SECRET_PATTERN.sub(r"\1\2[REDACTED]", rendered)
        record.args = ()

        if not hasattr(record, "correlation_id"):
            record.correlation_id = "-"

        return True


def configure_logging(log_level: str = "INFO") -> None:
    """Configure the backend logger once without exposing settings."""

    app_logger = logging.getLogger("app")
    app_logger.setLevel(log_level)
    app_logger.propagate = False

    if app_logger.handlers:
        return

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(RedactingFilter())
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    app_logger.addHandler(console_handler)
