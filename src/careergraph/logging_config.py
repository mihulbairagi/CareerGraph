"""Structured logging setup.

Rules of the road for this codebase:

* No ``print()``. Ever. ``print`` cannot be filtered by level, routed to a
  file, or parsed by a log aggregator.
* Every module gets its logger via ``get_logger(__name__)``, which yields
  dotted names like ``careergraph.agents.qa`` so you can raise or lower the
  verbosity of one subsystem without touching the others.
* Configuration happens exactly once, at the process entrypoint (API
  lifespan, CLI ``main``), never at import time. Libraries that configure
  logging on import stomp on the host application's setup.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_CONFIGURED = False

# Attributes present on every LogRecord; anything else was passed by the
# caller via `extra={...}` and therefore belongs in the structured payload.
_RESERVED_RECORD_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line, merging in any ``extra=`` fields.

    Human-readable text is nicer at a terminal; JSON is what you want the
    moment logs are shipped somewhere queryable. ``LOG_FORMAT`` picks.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Readable single-line format that still surfaces ``extra=`` fields."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)-34s %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED_RECORD_ATTRS and not k.startswith("_")
        }
        if extras:
            rendered = " ".join(f"{k}={v!r}" for k, v in extras.items())
            base = f"{base} | {rendered}"
        return base


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Install a single stderr handler on the root logger. Idempotent.

    Logs go to **stderr**, not stdout, so that CLI tools in this project can
    pipe machine-readable results on stdout without log lines corrupting them.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These libraries are extremely chatty at INFO and drown out our own
    # signal (Chroma logs every segment read, httpx logs every request).
    for noisy in ("chromadb", "httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Use as ``get_logger(__name__)``."""
    return logging.getLogger(name)
