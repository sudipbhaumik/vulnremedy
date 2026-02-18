import structlog
import logging
from vulnremedy.utils.config import settings


def setup_logging() -> None:
    """
    Configure structured JSON logging.
    In development: pretty colored output.
    In production: JSON lines for log aggregators (CloudWatch, Datadog).
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if settings.is_development:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer()
        ]
    else:
        processors = shared_processors + [
            structlog.processors.JSONRenderer()
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


# Module-level logger — import and use anywhere
logger = structlog.get_logger()