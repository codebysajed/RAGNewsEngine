from logging.handlers import RotatingFileHandler
from pathlib import Path
import logging


BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "storage" / "logs"
LOG_FILE = LOG_DIR / "app.log"

_CONFIGURED = False
_LOGGER_NAME = "news"


def configure_logging():
    global _CONFIGURED

    if _CONFIGURED:
        return logging.getLogger(_LOGGER_NAME)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    base_logger = logging.getLogger(_LOGGER_NAME)
    base_logger.setLevel(logging.INFO)
    base_logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=1_048_576,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    base_logger.addHandler(console_handler)
    base_logger.addHandler(file_handler)

    _CONFIGURED = True
    return base_logger


def get_logger(name: str):
    configure_logging()
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
