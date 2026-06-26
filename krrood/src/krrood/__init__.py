import importlib.metadata
import logging
from pathlib import Path

def _get_version():
    version_file = Path(__file__).resolve().parent.parent.parent.parent / "VERSION"
    with open(version_file) as f:
        return f.read().strip()

__version__ = _get_version()


logger = logging.getLogger("krrood")
logger.setLevel(logging.INFO)
