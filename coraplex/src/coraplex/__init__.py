import sys
import logging
from pathlib import Path

def _get_version():
    version_file = Path(__file__).resolve().parents[3] / "VERSION"
    with open(version_file) as f:
        return f.read().strip()

__version__ = _get_version()

format = "%(levelname)s:%(filename)s::%(lineno)s %(funcName)s %(message)s"
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
