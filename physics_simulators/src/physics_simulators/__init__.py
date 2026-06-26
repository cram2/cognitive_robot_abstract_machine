import sys
from pathlib import Path

def _get_version():
    version_file = Path(__file__).resolve().parent.parent.parent.parent / "VERSION"
    with open(version_file) as f:
        return f.read().strip()

__version__ = _get_version()