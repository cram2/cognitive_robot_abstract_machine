import sys
from pathlib import Path

def _get_version():
    version_file = Path(__file__).resolve().parent.parent.parent.parent / "VERSION"
    with open(version_file) as f:
        return f.read().strip()

__version__ = _get_version()


import logging

logger = logging.getLogger("semantic_digital_twin")
logger.setLevel(logging.INFO)

from krrood.symbol_graph.symbol_graph import SymbolGraph

SymbolGraph()
