import sys
from pathlib import Path
import logging
from krrood.symbol_graph.symbol_graph import SymbolGraph

def _get_version():
    version_file = Path(__file__).resolve().parents[3] / "VERSION"
    with open(version_file) as f:
        return f.read().strip()

__version__ = _get_version()

logger = logging.getLogger("semantic_digital_twin")
logger.setLevel(logging.INFO)

SymbolGraph()
