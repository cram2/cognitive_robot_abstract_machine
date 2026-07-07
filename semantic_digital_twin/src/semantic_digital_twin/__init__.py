import logging
from krrood.symbol_graph.symbol_graph import SymbolGraph

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

logger = logging.getLogger("semantic_digital_twin")
logger.setLevel(logging.INFO)

SymbolGraph()
