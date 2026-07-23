import logging
from krrood.symbol_graph.symbol_graph import SymbolGraph

from ._version import __version__

logger = logging.getLogger("semantic_digital_twin")
logger.setLevel(logging.INFO)

SymbolGraph()
