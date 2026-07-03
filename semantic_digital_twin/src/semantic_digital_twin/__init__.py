import logging

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

logger = logging.getLogger("semantic_digital_twin")
logger.setLevel(logging.INFO)


def _init_symbol_graph():
    from krrood.symbol_graph.symbol_graph import SymbolGraph
    SymbolGraph()


try:
    _init_symbol_graph()
except ModuleNotFoundError:
    pass
