__version__ = "0.0.6"


import logging

logger = logging.getLogger("semantic_digital_twin")
logger.setLevel(logging.INFO)

from krrood.entity_query_language.symbol_graph import SymbolGraph

SymbolGraph()
