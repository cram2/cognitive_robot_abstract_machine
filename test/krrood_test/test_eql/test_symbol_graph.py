import os

import pytest

from krrood.entity_query_language.entity import entity, variable
from krrood.entity_query_language.entity_result_processors import an
from krrood.entity_query_language.symbol_graph import SymbolGraph
from ..dataset.example_classes import Position

try:
    import pydot
    import pygraphviz
except ImportError:
    pydot = None
    pygraphviz = None


@pytest.mark.skipif(
    not (pydot and pygraphviz), reason="pydot and graphviz not installed"
)
def test_visualize_symbol_graph():
    SymbolGraph().clear()
    symbol_graph = SymbolGraph()
    symbol_graph.to_dot("symbol_graph.svg", format_="svg", graph_type="type")
    assert len(symbol_graph._class_diagram.wrapped_classes) >= 59
    if os.path.exists("symbol_graph.svg"):
        os.remove("symbol_graph.svg")


def test_memory_leak():
    """
    Test if the SymbolGraph does not artificially keep objects alive that would be garbage collected.
    """

    def create_data():
        point = Position(1, 2, 3)
        return point

    create_data()

    q = an(entity(variable(Position, domain=None)))
    result = list(q.evaluate())

    assert result == []

    assert len(SymbolGraph().wrapped_instances) == 0
