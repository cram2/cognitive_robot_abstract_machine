"""Root conftest for llmr tests — krrood-aligned test fixtures.


"""
from __future__ import annotations

import pytest
from krrood.class_diagrams import ClassDiagram
from krrood.ontomatic.property_descriptor.attribute_introspector import (
    DescriptorAwareIntrospector,
)
from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph
from krrood.utils import recursive_subclasses


@pytest.fixture(autouse=True)
def cleanup_after_test() -> None:
    """Rebuild SymbolGraph from scratch before each test, tear down after.

    Ensures no stale wrapped-class metadata leaks between tests.
    Krrood pattern: real singleton, no mocking.
    """
    SymbolGraph.clear()
    class_diagram = ClassDiagram(
        recursive_subclasses(Symbol),
        introspector=DescriptorAwareIntrospector(),
    )
    SymbolGraph(_class_diagram=class_diagram)
    yield
    SymbolGraph.clear()
    class_diagram.clear()
