"""Root conftest for llmr tests — krrood-aligned test fixtures.

Autouse cleanup: SymbolGraph is real, singleton, and cleared between tests.
Pattern: SymbolGraph() to ensure singleton exists, yield, clear() after test.
"""
from __future__ import annotations

import pytest
from krrood.symbol_graph.symbol_graph import SymbolGraph


@pytest.fixture(autouse=True)
def cleanup_after_test() -> None:
    """Ensure SymbolGraph exists, yield for test, clear after.

    Krrood pattern: real singleton, no mocking.
    """
    SymbolGraph()  # ensure singleton exists before test
    yield
    SymbolGraph().clear()  # wipe state after every test
