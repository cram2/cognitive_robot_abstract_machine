import pytest

from krrood.entity_query_language.symbol_graph import SymbolGraph

@pytest.fixture(autouse=True, scope="function")
def cleanup_after_test():
    # runs BEFORE each test
    yield
    # runs AFTER each test (even if the test fails or errors)
    SymbolGraph().clear()