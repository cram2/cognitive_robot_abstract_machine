"""Conftest for reasoning tests."""
from __future__ import annotations

import pytest
from llmr.pycram_bridge.introspector import PycramIntrospector


@pytest.fixture
def introspector() -> PycramIntrospector:
    """Return a fresh PycramIntrospector instance."""
    return PycramIntrospector()
