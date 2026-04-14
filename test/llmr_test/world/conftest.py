"""Conftest for world tests — provides simple world fixtures."""
from __future__ import annotations

from typing_extensions import Dict, Any
import pytest


@pytest.fixture
def simple_world() -> Dict[str, Any]:
    """Return a dict of mock world objects for grounding tests.

    Uses SimpleNamespace to simulate body objects with .name attributes.
    """
    from types import SimpleNamespace
    milk = SimpleNamespace(name="milk")
    table = SimpleNamespace(name="table")
    fridge = SimpleNamespace(name="fridge")
    return {"milk": milk, "table": table, "fridge": fridge}
