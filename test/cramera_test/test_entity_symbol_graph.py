"""
Tests that queryable knowledge entities live in krrood's SymbolGraph.
"""

import pytest

krrood = pytest.importorskip("krrood", reason="the SymbolGraph requires krrood")

from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph  # noqa: E402

from cramera.knowledge.entity import NamedEntity  # noqa: E402
from cramera.knowledge.entities import Robot  # noqa: E402


# %% entities are symbols
class TestEntitiesAreSymbols:
    """
    Every queryable entity is a :class:`Symbol`, so its instances are tracked in the
    SymbolGraph and can take part in symbol-level reasoning.
    """

    def test_every_queryable_entity_is_a_symbol(self):
        assert issubclass(NamedEntity, Symbol)

    def test_a_new_entity_is_tracked_by_the_symbol_graph(self):
        robot = Robot(name="recorded_robot", arm_count=1)

        instances = list(SymbolGraph().get_instances_of_type(Robot))

        assert instances == [robot]
        assert instances[0] is robot
