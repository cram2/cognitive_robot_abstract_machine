from dataclasses import dataclass

from krrood.entity_query_language.predicate import Symbol
from krrood.patterns.role import Role
from krrood.symbol_graph.symbol_graph import SymbolGraph


def test_role_registers_when_class_defined_after_symbol_graph_built():
    """
    A role class defined after the SymbolGraph class diagram was first built must still
    register.

    The class diagram is built lazily on first use. A role class created afterwards used
    to be missing from it, so its registration was silently skipped and the role
    registry returned nothing. Registration must add the late-defined class to the
    diagram on demand.
    """

    @dataclass(kw_only=True, eq=False)
    class PersistentEntityDefinedBeforeGraph(Symbol):
        name: str

        def __hash__(self) -> int:
            return hash(self.name)

    entity = PersistentEntityDefinedBeforeGraph(name="entity")

    # Ensure the class diagram has been built before the role class below is defined.
    assert SymbolGraph().class_diagram is not None

    @dataclass(kw_only=True, eq=False)
    class RoleDefinedAfterGraph(Role[PersistentEntityDefinedBeforeGraph]): ...

    role = RoleDefinedAfterGraph(role_taker=entity)

    assert Role.has_role(role_taker=entity, role_types=RoleDefinedAfterGraph)
    assert role in Role.roles_for(role_taker=entity, role_type=RoleDefinedAfterGraph)
