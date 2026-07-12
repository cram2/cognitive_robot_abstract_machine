"""
Regression test for ``UnderspecifiedNode``'s ``ActionDescription``/``BaseMotion`` type hints
resolving under ``krrood``'s class-diagram introspection.
"""

from krrood.class_diagrams.class_diagram import ClassDiagram

from coraplex.plans.plan_node import UnderspecifiedNode


def test_class_diagram_resolves_underspecified_node():
    """``UnderspecifiedNode``'s ``_action_iterator: Optional[Iterator[ActionDescription]]``
    annotation must resolve -- previously raised ``CouldNotResolveType`` because
    ``ActionDescription`` was imported under ``TYPE_CHECKING`` from the ``coraplex.robot_plans``
    package, which never re-exported it (and ``coraplex``'s ``__init__.py`` files stay import-free,
    so the fix imports directly from the defining submodule instead of adding a package re-export)."""
    ClassDiagram(classes=[UnderspecifiedNode])
