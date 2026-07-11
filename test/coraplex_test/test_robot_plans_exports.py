"""
Regression tests for ``coraplex.robot_plans``' package-level re-exports.

``coraplex.plans.plan_node`` type-checks against ``coraplex.robot_plans.ActionDescription`` and
``coraplex.robot_plans.BaseMotion`` (``from coraplex.robot_plans import ActionDescription,
BaseMotion``, guarded by ``TYPE_CHECKING``). ``krrood``'s class-diagram introspection resolves that
annotation by walking the same import path at runtime, so both names must actually be reachable as
package-level attributes, not just importable from their defining submodule.
"""

from krrood.class_diagrams.class_diagram import ClassDiagram

import coraplex.robot_plans as robot_plans
from coraplex.plans.plan_node import UnderspecifiedNode


def test_action_description_and_base_motion_are_package_level_exports():
    assert hasattr(robot_plans, "ActionDescription")
    assert hasattr(robot_plans, "BaseMotion")


def test_class_diagram_resolves_underspecified_node():
    """UnderspecifiedNode's ``_action_iterator: Optional[Iterator[ActionDescription]]`` annotation
    must resolve -- previously raised CouldNotResolveType because ``coraplex.robot_plans.actions``
    re-exported nothing, so ``ActionDescription`` was unreachable from ``coraplex.robot_plans``."""
    ClassDiagram(classes=[UnderspecifiedNode])
