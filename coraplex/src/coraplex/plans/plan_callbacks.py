from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from coraplex.plans.plan_entity import PlanEntity

if TYPE_CHECKING:
    from coraplex.plans.plan_node import PlanNode


@dataclass
class PlanCallback(PlanEntity):
    """
    Observe plan execution; unimplemented events leave execution unchanged.
    """

    def on_start(self, node: PlanNode) -> None:
        """
        Observe a node whose execution has begun.

        :param node: The started node.
        """

    def on_end(self, node: PlanNode) -> None:
        """
        Observe a node after its execution ends.

        :param node: The completed node, including its outcome.
        """
