from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any

from cramph.node import NodeRebuild, StatechartNode

# %% mimics


@dataclass(kw_only=True)
class RebuildLeavingNodesOut(NodeRebuild):
    """
    Builds every node anew except those it leaves out, and keeps every other value as it
    is.
    """

    left_out: list[StatechartNode] = field(default_factory=list)
    """
    The nodes it leaves out.
    """

    built: dict[StatechartNode, StatechartNode | None] = field(default_factory=dict)
    """
    The node each node was built anew as, None for one left out.
    """

    def rebuilt(self, *, node: StatechartNode) -> StatechartNode | None:
        if node not in self.built:
            self.built[node] = (
                None if node in self.left_out else node.rebuilt_by(rebuild=self)
            )
        return self.built[node]

    def rebuilt_children(
        self, *, owner: StatechartNode, children: list[StatechartNode]
    ) -> list[StatechartNode]:
        return [
            rebuilt
            for child in children
            if (rebuilt := self.rebuilt(node=child)) is not None
        ]

    def rebound(self, *, value: Any) -> Any:
        return value
