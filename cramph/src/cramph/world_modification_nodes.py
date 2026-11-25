"""
Nodes that change the kinematic structure of the world.
"""

from __future__ import annotations

from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from cramph.node import SucceedsOnObservingTrue
from cramph.node import NodeArtifacts, StatechartNode
from cramph.world_access import WorldAccess
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% changing the kinematic structure


@dataclass(eq=False, repr=False, kw_only=True)
class MoveBranch(SucceedsOnObservingTrue, StatechartNode):
    """
    Moves an entity, with everything below it, under a new parent when it starts, see
    :meth:`~semantic_digital_twin.world.World.move_branch`, and succeeds once it did.

    The statechart then builds every node again, so that nodes reading the moved branch
    follow its new parent.
    """

    branch_root: KinematicStructureEntity
    """
    The root of the branch that is moved.
    """

    new_parent: KinematicStructureEntity
    """
    The entity the branch is moved under.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())

    def on_start(self) -> None:
        self.context.world.move_branch(
            branch_root=self.branch_root, new_parent=self.new_parent
        )

    def compute_world_access(self) -> WorldAccess:
        """
        :return: What :meth:`~cramph.node.StatechartNode.compute_world_access`
            returns, with :attr:`branch_root` attached to a new parent.
        """
        return (
            super()
            .compute_world_access()
            .union_with(
                other=WorldAccess(reparented_frames=frozenset({self.branch_root}))
            )
        )
