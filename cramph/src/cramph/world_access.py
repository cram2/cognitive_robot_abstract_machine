"""
What a statechart node does to the world, named by the degrees of freedom and frames it
changes and reads rather than by their values.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomVariable,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% the parts of the world a node touches


@dataclass(frozen=True, kw_only=True)
class WorldAccess:
    """
    The degrees of freedom and frames a node changes, holds and reads.
    """

    moved: frozenset[DegreeOfFreedom] = frozenset()
    """
    The degrees of freedom the node moves somewhere.
    """

    held: frozenset[DegreeOfFreedom] = frozenset()
    """
    The degrees of freedom the node keeps where they were when it started.
    """

    read: frozenset[DegreeOfFreedom] = frozenset()
    """
    The degrees of freedom whose state decides what the node observes or aims for.
    """

    reattached: frozenset[KinematicStructureEntity] = frozenset()
    """
    The frames the node attaches to a new parent.
    """

    read_frames: frozenset[KinematicStructureEntity] = frozenset()
    """
    The frames whose place in the kinematic structure decides what the node aims for.
    """

    @property
    def driven(self) -> frozenset[DegreeOfFreedom]:
        """
        :return: The degrees of freedom the node moves or holds.
        """
        return self.moved | self.held

    @staticmethod
    def degrees_of_freedom_of(
        *, expression: Scalar | None
    ) -> frozenset[DegreeOfFreedom]:
        """
        :param expression: A symbolic expression, or None.
        :return: The degrees of freedom whose variables `expression` depends on.
        """
        if expression is None:
            return frozenset()
        return frozenset(
            variable.dof
            for variable in expression.free_variables()
            if isinstance(variable, DegreeOfFreedomVariable)
        )

    def merged_with(self, *, other: WorldAccess) -> WorldAccess:
        """
        :param other: More of what the same node does to the world.
        :return: An access holding what both hold.
        """
        return WorldAccess(
            moved=self.moved | other.moved,
            held=self.held | other.held,
            read=self.read | other.read,
            reattached=self.reattached | other.reattached,
            read_frames=self.read_frames | other.read_frames,
        )

    def count_moved_as_held(self) -> WorldAccess:
        """
        :return: This access, with what it moves counted as held instead.
        """
        return WorldAccess(
            held=self.held | self.moved,
            read=self.read,
            reattached=self.reattached,
            read_frames=self.read_frames,
        )
