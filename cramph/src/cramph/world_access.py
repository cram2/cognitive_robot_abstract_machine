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

    moved_degrees_of_freedom: frozenset[DegreeOfFreedom] = frozenset()
    """
    The degrees of freedom the node moves somewhere.
    """

    held_degrees_of_freedom: frozenset[DegreeOfFreedom] = frozenset()
    """
    The degrees of freedom the node keeps where they were when it started.
    """

    read_degrees_of_freedom: frozenset[DegreeOfFreedom] = frozenset()
    """
    The degrees of freedom whose state decides what the node observes or aims for.
    """

    reparented_frames: frozenset[KinematicStructureEntity] = frozenset()
    """
    The frames the node attaches to a new parent.
    """

    read_frames: frozenset[KinematicStructureEntity] = frozenset()
    """
    The frames whose place in the kinematic structure decides what the node aims for.
    """

    @property
    def moved_or_held_degrees_of_freedom(self) -> frozenset[DegreeOfFreedom]:
        """
        :return: The degrees of freedom the node moves or holds.
        """
        return self.moved_degrees_of_freedom | self.held_degrees_of_freedom

    @staticmethod
    def degrees_of_freedom_in_expression(
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

    def union_with(self, *, other: WorldAccess) -> WorldAccess:
        """
        :param other: More of what the same node does to the world.
        :return: An access holding what both hold.
        """
        return WorldAccess(
            moved_degrees_of_freedom=self.moved_degrees_of_freedom
            | other.moved_degrees_of_freedom,
            held_degrees_of_freedom=self.held_degrees_of_freedom
            | other.held_degrees_of_freedom,
            read_degrees_of_freedom=self.read_degrees_of_freedom
            | other.read_degrees_of_freedom,
            reparented_frames=self.reparented_frames | other.reparented_frames,
            read_frames=self.read_frames | other.read_frames,
        )

    def treat_moved_as_held(self) -> WorldAccess:
        """
        :return: This access, with what it moves counted as held instead.
        """
        return WorldAccess(
            held_degrees_of_freedom=self.held_degrees_of_freedom
            | self.moved_degrees_of_freedom,
            read_degrees_of_freedom=self.read_degrees_of_freedom,
            reparented_frames=self.reparented_frames,
            read_frames=self.read_frames,
        )
