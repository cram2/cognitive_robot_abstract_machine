"""
Monitors observing how close the collision groups of a robot come to other objects or to
each other.
"""

from __future__ import annotations

from dataclasses import dataclass

from cramph.context import ContextExtension
from cramph.node import EndedByOwner
from giskardpy.motion_statechart.context import MotionControlContext
from giskardpy.motion_statechart.graph_node import (
    MotionNodeArtifacts,
    MotionStatechartNode,
)
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.collision_checking.collision_variable_managers import (
    AvoidedExternalContact,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% distance to the environment


@dataclass(eq=False, repr=False, kw_only=True)
class ExternalCollisionDistanceMonitor(EndedByOwner, MotionStatechartNode):
    """
    Monitors the distance to the closest external object for a specific collision group
    of a body. Turns True if the distance falls below a given threshold.

    .. note:: the input bodies are only used to look up the collision groups.
    """

    body: Body
    """
    The robot body to monitor.
    """

    threshold: float
    """
    Distance threshold in meters.
    """

    collision_index: int = 0
    """Index of the closest collision (0 = closest, 1 = second closest, etc.)."""

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        """
        Differs from :meth:`~cramph.node.StatechartNode.required_context_extensions` in
        also requiring the
        :class:`~giskardpy.motion_statechart.context.MotionControlContext` that holds
        the collision managers.
        """
        return super().required_context_extensions + (MotionControlContext,)

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        :return: An observation that is True while the :attr:`collision_index`-th
            closest external contact of the group of :attr:`body` is closer than
            :attr:`threshold`.
        """
        manager = self.context.require_extension(
            MotionControlContext
        ).get_or_create_external_collision_manager(self.context)

        manager.register_group_of_body(self.body)
        group = manager.get_collision_group(self.body)

        distance_symbol = manager.get_contact_distance_symbol(
            AvoidedExternalContact(group=group, rank=self.collision_index)
        )

        return MotionNodeArtifacts(observation=distance_symbol < self.threshold)


# %% distance between two parts of the robot


@dataclass(eq=False, repr=False, kw_only=True)
class SelfCollisionDistanceMonitor(EndedByOwner, MotionStatechartNode):
    """
    Monitors the distance between the collision groups of two robot bodies.

    Turns True if the distance falls below a given threshold.

    .. note:: the input bodies are only used to look up the collision groups.
    """

    body_a: Body
    """
    First robot body to monitor.
    """

    body_b: Body
    """
    Second robot body to monitor.
    """

    threshold: float
    """
    Distance threshold in meters.
    """

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        """
        Differs from :meth:`~cramph.node.StatechartNode.required_context_extensions` in
        also requiring the
        :class:`~giskardpy.motion_statechart.context.MotionControlContext` that holds
        the collision managers.
        """
        return super().required_context_extensions + (MotionControlContext,)

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        :return: An observation that is True while the closest contact between the
            groups of :attr:`body_a` and :attr:`body_b` is closer than
            :attr:`threshold`.
        """
        manager = self.context.require_extension(
            MotionControlContext
        ).get_or_create_self_collision_manager(self.context)

        manager.register_groups_of_body_combination(self.body_a, self.body_b)
        combination = manager.body_pair_to_group_pair(self.body_a, self.body_b)

        # The contacts of a combination are ranked, so the first is the closest.
        distance_symbol = manager.get_contact_distance_symbol(
            manager.avoided_contacts_of(combination)[0]
        )

        return MotionNodeArtifacts(observation=distance_symbol < self.threshold)
