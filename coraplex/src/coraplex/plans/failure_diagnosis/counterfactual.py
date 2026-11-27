"""
Asks of a task that stopped approaching its goal whether it would have reached it had it
been free to move the degrees of freedom an earlier node left where they are, or had its
joints at a position limit been turned a full turn back from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from coraplex.plans.underspecified import ActionTrial, SubtreeRebuild
from cramph.node import StatechartNode
from giskardpy.motion_statechart.mixins import HasKinematicChainParameter
from giskardpy.motion_statechart.monitors.progress_monitors import BlockedPositionLimit
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain

# %% the answer of the counterfactual


class StallCounterfactual(StrEnum):
    """
    What trying a stalled task again, free to move what it only read or with its joints
    turned away from the position limits blocking it, showed.
    """

    REACHED = "reached its goal once free to move what an earlier node left"
    STILL_STALLED = "stalled again although free to move what an earlier node left"
    NOT_TRIED = "could not be tried free of what an earlier node left"
    REACHED_UNWOUND = (
        "reached its goal once its joints were turned a full turn back from the "
        "position limits blocking it"
    )
    STILL_STALLED_UNWOUND = (
        "stalled again although its joints were turned a full turn back from the "
        "position limits blocking it"
    )


# %% trying the task free of its inputs


@dataclass(kw_only=True)
class TaskWithChainFromWorldRoot(SubtreeRebuild):
    """
    Rebuilds a stalled task alone onto a copy of the world, with its kinematic chain
    reaching down to the root of that world, so it may move every degree of freedom
    between the root and its tip.
    """

    def build_for_trial(self, *, world: World) -> StatechartNode:
        """
        Differs from
        :meth:`~coraplex.plans.underspecified.SubtreeRebuild.build_for_trial` in letting
        the rebuilt task move its whole chain from the root of `world`.
        """
        freed = super().build_for_trial(world=world)
        freed.kinematic_chain = KinematicChain(
            root=world.root, tip=freed.kinematic_chain.tip
        )
        return freed

    def __str__(self) -> str:
        """
        :return: The stalled task and that it may move its whole chain.
        """
        return f"{self.root.name} free to move its whole chain from the world root"


def retry_task_with_chain_from_world_root(
    *, task: StatechartNode, trial: ActionTrial | None
) -> StallCounterfactual:
    """
    :param task: A task that stopped approaching its goal.
    :param trial: The trial to try it in, None if the plan has none.
    :return: Whether the task reaches its goal once free to move its whole chain.
    """
    if trial is None or not isinstance(task, HasKinematicChainParameter):
        return StallCounterfactual.NOT_TRIED
    if trial.succeeds(rebuild=TaskWithChainFromWorldRoot(root=task, candidates={})):
        return StallCounterfactual.REACHED
    return StallCounterfactual.STILL_STALLED


# %% trying the task with its joints unwound


@dataclass(kw_only=True)
class TaskWithJointsUnwound(SubtreeRebuild):
    """
    Rebuilds a stalled task alone onto a copy of the world, with each revolute joint
    that blocks it at a position limit turned a full turn back from that limit, which
    leaves the robot in the same geometric configuration.
    """

    blocked_position_limits: list[BlockedPositionLimit]
    """
    The position limits the joints are turned back from, each of a revolute joint whose
    range spans a full turn.
    """

    def build_for_trial(self, *, world: World) -> StatechartNode:
        """
        Differs from
        :meth:`~coraplex.plans.underspecified.SubtreeRebuild.build_for_trial` in first
        turning each joint of :attr:`blocked_position_limits` in `world` a full turn back
        from its limit.
        """
        with world.batch_state_changes():
            for blocked in self.blocked_position_limits:
                joint = self.joint_of(world=world, blocked=blocked)
                joint.position -= blocked.side * math.copysign(
                    math.tau, joint.multiplier
                )
        return super().build_for_trial(world=world)

    @staticmethod
    def joint_of(
        *, world: World, blocked: BlockedPositionLimit
    ) -> ScrewAxisConnection | None:
        """
        :param world: A world holding the degree of freedom of `blocked`.
        :param blocked: A position limit blocking a task.
        :return: The connection of `world` whose position is that degree of freedom,
            None if no screw axis connection has it.
        """
        return next(
            (
                connection
                for connection in world.get_connections_by_type(
                    connection_type=ScrewAxisConnection
                )
                if connection.raw_dof.id == blocked.degree_of_freedom.id
            ),
            None,
        )

    @classmethod
    def unwindable(
        cls, *, world: World, blocked_position_limits: list[BlockedPositionLimit]
    ) -> list[BlockedPositionLimit]:
        """
        :param world: The world the blocked degrees of freedom belong to.
        :param blocked_position_limits: Position limits blocking a task.
        :return: Those of a revolute joint whose range spans a full turn, so that the
            joint can be turned a full turn back from the limit.
        """
        unwindable = []
        for blocked in blocked_position_limits:
            joint = cls.joint_of(world=world, blocked=blocked)
            if joint is None or not joint.behaves_like_a_revolute_connection:
                continue
            limits = blocked.degree_of_freedom.limits
            if limits.upper.position - limits.lower.position >= math.tau / abs(
                joint.multiplier
            ):
                unwindable.append(blocked)
        return unwindable

    def __str__(self) -> str:
        """
        :return: The stalled task and that its joints were turned away from their
            limits.
        """
        return f"{self.root.name} with its joints turned a full turn back from their limits"


def retry_task_with_joints_unwound(
    *,
    task: StatechartNode,
    blocked_position_limits: list[BlockedPositionLimit],
    world: World,
    trial: ActionTrial | None,
) -> StallCounterfactual | None:
    """
    :param task: A task that stopped approaching its goal.
    :param blocked_position_limits: The position limits blocking `task`.
    :param world: The world `task` ran in.
    :param trial: The trial to try it in, None if the plan has none.
    :return: Whether the task reaches its goal once its revolute joints are turned a
        full turn back from the limits blocking them, None if it was not tried because
        the plan has no trial or no such joint can be turned a full turn.
    """
    unwindable = TaskWithJointsUnwound.unwindable(
        world=world, blocked_position_limits=blocked_position_limits
    )
    if trial is None or not unwindable:
        return None
    if trial.succeeds(
        rebuild=TaskWithJointsUnwound(
            root=task, candidates={}, blocked_position_limits=unwindable
        )
    ):
        return StallCounterfactual.REACHED_UNWOUND
    return StallCounterfactual.STILL_STALLED_UNWOUND
