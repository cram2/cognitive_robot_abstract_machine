"""
Asks of a task that stopped approaching its goal whether it would have reached it had it
been free to move the degrees of freedom an earlier node left where they are, or had its
joints at a position limit been turned a full turn back from it or put back where they
were before an earlier node moved them.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from coraplex.plans.underspecified import ActionTrial, SubtreeRebuild
from cramph.causal_edges import WorldChanges
from cramph.node import StatechartNode
from cramph.run_start_positions import RunStartPositions
from cramph.state_history import NodeRun
from giskardpy.motion_statechart.mixins import HasKinematicChainParameter
from giskardpy.motion_statechart.monitors.progress_monitors import BlockedPositionLimit
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain

# %% the answer of the counterfactual


class StallCounterfactual(StrEnum):
    """
    What trying a stalled task again, free to move what it only read or with its joints
    moved away from the position limits blocking it, showed.
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
    REACHED_FROM_EARLIER_POSITIONS = (
        "reached its goal once its joints at a position limit were put back where they "
        "were before an earlier node moved them"
    )
    STILL_STALLED_FROM_EARLIER_POSITIONS = (
        "stalled again although its joints at a position limit were put back where "
        "they were before an earlier node moved them"
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
    Rebuilds a stalled task, with the tasks running beside it, onto a copy of the world,
    with each revolute joint that blocks it at a position limit turned a full turn back
    from that limit, which leaves the robot in the same geometric configuration.
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
                joint = self.find_blocked_joint(world=world, blocked=blocked)
                joint.position -= blocked.side * math.copysign(
                    math.tau, joint.multiplier
                )
        return super().build_for_trial(world=world)

    @staticmethod
    def find_blocked_joint(
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
            joint = cls.find_blocked_joint(world=world, blocked=blocked)
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


# %% trying the task with its joints where they were before an earlier node moved them


@dataclass(kw_only=True)
class TaskFromEarlierPositions(SubtreeRebuild):
    """
    Rebuilds a stalled task, with the tasks running beside it, onto a copy of the world,
    with the joints that block it at a position limit put back where they were before an
    earlier node moved them.
    """

    earlier_positions: dict[DegreeOfFreedom, float]
    """
    Each blocked degree of freedom, and its position before the earlier node moved it.
    """

    def build_for_trial(self, *, world: World) -> StatechartNode:
        """
        Differs from
        :meth:`~coraplex.plans.underspecified.SubtreeRebuild.build_for_trial` in first
        putting each degree of freedom of :attr:`earlier_positions` in `world` back at
        its earlier position.
        """
        for degree_of_freedom, position in self.earlier_positions.items():
            world.state[degree_of_freedom.id].position = position
        world.notify_state_change()
        return super().build_for_trial(world=world)

    def __str__(self) -> str:
        """
        :return: The stalled task and that its joints were put back.
        """
        return f"{self.root.name} with its blocked joints where they were before"


# %% the ways to retry a task stalled at a position limit


@dataclass(kw_only=True)
class PositionLimitRetry(ABC):
    """
    One way to try a task stalled at position limits again with the joints that block it
    moved, which shows whether where those joints were made it stall.
    """

    @property
    @abstractmethod
    def reached(self) -> StallCounterfactual:
        """
        :return: What the retry showed when the task reached its goal.
        """

    @property
    @abstractmethod
    def still_stalled(self) -> StallCounterfactual:
        """
        :return: What the retry showed when the task stalled again.
        """

    @abstractmethod
    def rebuild_for(
        self,
        *,
        stalled: NodeRun,
        blocked_position_limits: list[BlockedPositionLimit],
    ) -> SubtreeRebuild | None:
        """
        :param stalled: The run that stopped approaching its goal.
        :param blocked_position_limits: The position limits blocking it.
        :return: The rebuild of its task, with the tasks running beside it and the
            blocking joints moved; None if this retry can move none of them.
        """

    def try_in(
        self,
        *,
        trial: ActionTrial | None,
        stalled: NodeRun,
        blocked_position_limits: list[BlockedPositionLimit],
    ) -> StallCounterfactual | None:
        """
        :param trial: The trial to try the task in, None if the plan has none.
        :param stalled: The run that stopped approaching its goal.
        :param blocked_position_limits: The position limits blocking it.
        :return: Whether the task reached its goal once retried, None if it was not
            tried because the plan has no trial or this retry can move no blocking
            joint.
        """
        if trial is None:
            return None
        rebuild = self.rebuild_for(
            stalled=stalled, blocked_position_limits=blocked_position_limits
        )
        if rebuild is None:
            return None
        if trial.succeeds(rebuild=rebuild):
            return self.reached
        return self.still_stalled


@dataclass(kw_only=True)
class RetryWithJointsUnwound(PositionLimitRetry):
    """
    Retries the task with each blocking revolute joint whose range spans a full turn
    turned a full turn back from its limit.
    """

    world: World
    """
    The world the task ran in.
    """

    @property
    def reached(self) -> StallCounterfactual:
        return StallCounterfactual.REACHED_UNWOUND

    @property
    def still_stalled(self) -> StallCounterfactual:
        return StallCounterfactual.STILL_STALLED_UNWOUND

    def rebuild_for(
        self,
        *,
        stalled: NodeRun,
        blocked_position_limits: list[BlockedPositionLimit],
    ) -> SubtreeRebuild | None:
        unwindable = TaskWithJointsUnwound.unwindable(
            world=self.world, blocked_position_limits=blocked_position_limits
        )
        if not unwindable:
            return None
        return TaskWithJointsUnwound(
            root=SubtreeRebuild.root_around(node=stalled.node),
            candidates={},
            blocked_position_limits=unwindable,
        )


@dataclass(kw_only=True)
class RetryFromEarlierPositions(PositionLimitRetry):
    """
    Retries the task with each blocking joint put back where it was when the last
    earlier run that moved it started.
    """

    world_changes: WorldChanges
    """
    What each run of the statechart changed in the world.
    """

    run_start_positions: RunStartPositions
    """
    Where each run of the statechart found the degrees of freedom it moves.
    """

    @property
    def reached(self) -> StallCounterfactual:
        return StallCounterfactual.REACHED_FROM_EARLIER_POSITIONS

    @property
    def still_stalled(self) -> StallCounterfactual:
        return StallCounterfactual.STILL_STALLED_FROM_EARLIER_POSITIONS

    def rebuild_for(
        self,
        *,
        stalled: NodeRun,
        blocked_position_limits: list[BlockedPositionLimit],
    ) -> SubtreeRebuild | None:
        earlier_positions = {}
        for limit in blocked_position_limits:
            mover = self.world_changes.last_run_that_moved(
                degree_of_freedom_id=limit.degree_of_freedom.id, reader=stalled
            )
            if mover is None:
                continue
            position = self.run_start_positions.position_at_start_of(
                run=mover, degree_of_freedom=limit.degree_of_freedom
            )
            if position is not None:
                earlier_positions[limit.degree_of_freedom] = position
        if not earlier_positions:
            return None
        return TaskFromEarlierPositions(
            root=SubtreeRebuild.root_around(node=stalled.node),
            candidates={},
            earlier_positions=earlier_positions,
        )
