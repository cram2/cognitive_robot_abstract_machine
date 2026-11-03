from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World

from ..statechart_json_round_trip import StatechartSentThroughJSON


def create_context_with_motion_control(
    world: World, qp_controller_config: QPControllerConfig | None = None
) -> StatechartContext:
    """
    Create a context in which motion statechart nodes can be built outside of an
    executor.

    :param world: The world the nodes act in.
    :param qp_controller_config: The controller configuration, the simulation defaults
        of :class:`~giskardpy.motion_control.MotionControl` if None.
    :return: A context extended the way :class:`~giskardpy.motion_control.MotionControl`
        extends the context of its executor.
    """
    context = StatechartContext(world=world)
    motion_control = (
        MotionControl()
        if qp_controller_config is None
        else MotionControl(qp_controller_config=qp_controller_config)
    )
    motion_control.extend_context(context)
    return context


def add_to_statechart_with_motion_control(
    node: StatechartNode,
    world: World,
    qp_controller_config: QPControllerConfig | None = None,
) -> Statechart:
    """
    Put `node` into a statechart of its own, so it can be built outside of an executor.

    :param node: The node to build.
    :param world: The world the node acts in.
    :param qp_controller_config: The controller configuration, the simulation defaults
        of :class:`~giskardpy.motion_control.MotionControl` if None.
    :return: The statechart holding `node`, in a context made by
        :func:`create_context_with_motion_control`.
    """
    statechart = Statechart(
        context=create_context_with_motion_control(world, qp_controller_config)
    )
    statechart.add_node(node)
    return statechart


# %% where the statechart an executor runs comes from


class StatechartOrigin(ABC):
    """
    Where the statechart an executor runs was built.
    """

    @abstractmethod
    def create_building_context(
        self, executor: StatechartExecutor
    ) -> StatechartContext:
        """
        :param executor: The executor that will run the statechart.
        :return: The context to build the statechart in.
        """

    @abstractmethod
    def hand_over(
        self, statechart: Statechart, executor: StatechartExecutor
    ) -> Statechart:
        """
        :param statechart: The statechart built in the context of
            :meth:`create_building_context`.
        :param executor: The executor that will run the statechart.
        :return: The statechart `executor` compiles.
        """


@dataclass
class BuiltForTheExecutor(StatechartOrigin):
    """
    The statechart is built in the context of the executor running it.
    """

    def create_building_context(
        self, executor: StatechartExecutor
    ) -> StatechartContext:
        return executor.context

    def hand_over(
        self, statechart: Statechart, executor: StatechartExecutor
    ) -> Statechart:
        return statechart


@dataclass
class SentThroughJSON(StatechartOrigin):
    """
    The statechart is built in a context of its own and reaches the executor as JSON.
    """

    def create_building_context(
        self, executor: StatechartExecutor
    ) -> StatechartContext:
        return create_context_with_motion_control(executor.context.world)

    def hand_over(
        self, statechart: Statechart, executor: StatechartExecutor
    ) -> Statechart:
        """
        .. note:: The sender releases what it registered in the world it shares with
            `executor` once its statechart is sent.
        """
        round_trip = StatechartSentThroughJSON.received_in(statechart, executor.context)
        statechart.context.cleanup()
        return round_trip.received
