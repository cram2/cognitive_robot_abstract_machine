"""
Frameworks sharing the statecharts of the cramph tests, and nodes each of them ticks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from cramph.composites import Sequence
from cramph.nodes_for_testing import (
    CompositeNodeWithChainedChildren,
    NodeObservingTheOppositeOfAnObservationPredicate,
    NodeSucceedingOnObservingTrue,
)
from krrood.patterns.classproperty import classproperty

# %% frameworks


class Framework(StrEnum):
    """
    The frameworks sharing the statecharts of the cramph tests.
    """

    PERCEPTION = "perception"
    """
    Ticks the nodes that look at the scene.
    """

    MOTION = "motion"
    """
    Ticks the nodes that move the robot.
    """

    PLANNING = "planning"
    """
    Decides which nodes run, and sends the statechart to the others.
    """


# %% nodes of the perception framework


@dataclass(eq=False, repr=False, kw_only=True)
class StepThePerceptionTicks(NodeSucceedingOnObservingTrue):
    """
    A node that succeeds once it observes True, ticked by the perception framework.
    """

    @classproperty
    def ticking_framework(cls) -> str:
        """
        :return: The perception framework, see
            :attr:`~cramph.node.StatechartNode.ticking_framework`.
        """
        return Framework.PERCEPTION


@dataclass(eq=False, repr=False, kw_only=True)
class SequenceThePerceptionTicks(Sequence):
    """
    A sequence ticked by the perception framework.
    """

    @classproperty
    def ticking_framework(cls) -> str:
        """
        :return: The perception framework, see
            :attr:`~cramph.node.StatechartNode.ticking_framework`.
        """
        return Framework.PERCEPTION


@dataclass(eq=False, repr=False, kw_only=True)
class ChainThePerceptionTicks(CompositeNodeWithChainedChildren):
    """
    A composite node chaining its children, ticked by the perception framework.
    """

    @classproperty
    def ticking_framework(cls) -> str:
        """
        :return: The perception framework, see
            :attr:`~cramph.node.StatechartNode.ticking_framework`.
        """
        return Framework.PERCEPTION


# %% nodes of the motion framework


@dataclass(eq=False, repr=False, kw_only=True)
class StepTheMotionTicks(NodeSucceedingOnObservingTrue):
    """
    A node that succeeds once it observes True, ticked by the motion framework.
    """

    @classproperty
    def ticking_framework(cls) -> str:
        """
        :return: The motion framework, see
            :attr:`~cramph.node.StatechartNode.ticking_framework`.
        """
        return Framework.MOTION


@dataclass(eq=False, repr=False, kw_only=True)
class WatcherTheMotionTicks(NodeObservingTheOppositeOfAnObservationPredicate):
    """
    A node observing the opposite of the node it watches, ticked by the motion
    framework.
    """

    @classproperty
    def ticking_framework(cls) -> str:
        """
        :return: The motion framework, see
            :attr:`~cramph.node.StatechartNode.ticking_framework`.
        """
        return Framework.MOTION
