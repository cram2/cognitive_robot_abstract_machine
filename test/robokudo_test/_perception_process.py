"""
RoboKudo perception processes as tests run them, and the passes tests put into their
world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from py_trees.behaviour import Behaviour
from py_trees.behaviours import Success
from py_trees.common import Status
from rclpy.node import Node

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.world_updating import WorldUpdatingAnnotator
from robokudo.cas import CAS
from robokudo.pipeline import Pipeline
from robokudo.process import RoboKudo
from robokudo.types.scene import ObjectHypothesis

from ._object_hypotheses import make_object_hypothesis
from ._perception_world import camera_pose, place_camera_of

# %% analysis engines


@dataclass(kw_only=True)
class EngineRunningABehaviour:
    """
    An analysis engine whose pipeline runs a single behaviour.
    """

    behaviour: Behaviour = field(default_factory=lambda: Success(name="Perceive"))
    """
    The behaviour the pipeline runs.
    """

    def name(self) -> str:
        return type(self).__name__

    def implementation(self) -> Pipeline:
        pipeline = Pipeline("RunningABehaviour")
        pipeline.add_child(self.behaviour)
        return pipeline


class FindsAnObjectInView(BaseAnnotator):
    """
    Finds one object in front of the camera and ends its part of the pass with
    ``outcome``.
    """

    def __init__(self, *, outcome: Status) -> None:
        super().__init__(name="FindsAnObjectInView")
        self.outcome = outcome

    def update(self) -> Status:
        cas = self.get_cas()
        place_camera_of(cas, self.world, camera_pose())
        cas.annotations.append(make_object_hypothesis())
        return self.outcome


@dataclass(kw_only=True)
class EngineFindingAnObject:
    """
    An analysis engine whose pipeline finds one object per pass and then puts it into
    the world.
    """

    outcome: Status = Status.SUCCESS
    """
    How finding the object ends.
    """

    def name(self) -> str:
        return type(self).__name__

    def implementation(self) -> Pipeline:
        pipeline = Pipeline("FindingAnObject")
        pipeline.add_children(
            [FindsAnObjectInView(outcome=self.outcome), WorldUpdatingAnnotator()]
        )
        return pipeline


# %% perception processes


def robokudo_of_its_own(
    node: Node, analysis_engine: AnalysisEngineInterface
) -> RoboKudo:
    """
    :return: A RoboKudo that looks for a shared world only once, finds none, and
        perceives into a world of its own.
    """
    return RoboKudo(
        node=node,
        analysis_engine=analysis_engine,
        world_discovery_timeout=timedelta(0),
    )


# %% finished passes


def put_pass_into_world(
    robokudo: RoboKudo,
    cas: CAS,
    *hypotheses: ObjectHypothesis,
    world_updating: WorldUpdatingAnnotator | None = None,
) -> Status:
    """
    Run ``world_updating`` on the world of ``robokudo`` in a pass of a pipeline that
    found ``hypotheses`` in ``cas``.

    :param world_updating: The annotator to run; one with its defaults if None.
    :return: The status the annotator ended with.
    """
    if world_updating is None:
        world_updating = WorldUpdatingAnnotator()
    pipeline = Pipeline("FinishedPass")
    pipeline.add_child(world_updating)
    pipeline.world = robokudo.world
    cas.annotations.extend(hypotheses)
    pipeline.cas = cas
    world_updating.tick_once()
    return world_updating.status
