"""
RoboKudo perception processes as tests run them, and the passes tests put into their
world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from py_trees.behaviour import Behaviour
from py_trees.behaviours import Success
from rclpy.node import Node

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.cas import CAS
from robokudo.pipeline import Pipeline
from robokudo.process import RoboKudo
from robokudo.types.scene import ObjectHypothesis

# %% analysis engines


@dataclass
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
    robokudo: RoboKudo, cas: CAS, *hypotheses: ObjectHypothesis
) -> None:
    """
    Update the world of ``robokudo`` after a pass of a pipeline that found
    ``hypotheses`` in ``cas``.
    """
    pipeline = Pipeline("FinishedPass")
    cas.annotations.extend(hypotheses)
    pipeline.cas = cas
    robokudo.update_world_with_perception_beliefs(pipeline=pipeline)
