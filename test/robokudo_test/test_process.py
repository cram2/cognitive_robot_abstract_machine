from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from threading import Thread

import numpy as np
import pytest
import rclpy
from py_trees.behaviour import Behaviour
from py_trees.blackboard import Blackboard
from py_trees.behaviours import Failure, Running, Success
from py_trees.common import Status
from rclpy.context import Context
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import IsPerceivable
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cup,
    PerceivedObject,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CAS
from robokudo.exceptions import (
    EngineRootIsNotAPipelineError,
    RoboKudoAlreadyRunningError,
)
from robokudo.identifier import BBIdentifier
from robokudo.pipeline import Pipeline
from robokudo.process import RoboKudo
from robokudo.types.scene import ObjectHypothesis

from ._object_hypotheses import make_object_hypothesis
from ._perception_world import (
    SharedWorld,
    camera_pose,
    cas_seen_from,
    place_camera_of,
    wait_until,
)


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


@dataclass
class EngineWithoutPipeline:
    """
    An analysis engine whose tree is a plain behaviour rather than a pipeline.
    """

    tree: Behaviour = field(default_factory=lambda: Success(name="Perceive"))
    """
    The tree the engine builds.
    """

    def name(self) -> str:
        return type(self).__name__

    def implementation(self) -> Behaviour:
        return self.tree


class FindsAnObjectInView(BaseAnnotator):
    """
    Finds one object in front of the camera and ends the pass with ``outcome``.
    """

    def __init__(self, outcome: Status) -> None:
        super().__init__(name="FindsAnObjectInView")
        self.outcome = outcome

    def update(self) -> Status:
        cas = self.get_cas()
        place_camera_of(cas, self.world, camera_pose())
        cas.annotations.append(make_object_hypothesis())
        return self.outcome


@dataclass
class EngineFindingAnObject:
    """
    An analysis engine whose pipeline finds one object per pass.
    """

    outcome: Status = Status.SUCCESS
    """
    How each pass ends.
    """

    def name(self) -> str:
        return type(self).__name__

    def implementation(self) -> Pipeline:
        pipeline = Pipeline("FindingAnObject")
        pipeline.add_child(FindsAnObjectInView(self.outcome))
        return pipeline


def cas_holding_hypothesis(world: World, hypothesis: ObjectHypothesis) -> CAS:
    """
    :return: A CAS seen from :func:`camera_pose` in ``world`` that found ``hypothesis``.
    """
    cas = cas_seen_from(world, camera_pose())
    cas.annotations.append(hypothesis)
    return cas


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


@pytest.fixture
def engine() -> EngineRunningABehaviour:
    return EngineRunningABehaviour()


@pytest.fixture
def following_node(ros_default: None) -> Iterator[Node]:
    """
    A node for a RoboKudo started after another one closed.

    Each behaviour tree declares its parameters on its node and never removes them, so a
    later tree needs a node of its own.
    """
    node = Node("following_robokudo")
    yield node
    node.destroy_node()


@pytest.fixture
def joined_robokudo(
    spinning_node: Node, shared_world: SharedWorld, engine: EngineRunningABehaviour
) -> Iterator[RoboKudo]:
    robokudo = RoboKudo(
        node=spinning_node,
        analysis_engine=engine,
        world_discovery_timeout=timedelta(seconds=10),
    )
    yield robokudo
    robokudo.close()


# %% perception world


class TestPerceptionWorld:
    def test_pipeline_perceives_into_the_world_of_robokudo(self, node: Node) -> None:
        robokudo = robokudo_of_its_own(node, EngineFindingAnObject())

        robokudo.start(include_gui=False)

        [annotator] = robokudo.pipeline.children
        assert annotator.world is robokudo.world
        robokudo.close()

    def test_without_a_shared_world_perceives_into_an_empty_world_of_its_own(
        self, node: Node
    ) -> None:
        robokudo = robokudo_of_its_own(node, EngineFindingAnObject())

        robokudo.start(include_gui=False)

        assert robokudo.pipeline.world is robokudo.world
        assert robokudo.world.is_empty()
        robokudo.close()

    def test_engine_whose_root_is_not_a_pipeline_cannot_start(self, node: Node) -> None:
        robokudo = robokudo_of_its_own(node, EngineWithoutPipeline())

        with pytest.raises(EngineRootIsNotAPipelineError):
            robokudo.start(include_gui=False)

        assert not Blackboard().exists(BBIdentifier.ROBOKUDO)

    def test_world_of_its_own_is_not_shared(
        self, node: Node, engine: EngineRunningABehaviour
    ) -> None:
        robokudo = robokudo_of_its_own(node, engine)

        assert robokudo.synchronizer is None
        assert (
            WorldSynchronizer.all_callbacks_of_this_type_from_world(robokudo.world)
            == []
        )


# %% joining a shared world


class TestJoiningSharedWorld:
    def test_starts_from_the_shared_world(
        self, shared_world: SharedWorld, joined_robokudo: RoboKudo
    ) -> None:
        assert WorldSynchronizer.all_callbacks_of_this_type_from_world(
            joined_robokudo.world
        ) == [joined_robokudo.synchronizer]
        assert [body.id for body in joined_robokudo.world.bodies] == [
            body.id for body in shared_world.world.bodies
        ]

    def test_changes_of_the_shared_world_reach_perception(
        self, shared_world: SharedWorld, joined_robokudo: RoboKudo
    ) -> None:
        table = Body(name=PrefixedName("table"))
        with shared_world.world.modify_world():
            shared_world.world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=shared_world.world.root,
                    child=table,
                    world=shared_world.world,
                )
            )

        assert wait_until(
            lambda: table.id in [body.id for body in joined_robokudo.world.bodies]
        )

    def test_perceived_object_reaches_the_shared_world_at_its_pose(
        self, shared_world: SharedWorld, joined_robokudo: RoboKudo
    ) -> None:
        cas = cas_seen_from(joined_robokudo.world, camera_pose(x=1.0))
        cas.annotations.append(make_object_hypothesis(translation=(0.0, 0.5, 2.0)))

        [body_id] = joined_robokudo.perceived_objects.apply(cas)

        perceived_body = joined_robokudo.world.get_kinematic_structure_entity_by_id(
            body_id
        )
        expected_pose = perceived_body.global_transform.to_np()
        assert wait_until(
            lambda: body_id in [body.id for body in shared_world.world.bodies]
            and np.allclose(
                shared_world.world.get_kinematic_structure_entity_by_id(
                    body_id
                ).global_transform.to_np(),
                expected_pose,
            )
        )
        shared_body = shared_world.world.get_kinematic_structure_entity_by_id(body_id)
        assert [shape.scale for shape in shared_body.visual.shapes] == [
            shape.scale for shape in perceived_body.visual.shapes
        ]

    def test_perceived_object_reaches_the_shared_world_with_its_class_label(
        self, shared_world: SharedWorld, joined_robokudo: RoboKudo
    ) -> None:
        class_label = "fork"
        cas = cas_seen_from(joined_robokudo.world, camera_pose())
        cas.annotations.append(make_object_hypothesis(class_label=class_label))

        [body_id] = joined_robokudo.perceived_objects.apply(cas)

        assert wait_until(
            lambda: [
                (type(annotation), annotation.class_label)
                for annotation in shared_world.world.get_semantic_annotations_by_type(
                    IsPerceivable
                )
                if annotation.root.id == body_id
            ]
            == [(PerceivedObject, class_label)]
        )

    def test_classified_object_reaches_the_shared_world_as_its_semantic_annotation(
        self, shared_world: SharedWorld, joined_robokudo: RoboKudo
    ) -> None:
        [body_id] = joined_robokudo.perceived_objects.apply(
            cas_holding_hypothesis(joined_robokudo.world, make_object_hypothesis())
        )

        joined_robokudo.perceived_objects.apply(
            cas_holding_hypothesis(
                joined_robokudo.world, make_object_hypothesis(class_label="cup")
            )
        )

        assert wait_until(
            lambda: [
                type(annotation)
                for annotation in shared_world.world.get_semantic_annotations_by_type(
                    IsPerceivable
                )
                if annotation.root.id == body_id
            ]
            == [Cup]
        )

    def test_closed_robokudo_no_longer_shares_its_world(
        self, joined_robokudo: RoboKudo
    ) -> None:
        joined_robokudo.close()

        assert (
            WorldSynchronizer.all_callbacks_of_this_type_from_world(
                joined_robokudo.world
            )
            == []
        )


# %% one RoboKudo per process


class TestOneRoboKudoPerProcess:
    def test_second_robokudo_cannot_start(
        self, node: Node, engine: EngineRunningABehaviour
    ) -> None:
        running = robokudo_of_its_own(node, engine)
        running.start(include_gui=False)

        with pytest.raises(RoboKudoAlreadyRunningError):
            robokudo_of_its_own(node, EngineRunningABehaviour()).start(
                include_gui=False
            )
        running.close()

    def test_closed_robokudo_makes_room_for_the_next(
        self, node: Node, following_node: Node, engine: EngineRunningABehaviour
    ) -> None:
        robokudo_of_its_own(node, engine).start(include_gui=False)
        RoboKudo.from_blackboard().close()
        following = robokudo_of_its_own(following_node, EngineRunningABehaviour())

        following.start(include_gui=False)

        assert RoboKudo.from_blackboard() is following
        following.close()


# %% ticking


class TestTicking:
    def test_runs_until_the_engine_fails(self, node: Node) -> None:
        failing_behaviour = Failure(name="Fail")
        robokudo = robokudo_of_its_own(
            node, EngineRunningABehaviour(behaviour=failing_behaviour)
        )
        robokudo.start(include_gui=False)
        running = Thread(target=robokudo.run, daemon=True)

        running.start()
        wait_until(lambda: not running.is_alive())

        assert not running.is_alive()
        assert failing_behaviour.status is Status.FAILURE
        robokudo.close()

    def test_stops_when_ros_shuts_down(self) -> None:
        context = Context()
        rclpy.init(context=context)
        node = Node("shutting_down_robokudo", context=context)
        robokudo = robokudo_of_its_own(
            node, EngineRunningABehaviour(behaviour=Running(name="Perceive"))
        )
        robokudo.start(include_gui=False)
        running = Thread(target=robokudo.run, daemon=True)
        running.start()

        rclpy.shutdown(context=context)
        wait_until(lambda: not running.is_alive())

        assert not running.is_alive()
        robokudo.close()
        node.destroy_node()


# %% objects of a finished pass


class TestObjectsOfAFinishedPass:
    def test_objects_of_a_finished_pass_reach_the_world(self, node: Node) -> None:
        robokudo = robokudo_of_its_own(node, EngineFindingAnObject())
        robokudo.start(include_gui=False)

        robokudo.tick()

        [perceived_object] = robokudo.world.get_semantic_annotations_by_type(
            PerceivedObject
        )
        assert perceived_object.root in robokudo.world.bodies
        robokudo.close()

    def test_failed_pass_puts_nothing_into_the_world(self, node: Node) -> None:
        robokudo = robokudo_of_its_own(
            node, EngineFindingAnObject(outcome=Status.FAILURE)
        )
        robokudo.start(include_gui=False)

        robokudo.tick()

        assert robokudo.world.get_semantic_annotations_by_type(PerceivedObject) == []
        robokudo.close()
