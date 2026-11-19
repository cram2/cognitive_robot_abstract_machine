"""
Tests of the RoboKudo perception process of :mod:`robokudo.process`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
import threading
import time
from threading import RLock, Thread, Timer
from uuid import UUID

import numpy as np
import pytest
import rclpy
from py_trees.behaviour import Behaviour
from py_trees.blackboard import Blackboard
from py_trees.behaviours import Failure, Running, Success
from py_trees.common import Status
from rclpy.context import Context
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.messages import WorldUpdate
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import IsPerceivable
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cup,
    UnknownObject,
)
from krrood.exceptions import DataclassException
from krrood.patterns.called_positionally import called_positionally
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body
from std_srvs.srv import Trigger

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.core import BaseAnnotator
from robokudo.exceptions import (
    EngineRootIsNotAPipelineError,
    RoboKudoAlreadyRunningError,
)
from robokudo.identifier import BBIdentifier
from robokudo.pipeline import Pipeline
from robokudo.process import RoboKudo
from robokudo.types.annotation import BoundingBox3DAnnotation

from ._object_hypotheses import make_object_hypothesis
from ._perception_process import (
    EngineRunningABehaviour,
    put_pass_into_world,
    robokudo_of_its_own,
)
from ._perception_world import (
    FrameName,
    SharedWorld,
    camera_pose,
    cas_seen_from,
    place_camera_of,
    wait_until,
)


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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


def robokudo_joining_a_shared_world(
    node: Node, analysis_engine: AnalysisEngineInterface
) -> RoboKudo:
    """
    :return: A RoboKudo that looks for a shared world long enough to join the one
        another process serves.
    """
    return RoboKudo(
        node=node,
        analysis_engine=analysis_engine,
        world_discovery_timeout=timedelta(seconds=10),
    )


def perceived_body_id(world: World) -> UUID:
    """
    :return: The id of the body of the one perceivable object of ``world``.
    """
    [annotation] = world.get_semantic_annotations_by_type(IsPerceivable)
    return annotation.root.id


@pytest.fixture
def engine() -> EngineRunningABehaviour:
    """
    An engine running a behaviour that is always successful.
    """
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


def answer_trigger(
    request: Trigger.Request, response: Trigger.Response
) -> Trigger.Response:
    """
    :return: ``response``, reporting success.
    """
    response.success = True
    return response


def lock_is_free_for_another_thread(
    lock: RLock, timeout: timedelta = timedelta(milliseconds=300)
) -> bool:
    """
    :param lock: A reentrant lock, which the thread holding it would always acquire
        again.
    :param timeout: How long the other thread waits for ``lock``.
    :return: Whether a thread other than the caller acquires ``lock`` within
        ``timeout``.
    """
    acquired = []

    def acquire_and_release() -> None:
        """
        Note whether this thread acquires ``lock`` within ``timeout``.
        """
        if lock.acquire(timeout=timeout.total_seconds()):
            acquired.append(True)
            lock.release()

    probe = Thread(target=acquire_and_release)
    probe.start()
    probe.join()
    return acquired == [True]


@dataclass(kw_only=True)
class WorldFetchAnsweredTooLate:
    """
    A world fetcher service that answers only after a delay, with no world.
    """

    answer_delay: timedelta
    """
    How long the service takes to answer a request.
    """

    @called_positionally
    def answer(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """
        :return: ``response``, holding no world, after :attr:`answer_delay`.
        """
        time.sleep(self.answer_delay.total_seconds())
        return response


@pytest.fixture
def world_fetch_answered_too_late(
    other_process_node: Node,
) -> Iterator[WorldFetchAnsweredTooLate]:
    """
    A process that offers a world fetcher service, which answers only after two seconds.
    """
    late_service = WorldFetchAnsweredTooLate(answer_delay=timedelta(seconds=2))
    service = other_process_node.create_service(
        Trigger, "/answering_too_late/fetch_world", late_service.answer
    )
    yield late_service
    other_process_node.destroy_service(service)


@pytest.fixture
def joined_robokudo(
    node: Node, shared_world: SharedWorld, engine: EngineRunningABehaviour
) -> Iterator[RoboKudo]:
    """
    A started RoboKudo that joined :func:`shared_world`.
    """
    robokudo = robokudo_joining_a_shared_world(node, engine)
    yield robokudo
    robokudo.close()


@pytest.fixture
def robokudo_joined_to_a_robot(
    node: Node, shared_world_with_a_robot: SharedWorld, engine: EngineRunningABehaviour
) -> Iterator[RoboKudo]:
    """
    A started RoboKudo that joined :func:`shared_world_with_a_robot`.
    """
    robokudo = robokudo_joining_a_shared_world(node, engine)
    yield robokudo
    robokudo.close()


# %% perception world


class TestPerceptionWorld:
    """
    The world a RoboKudo perceives into.
    """

    def test_pipeline_perceives_into_the_world_of_robokudo(self, *, node: Node) -> None:
        robokudo = robokudo_of_its_own(node, EngineFindingAnObject())

        robokudo.start(include_gui=False)

        [annotator] = robokudo.pipeline.children
        assert annotator.world is robokudo.world
        robokudo.close()

    def test_without_a_shared_world_perceives_into_an_empty_world_of_its_own(
        self, *, node: Node
    ) -> None:
        robokudo = robokudo_of_its_own(node, EngineFindingAnObject())

        robokudo.start(include_gui=False)

        assert robokudo.pipeline.world is robokudo.world
        assert robokudo.world.is_empty()
        robokudo.close()

    def test_engine_whose_root_is_not_a_pipeline_cannot_start(
        self, *, node: Node
    ) -> None:
        robokudo = robokudo_of_its_own(node, EngineWithoutPipeline())

        with pytest.raises(EngineRootIsNotAPipelineError):
            robokudo.start(include_gui=False)

        assert not Blackboard().exists(BBIdentifier.ROBOKUDO)
        robokudo.close()

    def test_world_of_its_own_is_not_shared(
        self, *, node: Node, engine: EngineRunningABehaviour
    ) -> None:
        robokudo = robokudo_of_its_own(node, engine)

        assert robokudo.synchronizer is None
        assert (
            WorldSynchronizer.all_callbacks_of_this_type_from_world(robokudo.world)
            == []
        )
        robokudo.close()


# %% joining a shared world


class TestJoiningSharedWorld:
    """
    A RoboKudo that joined the world another process shares.
    """

    def test_starts_from_the_shared_world(
        self, *, shared_world: SharedWorld, joined_robokudo: RoboKudo
    ) -> None:
        assert WorldSynchronizer.all_callbacks_of_this_type_from_world(
            joined_robokudo.world
        ) == [joined_robokudo.synchronizer]
        assert [body.id for body in joined_robokudo.world.bodies] == [
            body.id for body in shared_world.world.bodies
        ]

    def test_counts_the_changes_it_joined_with_as_applied(
        self,
        *,
        node: Node,
        shared_world: SharedWorld,
        engine: EngineRunningABehaviour,
    ) -> None:
        """
        A change the shared world published before RoboKudo joined is contained in the
        joined world, so RoboKudo has caught up with it without receiving it.
        """
        with shared_world.world.modify_world():
            shared_world.world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=shared_world.world.root,
                    child=Body(name=PrefixedName("table")),
                    world=shared_world.world,
                )
            )

        robokudo = robokudo_joining_a_shared_world(node, engine)

        assert robokudo.synchronizer.has_applied(
            position=shared_world.synchronizer.latest_published_position
        )
        robokudo.close()

    def test_changes_of_the_shared_world_reach_perception(
        self, *, shared_world: SharedWorld, joined_robokudo: RoboKudo
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

    def test_objects_of_a_pass_are_published_once_the_world_is_released(
        self, *, joined_robokudo: RoboKudo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Every update about the objects of a pass leaves RoboKudo only while no thread
        holds its world, as semDT requires of a modification.
        """
        synchronizer = joined_robokudo.synchronizer
        world_lock = joined_robokudo.world.state.world_lock
        world_released_during_publish = []
        publish = synchronizer.publish

        def publish_noting_the_world_lock(update: WorldUpdate) -> None:
            """
            Note whether the world is released when ``update`` is published, then
            publish it.
            """
            world_released_during_publish.append(
                lock_is_free_for_another_thread(world_lock)
            )
            publish(update)

        monkeypatch.setattr(synchronizer, "publish", publish_noting_the_world_lock)

        put_pass_into_world(
            joined_robokudo,
            cas_seen_from(joined_robokudo.world, camera_pose(x=1.0)),
            make_object_hypothesis(translation=(0.0, 0.5, 2.0)),
        )

        assert world_released_during_publish
        assert world_released_during_publish == [True] * len(
            world_released_during_publish
        )

    def test_perceived_object_reaches_the_shared_world_at_its_pose(
        self, *, shared_world: SharedWorld, joined_robokudo: RoboKudo
    ) -> None:
        put_pass_into_world(
            joined_robokudo,
            cas_seen_from(joined_robokudo.world, camera_pose(x=1.0)),
            make_object_hypothesis(translation=(0.0, 0.5, 2.0)),
        )

        body_id = perceived_body_id(joined_robokudo.world)

        perceived_body = joined_robokudo.world.get_kinematic_structure_entity_by_id(
            body_id
        )
        expected_pose = perceived_body.global_transform.to_np()
        assert wait_until(
            lambda: (
                body_id in [body.id for body in shared_world.world.bodies]
                and np.allclose(
                    shared_world.world.get_kinematic_structure_entity_by_id(
                        body_id
                    ).global_transform.to_np(),
                    expected_pose,
                )
            )
        )
        shared_body = shared_world.world.get_kinematic_structure_entity_by_id(body_id)
        assert [shape.scale for shape in shared_body.visual.shapes] == [
            shape.scale for shape in perceived_body.visual.shapes
        ]

    def test_perceived_object_reaches_the_shared_world_with_its_class_label(
        self, *, shared_world: SharedWorld, joined_robokudo: RoboKudo
    ) -> None:
        class_label = "fork"
        put_pass_into_world(
            joined_robokudo,
            cas_seen_from(joined_robokudo.world, camera_pose()),
            make_object_hypothesis(class_label=class_label),
        )

        body_id = perceived_body_id(joined_robokudo.world)

        assert wait_until(
            lambda: (
                [
                    (type(annotation), annotation.class_label)
                    for annotation in shared_world.world.get_semantic_annotations_by_type(
                        IsPerceivable
                    )
                    if annotation.root.id == body_id
                ]
                == [(UnknownObject, class_label)]
            )
        )

    def test_classified_object_reaches_the_shared_world_as_its_semantic_annotation(
        self, *, shared_world: SharedWorld, joined_robokudo: RoboKudo
    ) -> None:
        put_pass_into_world(
            joined_robokudo,
            cas_seen_from(joined_robokudo.world, camera_pose()),
            make_object_hypothesis(),
        )
        body_id = perceived_body_id(joined_robokudo.world)

        put_pass_into_world(
            joined_robokudo,
            cas_seen_from(joined_robokudo.world, camera_pose()),
            make_object_hypothesis(class_label="cup"),
        )

        assert wait_until(
            lambda: (
                [
                    type(annotation)
                    for annotation in shared_world.world.get_semantic_annotations_by_type(
                        IsPerceivable
                    )
                    if annotation.root.id == body_id
                ]
                == [Cup]
            )
        )

    def test_perceiving_adds_no_camera_to_the_shared_world(
        self, *, shared_world: SharedWorld, joined_robokudo: RoboKudo
    ) -> None:
        put_pass_into_world(
            joined_robokudo,
            cas_seen_from(joined_robokudo.world, camera_pose()),
            make_object_hypothesis(),
        )
        body_id = perceived_body_id(joined_robokudo.world)

        assert wait_until(
            lambda: body_id in [body.id for body in shared_world.world.bodies]
        )
        assert shared_world.world.get_bodies_by_name(FrameName.CAMERA) == []

    def test_closed_robokudo_no_longer_shares_its_world(
        self, *, joined_robokudo: RoboKudo
    ) -> None:
        joined_robokudo.close()

        assert (
            WorldSynchronizer.all_callbacks_of_this_type_from_world(
                joined_robokudo.world
            )
            == []
        )


# %% joining a world with a robot


class TestJoiningAWorldWithARobot:
    """
    A RoboKudo joined to a shared world with a robot places what it perceives where the
    robot's camera saw it.
    """

    def test_frame_is_placed_where_tf_located_the_camera_of_the_moved_head(
        self,
        *,
        shared_world_with_a_robot: SharedWorld,
        robokudo_joined_to_a_robot: RoboKudo,
    ) -> None:
        shared = shared_world_with_a_robot.world
        shared.get_body_by_name(
            FrameName.HEAD
        ).parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.5, z=1.2, yaw=0.3, reference_frame=shared.root
        )
        moved_camera_pose = shared.get_body_by_name(
            FrameName.CAMERA
        ).global_transform.to_np()
        camera_body = robokudo_joined_to_a_robot.world.get_body_by_name(
            FrameName.CAMERA
        )
        assert wait_until(
            lambda: np.allclose(camera_body.global_transform.to_np(), moved_camera_pose)
        )

        world_T_camera = camera_pose(x=4.0)

        cas = cas_seen_from(robokudo_joined_to_a_robot.world, world_T_camera)

        assert np.allclose(
            cas.camera_observation.world_T_camera.to_np(), world_T_camera.to_np()
        )

    def test_perceived_object_reaches_the_shared_world_where_the_robot_saw_it(
        self,
        *,
        shared_world_with_a_robot: SharedWorld,
        robokudo_joined_to_a_robot: RoboKudo,
    ) -> None:
        shared = shared_world_with_a_robot.world
        hypothesis = make_object_hypothesis(translation=(0.0, 0.5, 2.0))
        bounding_box = hypothesis.latest_annotation_of_type(
            annotation_type=BoundingBox3DAnnotation
        )
        expected_pose = (
            shared.get_body_by_name(FrameName.CAMERA).global_transform.to_np()
            @ HomogeneousTransformationMatrix.from_xyz_quaternion(
                *bounding_box.pose.translation, *bounding_box.pose.rotation
            ).to_np()
        )

        put_pass_into_world(
            robokudo_joined_to_a_robot,
            cas_seen_from(
                robokudo_joined_to_a_robot.world,
                HomogeneousTransformationMatrix(
                    data=shared.get_body_by_name(
                        FrameName.CAMERA
                    ).global_transform.to_np()
                ),
            ),
            hypothesis,
        )
        body_id = perceived_body_id(robokudo_joined_to_a_robot.world)

        assert wait_until(
            lambda: (
                body_id in [body.id for body in shared.bodies]
                and np.allclose(
                    shared.get_kinematic_structure_entity_by_id(
                        body_id
                    ).global_transform.to_np(),
                    expected_pose,
                )
            )
        )


# %% discovering a shared world


class TestDiscoveringASharedWorld:
    """
    A RoboKudo looking for the world another process serves.
    """

    def test_joins_a_world_served_while_it_is_still_looking(
        self, *, node: Node, other_process_node: Node, engine: EngineRunningABehaviour
    ) -> None:
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName(FrameName.WORLD)))
        served_worlds = []
        serving = Timer(
            1.0,
            lambda: served_worlds.append(
                SharedWorld.served_through(node=other_process_node, world=world)
            ),
        )
        serving.start()

        robokudo = robokudo_joining_a_shared_world(node, engine)

        serving.join()
        assert robokudo.synchronizer is not None
        assert [body.id for body in robokudo.world.bodies] == [
            body.id for body in world.bodies
        ]
        robokudo.close()
        served_worlds[0].close()

    def test_stops_serving_its_node_when_the_shared_world_arrives_too_late(
        self,
        *,
        node: Node,
        engine: EngineRunningABehaviour,
        world_fetch_answered_too_late: WorldFetchAnsweredTooLate,
    ) -> None:
        """
        A world fetch that takes longer than the configured fetch timeout fails
        RoboKudo's start, and leaves no thread serving its node.
        """
        threads_before = set(threading.enumerate())

        with pytest.raises(TimeoutError):
            RoboKudo(
                node=node,
                analysis_engine=engine,
                world_discovery_timeout=timedelta(seconds=10),
                world_fetch_timeout=world_fetch_answered_too_late.answer_delay / 4,
            )

        assert wait_until(lambda: set(threading.enumerate()) <= threads_before)

    def test_stops_serving_its_node_when_the_shared_world_cannot_be_synchronized(
        self,
        *,
        node: Node,
        engine: EngineRunningABehaviour,
        shared_world: SharedWorld,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A synchronizer that cannot be created fails RoboKudo's start, and leaves no
        thread serving its node.
        """
        threads_before = set(threading.enumerate())

        def unavailable_synchronizer(**kwargs: object) -> WorldSynchronizer:
            """
            Refuse to create a synchronizer, as the network may.
            """
            raise SynchronizerUnavailableError()

        monkeypatch.setattr(
            "semantic_digital_twin.adapters.ros.world_fetcher.WorldSynchronizer",
            unavailable_synchronizer,
        )

        with pytest.raises(SynchronizerUnavailableError):
            robokudo_joining_a_shared_world(node, engine)

        assert wait_until(lambda: set(threading.enumerate()) <= threads_before)


@dataclass(kw_only=True)
class SynchronizerUnavailableError(DataclassException):
    """
    Raised in place of creating a synchronizer, as when the network refuses one.
    """

    def error_message(self) -> str:
        return "The world synchronizer could not be created."

    def suggest_correction(self) -> str:
        return ""


# %% one RoboKudo per process


class TestOneRoboKudoPerProcess:
    """
    Only one RoboKudo runs per process at a time.
    """

    def test_second_robokudo_cannot_start(
        self, *, node: Node, following_node: Node, engine: EngineRunningABehaviour
    ) -> None:
        running = robokudo_of_its_own(node, engine)
        running.start(include_gui=False)
        second = robokudo_of_its_own(following_node, EngineRunningABehaviour())

        with pytest.raises(RoboKudoAlreadyRunningError):
            second.start(include_gui=False)
        second.close()
        running.close()

    def test_closed_robokudo_makes_room_for_the_next(
        self, *, node: Node, following_node: Node, engine: EngineRunningABehaviour
    ) -> None:
        closed = robokudo_of_its_own(node, engine)
        closed.start(include_gui=False)
        closed.close()
        following = robokudo_of_its_own(following_node, EngineRunningABehaviour())

        following.start(include_gui=False)

        assert Blackboard().get(BBIdentifier.ROBOKUDO) is following
        following.close()


# %% serving its node


class TestServingItsNode:
    """
    The node a RoboKudo serves in the background.
    """

    def test_requests_to_its_node_are_answered(
        self, *, node: Node, other_process_node: Node, engine: EngineRunningABehaviour
    ) -> None:
        node.create_service(Trigger, "trigger_robokudo", answer_trigger)
        robokudo = robokudo_of_its_own(node, engine)
        client = other_process_node.create_client(Trigger, "trigger_robokudo")
        assert client.wait_for_service(timeout_sec=5.0)

        response = client.call_async(Trigger.Request())

        assert wait_until(response.done)
        robokudo.close()

    def test_closed_robokudo_no_longer_answers_requests_to_its_node(
        self, *, node: Node, other_process_node: Node, engine: EngineRunningABehaviour
    ) -> None:
        node.create_service(Trigger, "trigger_robokudo", answer_trigger)
        robokudo_of_its_own(node, engine).close()
        client = other_process_node.create_client(Trigger, "trigger_robokudo")
        assert client.wait_for_service(timeout_sec=5.0)

        response = client.call_async(Trigger.Request())

        assert not wait_until(response.done, timeout=timedelta(seconds=1))


# %% ticking


class TestTicking:
    """
    RoboKudo ticking its analysis engine.
    """

    def test_runs_until_the_engine_fails(self, *, node: Node) -> None:
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
    """
    The objects a pass of the analysis engine finds.
    """

    def test_objects_of_a_finished_pass_reach_the_world(self, *, node: Node) -> None:
        robokudo = robokudo_of_its_own(node, EngineFindingAnObject())
        robokudo.start(include_gui=False)

        robokudo.tick_tree()

        [unknown_object] = robokudo.world.get_semantic_annotations_by_type(
            UnknownObject
        )
        assert unknown_object.root in robokudo.world.bodies
        robokudo.close()

    def test_failed_pass_puts_nothing_into_the_world(self, *, node: Node) -> None:
        robokudo = robokudo_of_its_own(
            node, EngineFindingAnObject(outcome=Status.FAILURE)
        )
        robokudo.start(include_gui=False)

        robokudo.tick_tree()

        assert robokudo.world.get_semantic_annotations_by_type(UnknownObject) == []
        robokudo.close()


# %% publishing its own world


def robokudo_publishing_its_own_world(node: Node) -> RoboKudo:
    """
    :return: A RoboKudo that perceives into a world of its own, finds one object per
        pass and publishes its world.
    """
    return RoboKudo(
        node=node,
        analysis_engine=EngineFindingAnObject(),
        world_discovery_timeout=timedelta(0),
        publishes_own_world=True,
    )


class TestPublishingItsOwnWorld:
    """
    A RoboKudo with a world of its own publishes that world to RViz.
    """

    def test_publishes_its_objects_without_claiming_the_frames_of_the_camera_pose(
        self, *, node: Node
    ) -> None:
        robokudo = robokudo_publishing_its_own_world(node)
        robokudo.start(include_gui=False)

        robokudo.tick_tree()

        [unknown_object] = robokudo.world.get_semantic_annotations_by_type(
            UnknownObject
        )
        tf_publisher = robokudo.world_publisher.tf_publisher
        published = tf_publisher.tf_model_callback
        transforms = (
            published.tf_message.transforms + published.static_tf_message.transforms
        )
        published_frames = {
            frame
            for transform in transforms
            for frame in (transform.header.frame_id, transform.child_frame_id)
        }
        assert tf_publisher.frame_names.assign(unknown_object.root) in published_frames
        assert published_frames & {FrameName.WORLD, FrameName.CAMERA} == set()
        robokudo.close()

    def test_publishes_markers_of_its_objects(self, *, node: Node) -> None:
        robokudo = robokudo_publishing_its_own_world(node)
        robokudo.start(include_gui=False)

        robokudo.tick_tree()

        [unknown_object] = robokudo.world.get_semantic_annotations_by_type(
            UnknownObject
        )
        world_publisher = robokudo.world_publisher
        assert {
            marker.header.frame_id for marker in world_publisher.markers.markers
        } == {world_publisher.tf_publisher.frame_names.assign(unknown_object.root)}
        robokudo.close()

    def test_publishes_nothing_unless_asked_to(self, *, node: Node) -> None:
        robokudo = robokudo_of_its_own(node, EngineFindingAnObject())
        robokudo.start(include_gui=False)

        robokudo.tick_tree()

        assert robokudo.world_publisher is None
        robokudo.close()

    def test_leaves_publishing_a_shared_world_to_the_process_serving_it(
        self, *, node: Node, shared_world: SharedWorld
    ) -> None:
        robokudo = RoboKudo(
            node=node,
            analysis_engine=EngineFindingAnObject(),
            world_discovery_timeout=timedelta(seconds=10),
            publishes_own_world=True,
        )
        robokudo.start(include_gui=False)

        robokudo.tick_tree()

        assert robokudo.world_publisher is None
        robokudo.close()

    def test_closed_robokudo_no_longer_publishes_its_world(self, *, node: Node) -> None:
        robokudo = robokudo_publishing_its_own_world(node)
        robokudo.start(include_gui=False)
        robokudo.tick_tree()

        robokudo.close()

        assert (
            VizMarkerPublisher.all_callbacks_of_this_type_from_world(robokudo.world)
            == []
        )
        assert TFPublisher.all_callbacks_of_this_type_from_world(robokudo.world) == []
