"""
Coverage for the perception sources and for writing their detections into the world.

The sim and the real robot run the same plan; only the source of the detections differs.
These tests pin that seam: that the right source is chosen for an execution type, that
each source reports the same :class:`~coraplex.perception.Detection` shape, and that
applying a detection moves the annotated body.
"""

from __future__ import annotations

import inspect
import json
import threading
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest
import rclpy
from rclpy.action import ActionServer
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from typing_extensions import List, Tuple

from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    ExecutionType,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.exceptions import (
    AmbiguousDetection,
    NothingDetected,
    PerceivedObjectNotInWorld,
    PerceptionSourceUnavailable,
    UnidentifiedDetections,
    UnknownExecutionType,
)
from coraplex.execution_environment import simulated_robot
from coraplex.perception import (
    ROBOKUDO_QUERY_ACTION_NAME,
    Detection,
    PerceptionInterface,
    PerceptionQuery,
    RoboKudoPerception,
    WorldPerception,
)
from coraplex.plans.factories import execute_single
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans import MoveToolCenterPointMotion
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.motions.misc import DetectingMotion, PerceptionTask
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.ros_context import RosContextExtension
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox

PERCEIVED_MILK_POSITION = (2.6, 2.2, 1.05)
"""
Where the perception sources in these tests claim to have seen the milk.

Offset from where the fixture spawns it (2.37, 2.0, 1.05) so that a body which did not
move fails the assertions.
"""

# %% choosing a source


@pytest.mark.parametrize(
    "execution_type, expected_source",
    [
        (ExecutionType.SIMULATED, WorldPerception),
        (ExecutionType.NO_EXECUTION, WorldPerception),
        (ExecutionType.REAL, RoboKudoPerception),
    ],
)
def test_source_is_chosen_by_execution_type(execution_type, expected_source):
    """
    The plan does not change between sim and real; the execution type alone decides
    where detections come from.
    """
    source = PerceptionInterface.for_execution_type(execution_type, ros_node=None)

    assert type(source) is expected_source


# %% applying detections


def test_detection_moves_the_annotated_body_to_the_perceived_pose(
    immutable_model_world,
):
    """
    Applying a detection is what makes perception load-bearing: the body ends up where
    the source saw it, not where it was spawned.
    """
    world, view, context = immutable_model_world
    milk_body = world.get_body_by_name("milk.stl")
    perceived_pose = Pose.from_xyz_rpy(
        *PERCEIVED_MILK_POSITION, reference_frame=world.root
    )

    annotations = Detection(semantic_annotation=Milk, pose=perceived_pose).apply_to(
        world
    )

    assert annotations == world.get_semantic_annotations_by_type(Milk)
    np.testing.assert_allclose(
        milk_body.global_pose.to_position().to_np().flatten()[:3],
        PERCEIVED_MILK_POSITION,
        atol=1e-9,
    )


def test_detection_of_an_object_the_world_does_not_hold_is_rejected(
    immutable_model_world,
):
    """
    An annotation with nothing behind it in the world has no body to write a pose to, so
    it must not pass silently.
    """
    world, view, context = immutable_model_world

    with pytest.raises(PerceivedObjectNotInWorld):
        Detection(
            semantic_annotation=Bowl,
            pose=Pose(reference_frame=world.root),
        ).apply_to(world)


def test_detection_matching_several_bodies_is_rejected(mutable_model_world):
    """
    With the annotation on two different bodies there is no way to tell which one was
    seen, so the ambiguity is reported instead of guessed away.

    Uses the mutable world because adding an annotation is a model change, which the
    immutable fixture does not roll back.
    """
    world, view, context = mutable_model_world
    with world.modify_world():
        world.add_semantic_annotation(Milk(root=world.get_body_by_name("spoon.stl")))

    with pytest.raises(AmbiguousDetection):
        Detection(
            semantic_annotation=Milk, pose=Pose(reference_frame=world.root)
        ).apply_to(world)


def test_several_annotations_on_one_body_are_not_ambiguous(mutable_model_world):
    """
    Two annotations describing the same body name one object, so the detection applies
    to both rather than being rejected.
    """
    world, view, context = mutable_model_world
    milk_body = world.get_body_by_name("milk.stl")
    with world.modify_world():
        world.add_semantic_annotation(Milk(root=milk_body))
    perceived_pose = Pose.from_xyz_rpy(
        *PERCEIVED_MILK_POSITION, reference_frame=world.root
    )

    annotations = Detection(semantic_annotation=Milk, pose=perceived_pose).apply_to(
        world
    )

    assert len(annotations) == 2
    assert {annotation.root for annotation in annotations} == {milk_body}
    np.testing.assert_allclose(
        milk_body.global_pose.to_position().to_np().flatten()[:3],
        PERCEIVED_MILK_POSITION,
        atol=1e-9,
    )


# %% distrusting a source's orientation


def test_untrusted_orientation_still_moves_the_body_to_the_perceived_position(
    immutable_model_world,
):
    """
    With ``trust_orientation=False``, the position still comes from the detection: only
    the orientation is left alone.
    """
    world, view, context = immutable_model_world
    milk_body = world.get_body_by_name("milk.stl")
    perceived_pose = Pose.from_xyz_rpy(
        *PERCEIVED_MILK_POSITION, yaw=np.pi / 2, reference_frame=world.root
    )

    Detection(semantic_annotation=Milk, pose=perceived_pose).apply_to(
        world, trust_orientation=False
    )

    np.testing.assert_allclose(
        milk_body.global_pose.to_position().to_np().flatten()[:3],
        PERCEIVED_MILK_POSITION,
        atol=1e-9,
    )


def test_untrusted_orientation_keeps_the_bodys_existing_orientation(
    immutable_model_world,
):
    """
    The detected rotation (a 90 degree yaw here) must not reach the body: it keeps
    whatever rotation it already had before the detection was applied.
    """
    world, view, context = immutable_model_world
    milk_body = world.get_body_by_name("milk.stl")
    orientation_before_detection = (
        milk_body.parent_connection.origin.to_rotation_matrix().to_np()
    )
    perceived_pose = Pose.from_xyz_rpy(
        *PERCEIVED_MILK_POSITION, yaw=np.pi / 2, reference_frame=world.root
    )

    Detection(semantic_annotation=Milk, pose=perceived_pose).apply_to(
        world, trust_orientation=False
    )

    np.testing.assert_allclose(
        milk_body.parent_connection.origin.to_rotation_matrix().to_np(),
        orientation_before_detection,
        atol=1e-9,
    )


def test_trusting_orientation_is_the_default():
    """
    Existing callers of ``apply_to`` must keep getting the detected orientation applied
    unless they explicitly opt out.
    """
    assert (
        inspect.signature(Detection.apply_to).parameters["trust_orientation"].default
        is True
    )


# %% reading detections out of the world


def test_world_perception_reports_the_pose_the_world_holds(
    immutable_model_world, whole_scene_region
):
    """
    The simulated source stands in for a perfect sensor, so its detection must match the
    body's current pose rather than any stored or spawned value.
    """
    world, view, context = immutable_model_world
    milk_body = world.get_body_by_name("milk.stl")
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        *PERCEIVED_MILK_POSITION, reference_frame=world.root
    )
    query = PerceptionQuery(Milk, whole_scene_region, view, world)

    detections = WorldPerception().detect(query)

    assert [detection.semantic_annotation for detection in detections] == [Milk]
    np.testing.assert_allclose(
        detections[0].pose.to_position().to_np().flatten()[:3],
        milk_body.global_pose.to_position().to_np().flatten()[:3],
        atol=1e-9,
    )


def test_world_perception_reports_nothing_outside_the_queried_region(
    immutable_model_world,
):
    """
    The region is part of the question, so a body outside it is not an answer.
    """
    world, view, context = immutable_model_world
    empty_region = BoundingBox(
        origin=HomogeneousTransformationMatrix(reference_frame=world.root),
        min_x=-10,
        min_y=-10,
        min_z=-10,
        max_x=-9,
        max_y=-9,
        max_z=-9,
    )
    query = PerceptionQuery(Milk, empty_region, view, world)

    assert WorldPerception().detect(query) == []


# %% perception correcting a grasp


def test_detection_corrects_a_grasp_planned_before_it(immutable_model_world):
    """
    What the whole seam is for: the plan is expanded (and the grasp planned) against
    whatever pose the world happened to hold, and the detection that runs afterwards
    moves the grasp with the object.

    Without this, a wrong prior in the world silently aims the reach at empty space.
    """
    world, view, context = immutable_model_world
    milk_body = world.get_body_by_name("milk.stl")
    wrong_prior = (1.0, 1.0, 1.0)
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        *wrong_prior, reference_frame=world.root
    )

    plan = execute_single(
        PickUpAction(
            milk_body,
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.NoAlignment,
                view.right_arm.end_effector,
            ),
        ),
        context=context,
    )
    plan.notify()
    targets = [
        node.designator.target
        for node in plan.descendants
        if isinstance(node, MotionNode)
        and isinstance(node.designator, MoveToolCenterPointMotion)
    ]

    def distances_to(position) -> list[float]:
        return [
            float(
                np.linalg.norm(
                    world.transform(target, world.root)
                    .to_position()
                    .to_np()
                    .flatten()[:3]
                    - np.array(position)
                )
            )
            for target in targets
        ]

    # Planned against the wrong prior, one motion sits exactly on the believed object.
    assert min(distances_to(wrong_prior)) < 1e-9
    assert min(distances_to(PERCEIVED_MILK_POSITION)) > 1.0

    Detection(
        semantic_annotation=Milk,
        pose=Pose.from_xyz_rpy(*PERCEIVED_MILK_POSITION, reference_frame=world.root),
    ).apply_to(world)

    assert min(distances_to(PERCEIVED_MILK_POSITION)) < 1e-9


# %% reading detections off the robokudo action


@dataclass
class ReportedObject:
    """
    One object a stand-in perception pipeline claims to have seen.
    """

    class_label: str
    """
    Label to report it under; empty for a pipeline that localizes without recognizing.
    """

    position: Tuple[float, float, float]
    """
    Where to report it.
    """


class RecordedQueryServer:
    """
    Action server answering every perception query with a fixed set of object
    designators.

    Stands in for a running perception pipeline so the conversion from its message
    format into a :class:`~coraplex.perception.Detection` is exercised without one, and
    so the cases a real pipeline produces — nothing found, several candidates, no class
    label — can each be reproduced.
    """

    def __init__(
        self, node_name: str, action_name: str, reports: List[ReportedObject]
    ) -> None:
        from robokudo_msgs.action import Query

        self.reports = reports
        self.received_types = []
        self.node = rclpy.create_node(node_name)
        self.server = ActionServer(self.node, Query, action_name, self.execute_callback)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(
            target=self.executor.spin, daemon=True, name=node_name
        )
        self.thread.start()

    def execute_callback(self, goal_handle):
        from geometry_msgs.msg import PoseStamped
        from robokudo_msgs.action import Query
        from robokudo_msgs.msg import ObjectDesignator
        from std_msgs.msg import Header

        self.received_types.append(goal_handle.request.obj.type)
        goal_handle.succeed()

        designators = []
        for report in self.reports:
            pose_stamped = PoseStamped(header=Header(frame_id="map"))
            (
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z,
            ) = report.position
            pose_stamped.pose.orientation.w = 1.0
            designators.append(
                ObjectDesignator(type=report.class_label, pose=[pose_stamped])
            )
        return Query.Result(res=designators)

    def stop(self):
        self.server.destroy()
        self.executor.shutdown()
        self.thread.join(timeout=2.0)
        self.node.destroy_node()


@pytest.fixture
def robokudo_query_server(rclpy_node):
    """
    A stand-in perception pipeline serving the query action on its own node.
    """
    pytest.importorskip("robokudo_msgs")
    server = RecordedQueryServer(
        node_name="robokudo_stand_in",
        action_name="robokudo/query",
        reports=[ReportedObject("Milk", PERCEIVED_MILK_POSITION)],
    )
    yield server
    server.stop()


@pytest.fixture
def query_server_reporting(rclpy_node):
    """
    Start a stand-in pipeline reporting whatever a test asks it to.
    """
    pytest.importorskip("robokudo_msgs")
    started = []

    def start(reports: List[ReportedObject]) -> RecordedQueryServer:
        server = RecordedQueryServer(
            node_name="robokudo_stand_in",
            action_name="robokudo/query",
            reports=reports,
        )
        started.append(server)
        return server

    yield start
    for server in started:
        server.stop()


def test_robokudo_detection_is_named_and_placed_by_the_pipeline(
    immutable_model_world, whole_scene_region, rclpy_node, robokudo_query_server
):
    """
    The real source is asked for the queried annotation and contributes the pose;
    everything downstream treats its detection the same as a simulated one.
    """
    world, view, context = immutable_model_world
    query = PerceptionQuery(Milk, whole_scene_region, view, world)

    detections = RoboKudoPerception(ros_node=rclpy_node).detect(query)

    assert robokudo_query_server.received_types == ["milk"]
    assert [detection.semantic_annotation for detection in detections] == [Milk]
    np.testing.assert_allclose(
        detections[0].pose.to_position().to_np().flatten()[:3],
        PERCEIVED_MILK_POSITION,
        atol=1e-9,
    )


def test_robokudo_detection_moves_the_body_in_the_world(
    immutable_model_world, whole_scene_region, rclpy_node, robokudo_query_server
):
    """
    End to end for the real path: what the pipeline reports is what the world ends up
    holding, which is what the grasp is later planned against.
    """
    world, view, context = immutable_model_world
    query = PerceptionQuery(Milk, whole_scene_region, view, world)

    for detection in RoboKudoPerception(ros_node=rclpy_node).detect(query):
        detection.apply_to(world)

    np.testing.assert_allclose(
        world.get_body_by_name("milk.stl")
        .global_pose.to_position()
        .to_np()
        .flatten()[:3],
        PERCEIVED_MILK_POSITION,
        atol=1e-9,
    )


# %% pipelines that localize without recognizing


def test_untyped_detection_is_identified_from_the_query(
    immutable_model_world, whole_scene_region, rclpy_node, query_server_reporting
):
    """
    A pipeline of plane and cluster annotators reports where an object is but not what
    it is, so the annotation comes from what was asked for.
    """
    world, view, context = immutable_model_world
    query_server_reporting([ReportedObject("", PERCEIVED_MILK_POSITION)])
    query = PerceptionQuery(Milk, whole_scene_region, view, world)

    detections = RoboKudoPerception(ros_node=rclpy_node).detect(query)

    assert [detection.semantic_annotation for detection in detections] == [Milk]
    np.testing.assert_allclose(
        detections[0].pose.to_position().to_np().flatten()[:3],
        PERCEIVED_MILK_POSITION,
        atol=1e-9,
    )


def test_pipeline_reporting_nothing_is_an_error(
    immutable_model_world, whole_scene_region, rclpy_node, query_server_reporting
):
    """
    Finding nothing must not pass as "saw nothing worth moving": the plan would then
    grasp at the pose the object was spawned with, believing it was confirmed.
    """
    world, view, context = immutable_model_world
    query_server_reporting([])
    query = PerceptionQuery(Milk, whole_scene_region, view, world)

    with pytest.raises(NothingDetected):
        RoboKudoPerception(ros_node=rclpy_node).detect(query)


def test_several_untyped_candidates_are_not_guessed_between(
    immutable_model_world, whole_scene_region, rclpy_node, query_server_reporting
):
    """
    Without a class label there is nothing to tell two clusters apart, so the ambiguity
    is reported rather than resolved by picking one.
    """
    world, view, context = immutable_model_world
    query_server_reporting(
        [
            ReportedObject("", PERCEIVED_MILK_POSITION),
            ReportedObject("", (1.0, 1.0, 1.0)),
        ]
    )
    query = PerceptionQuery(Milk, whole_scene_region, view, world)

    with pytest.raises(UnidentifiedDetections):
        RoboKudoPerception(ros_node=rclpy_node).detect(query)


def test_labelled_candidates_are_narrowed_to_the_requested_type(
    immutable_model_world, whole_scene_region, rclpy_node, query_server_reporting
):
    """
    Once a classifying annotator is in the pipeline its labels are used to discard the
    objects that were not asked for, instead of reporting them as ambiguous.
    """
    world, view, context = immutable_model_world
    query_server_reporting(
        [
            ReportedObject("Milk", PERCEIVED_MILK_POSITION),
            ReportedObject("Spoon", (1.0, 1.0, 1.0)),
        ]
    )
    query = PerceptionQuery(Milk, whole_scene_region, view, world)

    detections = RoboKudoPerception(ros_node=rclpy_node).detect(query)

    assert [detection.semantic_annotation for detection in detections] == [Milk]


# %% perception inside the motion chart


def build_perception_task(
    task: PerceptionTask, world: World, ros_node: Node
) -> MotionStatechartContext:
    """
    Put a perception task through the build phase a motion state chart would give it.

    :param task: The task to build.
    :param world: The world the chart runs against.
    :param ros_node: Node the task reaches a real perception pipeline through.
    :return: The context it was built with.
    """
    context = MotionStatechartContext(world=world)
    context.add_extension(RosContextExtension(ros_node))
    task.build(context)
    return context


def run_perception_task(task: PerceptionTask, context: MotionStatechartContext) -> None:
    """
    Start a built perception task and tick it once, as its chart does.

    :param task: The built task to run.
    :param context: The context it was built with.
    """
    task.on_start(context)
    assert task.on_tick(context) == ObservationStateValues.TRUE


def test_perception_task_moves_the_detected_body(
    immutable_model_world, whole_scene_region, rclpy_node, robokudo_query_server
):
    """
    Answering the query inside the chart has to be worth as much as answering it between
    charts: the body ends up where the pipeline saw it.
    """
    world, view, context = immutable_model_world
    query = PerceptionQuery(Milk, whole_scene_region, view, world)
    task = PerceptionTask(query=query, execution_type=ExecutionType.REAL)

    run_perception_task(task, build_perception_task(task, world, rclpy_node))

    np.testing.assert_allclose(
        world.get_body_by_name("milk.stl")
        .global_pose.to_position()
        .to_np()
        .flatten()[:3],
        PERCEIVED_MILK_POSITION,
        atol=1e-9,
    )


@dataclass
class UnanswerablePerception(PerceptionInterface):
    """
    Source that cannot answer, standing in for any reason a query fails.

    Lets the test choose the failure a task has to surface, instead of arranging the
    conditions that would provoke it.
    """

    failure: BaseException
    """
    What answering the query raises.
    """

    def detect(self, query: PerceptionQuery) -> List[Detection]:
        raise self.failure


def test_perception_task_reports_a_failed_query_as_itself(
    immutable_model_world, whole_scene_region, rclpy_node
):
    """
    A detection that could not be made must reach the plan as the failure it was, not as
    a motion that merely did not finish, so failure handling can tell the reasons apart.

    The source raises on the tick, which is the same way a
    :class:`~giskardpy.motion_statechart.graph_node.CancelMotion` aborts a chart.
    """
    world, view, context = immutable_model_world
    query = PerceptionQuery(Milk, whole_scene_region, view, world)
    task = PerceptionTask(query=query, execution_type=ExecutionType.SIMULATED)
    build_context = build_perception_task(task, world, rclpy_node)
    task.perception_source = UnanswerablePerception(
        PerceptionSourceUnavailable(ROBOKUDO_QUERY_ACTION_NAME)
    )

    with pytest.raises(PerceptionSourceUnavailable):
        run_perception_task(task, build_context)


def test_perception_task_answers_its_query_only_once(
    immutable_model_world, whole_scene_region, rclpy_node, robokudo_query_server
):
    """
    The query is expensive, so a task that is ticked again after it answered must report
    what it already found instead of asking the pipeline a second time.
    """
    world, view, context = immutable_model_world
    query = PerceptionQuery(Milk, whole_scene_region, view, world)
    task = PerceptionTask(query=query, execution_type=ExecutionType.REAL)
    build_context = build_perception_task(task, world, rclpy_node)

    run_perception_task(task, build_context)
    assert task.on_tick(build_context) == ObservationStateValues.TRUE

    assert robokudo_query_server.received_types == ["milk"]


def receiving_world_kwargs(world: World) -> dict:
    """
    Build the deserialization keyword arguments a controller receiving a chart builds.

    :param world: The world the arriving chart is resolved against.
    :return: The tracker and world keyword arguments.
    """
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    return {"world": world, **tracker.create_kwargs()}


def test_perception_task_survives_a_json_round_trip(
    immutable_model_world, whole_scene_region
):
    """
    On the real robot the chart is serialized to a controller holding its own copy of
    the world, so the query has to arrive pointing at that copy's entities rather than
    at the ones it was built from.

    Deserializing into a separate world is what makes this meaningful: resolving the robot
    and the region's frame by id can only be seen to work when the objects behind those
    ids are not the ones that were serialized.
    """
    world, view, context = immutable_model_world
    receiving_world = deepcopy(world)
    query = PerceptionQuery(Milk, whole_scene_region, view, world)
    task = PerceptionTask(query=query, execution_type=ExecutionType.REAL)

    restored = from_json(to_json(task), **receiving_world_kwargs(receiving_world))

    assert restored.execution_type == ExecutionType.REAL
    assert restored.query.semantic_annotation is Milk
    assert restored.query.world is receiving_world
    assert restored.query.robot is receiving_world.get_world_entity_with_id_by_id(
        view.id
    )
    assert restored.query.region.origin.reference_frame is receiving_world.root


def test_perception_task_without_an_execution_type_is_rejected(
    immutable_model_world, whole_scene_region, rclpy_node
):
    """
    A chart built while nothing is executing the plan has no source to answer with,
    which has to be said plainly rather than silently defaulting to reading the world
    model.
    """
    world, view, context = immutable_model_world
    query = PerceptionQuery(Milk, whole_scene_region, view, world)
    task = PerceptionTask(query=query, execution_type=None)

    with pytest.raises(UnknownExecutionType):
        build_perception_task(task, world, rclpy_node)


def test_detecting_motion_takes_the_execution_type_of_the_environment(
    immutable_model_world, whole_scene_region
):
    """
    The motion is written once and run in both worlds, so which source answers it is
    decided by the environment executing the plan rather than by the plan itself.
    """
    world, view, context = immutable_model_world
    query = PerceptionQuery(Milk, whole_scene_region, view, world)
    plan = execute_single(DetectingMotion(query=query), context=context)

    with simulated_robot:
        executable = plan.parse()

    tasks = list(executable.motion_mappings.values())
    assert [type(task) for task in tasks] == [PerceptionTask]
    assert tasks[0].execution_type is ExecutionType.SIMULATED


def test_perception_task_survives_a_chart_round_trip(
    immutable_model_world, whole_scene_region
):
    """
    What actually crosses to the controller is the whole motion state chart as JSON
    text, not the task on its own, so the trip is only proven by making it as the chart.

    Going through :func:`json.dumps` is part of the point: a value that survives
    ``to_json`` but is not JSON at all would pass a round trip that skipped the text.
    """
    world, view, context = immutable_model_world
    receiving_world = deepcopy(world)
    query = PerceptionQuery(Milk, whole_scene_region, view, world)
    chart = MotionStatechart()
    chart.add_node(
        task := PerceptionTask(query=query, execution_type=ExecutionType.REAL)
    )
    chart.add_node(EndMotion.when_true(task))

    restored_chart = MotionStatechart.from_json(
        json.loads(json.dumps(chart.to_json())),
        **receiving_world_kwargs(receiving_world),
    )

    restored_tasks = [
        node for node in restored_chart.nodes if isinstance(node, PerceptionTask)
    ]
    assert len(restored_tasks) == 1
    assert restored_tasks[0].execution_type == ExecutionType.REAL
    assert restored_tasks[0].query.world is receiving_world
    assert restored_tasks[
        0
    ].query.robot is receiving_world.get_world_entity_with_id_by_id(view.id)


def test_detection_in_a_chart_corrects_a_reach_planned_before_it(
    immutable_model_world, whole_scene_region, rclpy_node, robokudo_query_server
):
    """
    Why perception belongs in the chart at all: a reach compiled alongside the detection
    still binds its goal when it starts, so it follows the object to where the detection
    put it rather than to the pose the plan was expanded against.
    """
    world, view, context = immutable_model_world
    milk_body = world.get_body_by_name("milk.stl")
    query = PerceptionQuery(Milk, whole_scene_region, view, world)
    detection = PerceptionTask(query=query, execution_type=ExecutionType.REAL)
    reach = CartesianPose(
        root_link=world.root,
        tip_link=view.right_arm.end_effector.tool_frame,
        goal_pose=Pose(reference_frame=milk_body),
        name="MoveTCP",
    )
    build_context = build_perception_task(detection, world, rclpy_node)
    reach.build(build_context)

    run_perception_task(detection, build_context)
    reach.on_start(build_context)

    np.testing.assert_allclose(
        reach.root_T_goal_reference_frame.to_position().evaluate().flatten()[:3],
        PERCEIVED_MILK_POSITION,
        atol=1e-9,
    )
