"""
Tests for what detectors check an object against: the robot is left out, so what the
robot does to an object is not read as something the scene did to it, and a gripper is
never what an object rests on.
"""

from __future__ import annotations

from giskardpy.motion_statechart.context import MotionStatechartContext
from typing_extensions import List

from semantic_digital_twin.reasoning.predicates import contact
from segmind.datastructures.events import SupportEvent
from segmind.detectors.base import SegmindContext
from segmind.datastructures.events import ContactEvent
from segmind.detectors.atomic_event_detectors_nodes import ContactDetector
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

BOX_SIZE = 0.05
"""
The edge length of the box a test rests on something.
"""

SUNK_INTO_WHAT_IT_RESTS_ON = 0.01
"""
How far a test sinks a resting box into what it rests on, so the two touch.
"""

SUNK_INTO_A_GRIPPER = 0.03
"""
How far a test sinks the box into a gripper.

A gripper's bounding box stands well above the hand itself, so a box balanced on that
box would hang in the air beside it rather than being held.
"""


def _top_of(body: Body) -> float:
    """
    :return: How high the top of ``body``'s collision geometry stands in the world.
    """
    world = body._world
    boxes = body.collision.as_bounding_box_collection_in_frame(world.root)
    return max(box.max_z for box in boxes)


def _bottom_of(body: Body) -> float:
    """
    :return: How low the bottom of ``body``'s collision geometry hangs in the world.
    """
    world = body._world
    boxes = body.collision.as_bounding_box_collection_in_frame(world.root)
    return min(box.min_z for box in boxes)


def _box_resting_on(
    world: World, supporter: Body, sunk_by: float = SUNK_INTO_WHAT_IT_RESTS_ON
) -> Body:
    """
    Add a box free to move to ``world``, standing on top of ``supporter``.

    :param sunk_by: How far the box reaches into ``supporter``, so that the two touch
        rather than only their bounding boxes meeting.
    """
    box = Body(
        name=PrefixedName("resting_box"),
        collision=ShapeCollection([Box(scale=Scale(BOX_SIZE, BOX_SIZE, BOX_SIZE))]),
    )
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=world.root, child=box)
        )
    x, y, _ = supporter.global_pose.to_position().to_np()[:3]
    top = _top_of(supporter)
    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x,
        y,
        top + BOX_SIZE / 2 - sunk_by,
        reference_frame=world.root,
    )
    return box


def _supporters_detected_for(
    world: World, box: Body, exclude_robot: bool = True
) -> List[Body]:
    executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    executor.compile(
        SegmindStatechart().build_statechart(
            [SupportDetector(tracked_object=box, exclude_robot=exclude_robot)]
        )
    )
    executor.tick()
    segmind_context = executor.context.require_extension(SegmindContext)
    return [
        event.with_object
        for event in segmind_context.logger.get_events()
        if isinstance(event, SupportEvent) and event.tracked_object is box
    ]


def test_an_object_resting_on_the_environment_is_supported_by_it(cylinder_bot_world):
    environment = cylinder_bot_world.get_body_by_name("environment")
    box = _box_resting_on(cylinder_bot_world, environment)

    assert _supporters_detected_for(cylinder_bot_world, box) == [environment]


def test_an_object_on_a_gripper_is_not_supported_by_the_gripper(pr2_world_copy):
    """
    A gripper holds what it touches, which is what the detector has to disregard, so the
    box is stood where the hand really reaches it: one the hand does not touch would be
    disregarded for want of contact instead. With the robot left out there would be
    nothing to disregard, so the run is the one that reads the robot too.
    """
    palm = pr2_world_copy.get_body_by_name("l_gripper_palm_link")
    box = _box_resting_on(pr2_world_copy, palm, sunk_by=SUNK_INTO_A_GRIPPER)
    assert contact(box, palm)
    gripper_bodies = {
        body
        for end_effector in pr2_world_copy.get_semantic_annotations_by_type(EndEffector)
        for body in end_effector.bodies
    }

    assert gripper_bodies.isdisjoint(
        _supporters_detected_for(pr2_world_copy, box, exclude_robot=False)
    )


# %% the robot is not part of the scene a detector reads


def _contacts_detected_for(
    world: World, box: Body, exclude_robot: bool = True
) -> List[Body]:
    executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    executor.compile(
        SegmindStatechart().build_statechart(
            [ContactDetector(tracked_object=box, exclude_robot=exclude_robot)]
        )
    )
    executor.tick()
    segmind_context = executor.context.require_extension(SegmindContext)
    return [
        event.with_object
        for event in segmind_context.logger.get_events()
        if isinstance(event, ContactEvent) and event.tracked_object is box
    ]


def test_nothing_of_the_robot_is_checked_against_an_object(pr2_world_copy):
    """
    With the robot left out, a hand around an object is not read as the scene touching
    it.
    """
    palm = pr2_world_copy.get_body_by_name("l_gripper_palm_link")
    box = _box_resting_on(pr2_world_copy, palm, sunk_by=SUNK_INTO_A_GRIPPER)

    assert _contacts_detected_for(pr2_world_copy, box) == []


def test_the_robot_is_checked_against_an_object_when_it_is_not_left_out(pr2_world_copy):
    """
    The robot is left out because a run is asked to read the scene, not the robot; a run
    that wants it back says so.
    """
    palm = pr2_world_copy.get_body_by_name("l_gripper_palm_link")
    box = _box_resting_on(pr2_world_copy, palm, sunk_by=SUNK_INTO_A_GRIPPER)

    assert palm in _contacts_detected_for(pr2_world_copy, box, exclude_robot=False)
