"""
Tests for detecting that an agent has taken hold of an object: a gripper holding
something touches it with the hand, and the grasp is named by the tool frame that
carries it.
"""

from __future__ import annotations

from giskardpy.motion_statechart.context import MotionStatechartContext
from typing_extensions import List, Tuple, Type

from segmind.datastructures.events import (
    DetectionEvent,
    GraspEvent,
    LossOfGraspEvent,
    PickUpEvent,
    PlacingEvent,
    SupportEvent,
)
from segmind.detectors.base import AbstractDetector, SegmindContext
from segmind.detectors.coarse_event_detector_nodes import (
    PickUpDetector,
    PlacingDetector,
)
from segmind.detectors.spatial_relation_detector_nodes import (
    LossOfSupportDetector,
    SupportDetector,
)
from segmind.detectors.grasp_detector_nodes import GraspDetector, LossOfGraspDetector
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

BOX_SIZE = 0.05
"""
The edge length of the box a gripper holds in these tests.
"""

CARRIED_AWAY = 5.0
"""
How far a test moves the box to take it out of every gripper's reach.
"""


def _left_gripper(world: World) -> EndEffector:
    """
    :return: The end effector whose palm is the left one.
    """
    [gripper] = [
        end_effector
        for end_effector in world.get_semantic_annotations_by_type(EndEffector)
        if end_effector.root.name.name.startswith("l_")
    ]
    return gripper


def _box_at(world: World, position: Tuple[float, float, float]) -> Body:
    """
    Add a box free to move to ``world``, standing at ``position``.
    """
    box = Body(
        name=PrefixedName("held_box"),
        collision=ShapeCollection([Box(scale=Scale(BOX_SIZE, BOX_SIZE, BOX_SIZE))]),
    )
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=world.root, child=box)
        )
    _move_to(box, position)
    return box


def _move_to(box: Body, position: Tuple[float, float, float]) -> None:
    """
    Stand ``box`` at ``position``, in the world's own frame.
    """
    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        *position, reference_frame=box.parent_connection.parent
    )


def _box_in_the_hand_of(world: World, gripper: EndEffector) -> Body:
    """
    Add a box free to move to ``world``, standing where ``gripper``'s tool frame is.

    That is where a held object sits: between the fingers, clear of the wrist and the
    rest of the arm.
    """
    return _box_at(world, gripper.tool_frame.numeric_global_pose.position)


def _executor_for(
    world: World, detectors: List[AbstractDetector]
) -> EpisodeSegmenterExecutor:
    executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    executor.compile(SegmindStatechart().build_statechart(detectors))
    return executor


def _events_of(
    executor: EpisodeSegmenterExecutor, event_type: Type[DetectionEvent], body: Body
) -> List[DetectionEvent]:
    return [
        event
        for event in executor.context.require_extension(
            SegmindContext
        ).logger.get_events()
        if type(event) is event_type and event.tracked_object is body
    ]


# %% taking hold of something


def test_a_body_a_gripper_has_hold_of_is_grasped(pr2_world_copy):
    """
    The grasp names the tool frame the object is carried by, though what touches the
    object is the hand around it.
    """
    gripper = _left_gripper(pr2_world_copy)
    box = _box_in_the_hand_of(pr2_world_copy, gripper)

    executor = _executor_for(pr2_world_copy, [GraspDetector(tracked_object=box)])
    executor.tick()

    [grasp] = _events_of(executor, GraspEvent, box)
    assert grasp.with_object is gripper.tool_frame


def test_a_body_the_gripper_no_longer_holds_is_let_go_of(pr2_world_copy):
    gripper = _left_gripper(pr2_world_copy)
    box = _box_in_the_hand_of(pr2_world_copy, gripper)
    executor = _executor_for(
        pr2_world_copy,
        [GraspDetector(tracked_object=box), LossOfGraspDetector(tracked_object=box)],
    )
    executor.tick()

    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        CARRIED_AWAY, CARRIED_AWAY, CARRIED_AWAY, reference_frame=pr2_world_copy.root
    )
    executor.tick()

    [let_go] = _events_of(executor, LossOfGraspEvent, box)
    assert let_go.with_object is gripper.tool_frame


# %% what a grasp makes of a pick-up and a placing

TABLE_SIZE = 0.3
"""
The edge length of the surface a test rests the held box on.
"""

RESTING_PLACE = (1.0, 1.0, 0.5)
"""
Where a test stands the box before any hand reaches it, clear of the robot.
"""

IN_THE_AIR = (1.0, 1.0, 1.5)
"""
Where a test holds the box clear of every hand and every surface.
"""

MOVED_ASIDE = 0.3
"""
How far a test moves the box to take it out of the hand, which is far enough to clear
every part of the gripper.
"""


def _surface_under(world: World, box: Body, aside: float = 0.0) -> Body:
    """
    Add a fixed surface to ``world``, its top where the underside of ``box`` is.

    :param aside: How far to the side of ``box`` the surface stands, so that the box
        reaches it only once it is moved that far.
    """
    surface = Body(
        name=PrefixedName("surface"),
        collision=ShapeCollection(
            [Box(scale=Scale(TABLE_SIZE, TABLE_SIZE, TABLE_SIZE))]
        ),
    )
    x, y, _ = box.numeric_global_pose.position
    x += aside
    bottom = box.numeric_global_bounds.lower[2]
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=surface,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x, y, bottom - TABLE_SIZE / 2, reference_frame=world.root
                ),
            )
        )
    return surface


def _raise_by(box: Body, height: float) -> None:
    """
    Move ``box`` straight up by ``height``.
    """
    x, y, z = box.numeric_global_pose.position
    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x, y, z + height, reference_frame=box.parent_connection.parent
    )


def test_an_object_is_picked_up_when_an_agent_lifts_it_off_what_it_rested_on(
    pr2_world_copy,
):
    """
    A pick-up is one agent taking hold of an object and lifting it: the object rests on
    something, is grasped, and leaves what held it up.
    """
    gripper = _left_gripper(pr2_world_copy)
    box = _box_at(pr2_world_copy, RESTING_PLACE)
    _surface_under(pr2_world_copy, box)
    executor = _executor_for(
        pr2_world_copy,
        [
            GraspDetector(tracked_object=box),
            SupportDetector(tracked_object=box),
            LossOfSupportDetector(tracked_object=box),
            PickUpDetector(use_grasp_logic=True),
        ],
    )

    executor.tick()
    assert _events_of(executor, PickUpEvent, box) == []

    _move_to(box, gripper.tool_frame.numeric_global_pose.position)
    executor.tick()

    assert len(_events_of(executor, PickUpEvent, box)) == 1


def test_an_object_is_placed_where_the_agent_let_go_of_it(pr2_world_copy):
    """
    A placing is where the object was released, so a surface it has not been let go
    onto is not somewhere it was put down.
    """
    gripper = _left_gripper(pr2_world_copy)
    box = _box_in_the_hand_of(pr2_world_copy, gripper)
    surface = _surface_under(pr2_world_copy, box, aside=MOVED_ASIDE)
    executor = _executor_for(
        pr2_world_copy,
        [
            GraspDetector(tracked_object=box),
            LossOfGraspDetector(tracked_object=box),
            SupportDetector(tracked_object=box),
            PlacingDetector(use_grasp_logic=True),
        ],
    )

    executor.tick()
    assert _events_of(executor, PlacingEvent, box) == []

    # out of the hand and onto the surface beside it
    x, y, z = box.numeric_global_pose.position
    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x + MOVED_ASIDE, y, z, reference_frame=pr2_world_copy.root
    )
    executor.tick()

    [placing] = _events_of(executor, PlacingEvent, box)
    assert placing.with_object is surface


def test_a_held_object_does_not_come_to_rest_on_what_it_brushes(pr2_world_copy):
    """
    An object being carried is held, so a surface it touches on the way does not become
    something it rests on.
    """
    gripper = _left_gripper(pr2_world_copy)
    box = _box_in_the_hand_of(pr2_world_copy, gripper)
    _surface_under(pr2_world_copy, box)
    executor = _executor_for(
        pr2_world_copy,
        [GraspDetector(tracked_object=box), SupportDetector(tracked_object=box)],
    )

    executor.tick()

    assert _events_of(executor, SupportEvent, box) == []


def test_taking_hold_again_mid_carry_is_not_a_second_pick_up(pr2_world_copy):
    """
    One loss of what held an object up is one pick-up: a hand that loses its grip and
    takes hold again while carrying has not picked the object up a second time.
    """
    gripper = _left_gripper(pr2_world_copy)
    box = _box_at(pr2_world_copy, RESTING_PLACE)
    _surface_under(pr2_world_copy, box)
    executor = _executor_for(
        pr2_world_copy,
        [
            GraspDetector(tracked_object=box),
            LossOfGraspDetector(tracked_object=box),
            SupportDetector(tracked_object=box),
            LossOfSupportDetector(tracked_object=box),
            PickUpDetector(use_grasp_logic=True),
        ],
    )
    held = gripper.tool_frame.numeric_global_pose.position

    executor.tick()
    _move_to(box, held)
    executor.tick()
    assert len(_events_of(executor, PickUpEvent, box)) == 1

    # the grip is lost and taken again, with nothing holding the box up in between
    _move_to(box, IN_THE_AIR)
    executor.tick()
    _move_to(box, held)
    executor.tick()

    assert len(_events_of(executor, PickUpEvent, box)) == 1
