from copy import deepcopy

import numpy as np

from semantic_digital_twin.reasoning.predicates import (
    contact,
    visible,
    Above,
    Below,
    LeftOf,
    RightOf,
    Behind,
    InFrontOf,
    is_body_in_region,
    occluding_bodies,
    is_supported_by,
    reachable,
)
from semantic_digital_twin.reasoning.robot_predicates import (
    robot_in_collision,
    robot_holds_body,
    blocking,
    is_body_in_gripper,
)
from semantic_digital_twin.robots.abstract_robot import Camera, ParallelGripper
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.testing import *
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Region


@pytest.fixture(scope="function")
def two_block_world():
    def make_body(name: str) -> Body:
        result = Body(name=PrefixedName(name))
        collision = Box(
            scale=Scale(1.0, 1.0, 1.0),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=result),
        )
        result.collision = ShapeCollection([collision], reference_frame=result)
        return result

    world = World()

    body_1 = make_body("body_1")
    body_2 = make_body("body_2")

    with world.modify_world():
        connection = FixedConnection(
            parent=body_1,
            child=body_2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=3, reference_frame=body_1
            ),
        )
        world.add_connection(connection)
    return body_1, body_2


def test_in_contact():
    w = World()

    b1 = Body(name=PrefixedName("b1"))
    collision1 = Box(
        scale=Scale(1.0, 1.0, 1.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            0,
            0,
            0.0,
            0,
            0,
            0,
            reference_frame=b1,
        ),
        color=Color(1.0, 0.0, 0.0),
    )
    b1.collision = ShapeCollection([collision1])

    b2 = Body(name=PrefixedName("b2"))
    collision2 = Box(
        scale=Scale(1.0, 1.0, 1.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            0.9, 0, 0.0, 0, 0, 0, reference_frame=b2
        ),
        color=Color(0.0, 1.0, 0.0),
    )
    b2.collision = ShapeCollection([collision2])

    b3 = Body(name=PrefixedName("b3"))
    collision3 = Box(
        scale=Scale(1.0, 1.0, 1.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            1.8, 0, 0.0, 0, 0, 0, reference_frame=b3
        ),
        color=Color(0.0, 0.0, 1.0),
    )
    b3.collision = ShapeCollection([collision3])

    with w.modify_world():
        w.add_kinematic_structure_entity(b1)
        w.add_kinematic_structure_entity(b2)
        w.add_kinematic_structure_entity(b3)
        w.add_connection(Connection6DoF.create_with_dofs(parent=b1, child=b2, world=w))
        w.add_connection(Connection6DoF.create_with_dofs(parent=b2, child=b3, world=w))
    assert contact(b1, b2)
    assert not contact(b1, b3)
    assert contact(b2, b3)


def test_robot_in_contact(pr2_world_state_reset: World):
    pr2: PR2 = PR2.from_world(pr2_world_state_reset)

    body = Body(name=PrefixedName("test_body"))
    collision1 = Box(
        scale=Scale(1.0, 1.0, 1.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            z=0.5,
            reference_frame=body,
        ),
        color=Color(1.0, 0.0, 0.0),
    )
    body.collision = ShapeCollection([collision1])

    with pr2_world_state_reset.modify_world():
        pr2_world_state_reset.add_connection(
            Connection6DoF.create_with_dofs(
                parent=pr2_world_state_reset.root,
                child=body,
                world=pr2_world_state_reset,
            )
        )

    # Ensure the call runs without raising
    assert robot_in_collision(pr2)

    body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        4, 0, 0.5, 0, 0, 0, pr2_world_state_reset.root
    )
    assert not robot_in_collision(pr2)


def test_get_visible_objects(pr2_world_state_reset: World):

    pr2: PR2 = PR2.from_world(pr2_world_state_reset)

    body = Body(name=PrefixedName("test_body"))
    collision1 = Box(
        scale=Scale(1.0, 1.0, 1.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=2.0,
            z=1.0,
            reference_frame=body,
        ),
        color=Color(1.0, 0.0, 0.0),
    )
    body.collision = ShapeCollection([collision1])

    with pr2_world_state_reset.modify_world():
        pr2_world_state_reset.add_connection(
            Connection6DoF.create_with_dofs(
                parent=pr2_world_state_reset.root,
                child=body,
                world=pr2_world_state_reset,
            )
        )

    camera = pr2_world_state_reset.get_semantic_annotations_by_type(Camera)[0]

    assert visible(camera, body)


def test_occluding_bodies(pr2_world_state_reset: World):
    world = deepcopy(pr2_world_state_reset)
    world.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(0, 0, 0)
    )
    pr2: PR2 = PR2.from_world(world)

    def make_body(name: str) -> Body:
        result = Body(name=PrefixedName(name))
        collision = Box(
            scale=Scale(1.0, 1.0, 1.0),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=result),
        )
        result.collision = ShapeCollection([collision])
        return result

    obstacle = make_body("obstacle")
    occluded_body = make_body("occluded_body")

    with world.modify_world():
        root = world.root
        c1 = FixedConnection(
            parent=root,
            child=obstacle,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=root, x=3, z=0.8
            ),
        )
        c2 = FixedConnection(
            parent=root,
            child=occluded_body,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=root, x=10, z=0.5
            ),
        )
        world.add_connection(c1)
        world.add_connection(c2)

    camera = world.get_semantic_annotations_by_type(Camera)[0]

    bodies = occluding_bodies(camera, occluded_body)
    assert obstacle in bodies
    assert camera not in bodies
    assert occluded_body not in bodies


def test_above_and_below(two_block_world):
    center, top = two_block_world

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=-3)
    assert Above(top, center, pov)()
    assert Below(center, top, pov)()

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=3, yaw=np.pi)
    assert Above(top, center, pov)()
    assert Below(center, top, pov)()

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=3, roll=np.pi)
    assert Above(center, top, pov)()
    assert Below(top, center, pov)()


def test_left_and_right(two_block_world):
    center, top = two_block_world

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=3, roll=np.pi / 2)
    assert LeftOf(top, center, pov)()
    assert RightOf(center, top, pov)()

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=3, roll=-np.pi / 2)
    assert RightOf(top, center, pov)()
    assert LeftOf(center, top, pov)()


def test_behind_and_in_front_of(two_block_world):
    center, top = two_block_world

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(z=-5, pitch=np.pi / 2)
    assert Behind(top, center, pov)()
    assert InFrontOf(center, top, pov)()

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(z=5, pitch=-np.pi / 2)
    assert InFrontOf(top, center, pov)()
    assert Behind(center, top, pov)()


def test_body_in_region(two_block_world):
    center, top = two_block_world
    region = Region(name=PrefixedName("test_region"))
    region_box = Box(
        scale=Scale(1.0, 1.0, 1.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=region),
    )
    region.area = ShapeCollection([region_box])

    with center._world.modify_world():
        connection = FixedConnection(
            parent=center,
            child=region,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=0.5, reference_frame=center
            ),
        )
        center._world.add_connection(connection)
    assert is_body_in_region(center, region) == 0.5
    assert is_body_in_region(top, region) == 0.0


def test_supporting(two_block_world):
    center, top = two_block_world

    with center._world.modify_world():
        top.parent_connection.parent_T_connection_expression = (
            HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=center, z=1.0)
        )
    assert is_supported_by(top, center)
    assert not is_supported_by(center, top)


def test_is_body_in_gripper(
    pr2_world_state_reset,
):
    pr2: PR2 = PR2.from_world(pr2_world_state_reset)

    gripper = pr2_world_state_reset.get_semantic_annotations_by_type(ParallelGripper)

    left_gripper = (
        gripper[0]
        if LeftOf(gripper[0].root, gripper[1].root, pr2.root.global_pose)()
        else gripper[1]
    )

    # Create krrood_test box between fingers
    test_box = Body(name=PrefixedName("test_box"))
    box_collision = Box(
        scale=Scale(0.05, 0.01, 0.05),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=test_box),
        color=Color(1.0, 0.0, 0.0),
    )
    test_box.collision = ShapeCollection([box_collision])

    # Calculate position between fingers
    finger1_pos = (
        left_gripper.finger.tip.collision.center_of_mass_in_world().to_vector3()
    )
    finger2_pos = (
        left_gripper.thumb.tip.collision.center_of_mass_in_world().to_vector3()
    )
    between_fingers = (finger1_pos + finger2_pos) / 2.0

    # Add box to world
    with pr2_world_state_reset.modify_world():
        root = pr2_world_state_reset.root
        connection = Connection6DoF.create_with_dofs(
            parent=root,
            child=test_box,
            world=pr2_world_state_reset,
        )
        pr2_world_state_reset.add_connection(connection)
        connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=between_fingers[0],
            y=between_fingers[1],
            z=between_fingers[2],
            reference_frame=root,
        )

    assert is_body_in_gripper(test_box, left_gripper) > 0
    assert robot_holds_body(pr2, test_box)
    connection.origin = HomogeneousTransformationMatrix()
    assert is_body_in_gripper(test_box, left_gripper) == 0


def test_reachable(pr2_world_state_reset):
    pr2: PR2 = PR2.from_world(pr2_world_state_reset)

    tool_frame_T_reachable_goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-0.2,
        y=0.3,
        reference_frame=pr2.left_arm.manipulator.tool_frame,
    )

    assert reachable(
        tool_frame_T_reachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.manipulator.tool_frame,
    )
    assert not blocking(
        tool_frame_T_reachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.manipulator.tool_frame,
    )
    tool_frame_T_unreachable_goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=10, y=10, reference_frame=pr2.left_arm.manipulator.tool_frame
    )
    assert not reachable(
        tool_frame_T_unreachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.manipulator.tool_frame,
    )

    tool_frame_T_rotated_reachable_goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-0.2,
        y=0.3,
        yaw=np.pi / 2,
        reference_frame=pr2.left_arm.manipulator.tool_frame,
    )
    assert reachable(
        tool_frame_T_rotated_reachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.manipulator.tool_frame,
    )

    tool_frame_T_rotated_unreachable_goal = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-0.2,
            y=0.3,
            yaw=-np.pi / 2,
            reference_frame=pr2.left_arm.manipulator.tool_frame,
        )
    )
    assert not reachable(
        tool_frame_T_rotated_unreachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.manipulator.tool_frame,
    )


def test_blocking(pr2_world_state_reset):

    pr2: PR2 = PR2.from_world(pr2_world_state_reset)

    obstacle = Body(name=PrefixedName("obstacle"))
    collision = Box(
        scale=Scale(3.0, 1.0, 1.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, z=0.5),
    )
    obstacle.collision = ShapeCollection([collision])

    with pr2_world_state_reset.modify_world():
        new_root = Body(name=PrefixedName("new_root"))
        pr2_world_state_reset.add_connection(
            Connection6DoF.create_with_dofs(
                parent=new_root,
                child=pr2_world_state_reset.root,
                world=pr2_world_state_reset,
            )
        )
        pr2_world_state_reset.add_connection(
            Connection6DoF.create_with_dofs(
                parent=new_root,
                child=obstacle,
                world=pr2_world_state_reset,
            )
        )

    assert obstacle not in pr2.bodies
    assert robot_in_collision(pr2)

    tool_frame_T_reachable_goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-0.2,
        y=0.3,
        reference_frame=pr2.left_arm.manipulator.tool_frame,
    )
    assert blocking(
        tool_frame_T_reachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.manipulator.tool_frame,
    )
