from copy import deepcopy
from dataclasses import dataclass, fields
from typing import List
from uuid import UUID

import numpy as np

from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.reasoning.predicates import (
    Above,
    Below,
    LeftOf,
    RightOf,
    Behind,
    InFrontOf,
    InsideOf,
    InsideRegion,
    InContactWith,
    PlaceIsOccupied,
    Reachable,
    SupportedBy,
    Supports,
    ViewDependentSpatialRelation,
)
from krrood.entity_query_language.predicate import Predicate
from krrood.entity_query_language.testing.result_verification import (
    placeholder_operands,
)
from krrood.entity_query_language.verbalization.pipeline import (
    verbalize_expression,
)
from semantic_digital_twin.reasoning.robot_predicates import (
    robot_in_collision,
    robot_holds_body,
    blocking,
    is_body_in_gripper,
    bodies_in_gripper,
    is_pose_free_for_robot,
    is_gripper_holding_something,
    occluding_bodies,
    VisibleTo,
)
from semantic_digital_twin.robots.robot_parts import Camera, EndEffector, TCamera
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.testing import *
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
    Color,
    Sphere,
    VolumetricBoundingBox,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.semantic_annotations.mixins import TKinematicStructureEntity
from semantic_digital_twin.world_description.world_entity import (
    Body,
    Region,
    KinematicStructureEntity,
    TBody,
    TRegion,
)

BALL_RADIUS = 0.5
"""
The radius of the ball a test stands a body beside, inside its bounding box.
"""

CONTAINER_FLOOR_THICKNESS = 0.05
"""
How thick the floor of the container a test stands a body in is.
"""


def _supported_by_default(field_name: str) -> float:
    """
    :return: The default :class:`SupportedBy` gives the tolerance ``field_name``.
    """
    return next(
        dataclass_field.default
        for dataclass_field in fields(SupportedBy)
        if dataclass_field.name == field_name
    )


RESTING_CONTACT_TOLERANCE = _supported_by_default("contact_tolerance")
"""
How far above a surface a body may stand and still rest on it, as the predicate defaults it.
"""

CONTAINER_WALL_HEIGHT = 0.5
"""
How high the walls of that container rise, which is what carries its middle above the
body standing on its floor.
"""


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
    assert InContactWith(b1, b2)()
    assert not InContactWith(b1, b3)()
    assert InContactWith(b2, b3)()


def test_robot_in_contact(pr2_world_copy: World):
    pr2 = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
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

    with pr2_world_copy.modify_world():
        pr2_world_copy.add_connection(
            Connection6DoF.create_with_dofs(
                parent=pr2_world_copy.root,
                child=body,
                world=pr2_world_copy,
            )
        )

    # Ensure the call runs without raising
    assert robot_in_collision(pr2)

    body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        4, 0, 0.5, 0, 0, 0, pr2_world_copy.root
    )
    assert not robot_in_collision(pr2)


def test_get_visible_objects(pr2_world_copy: World):
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

    with pr2_world_copy.modify_world():
        pr2_world_copy.add_connection(
            Connection6DoF.create_with_dofs(
                parent=pr2_world_copy.root,
                child=body,
                world=pr2_world_copy,
            )
        )

    camera = pr2_world_copy.get_semantic_annotations_by_type(Camera)[0]

    assert VisibleTo(entity=body, camera=camera)()


def test_camera_view_frame_x_axis_is_the_forward_axis(pr2_world_copy: World):
    """
    The frame the ray tracer casts along must be the camera's own frame, turned so that
    its x axis is the direction the camera looks.
    """
    camera = pr2_world_copy.get_semantic_annotations_by_type(Camera)[0]
    root_T_camera = camera.root.global_transform

    root_T_view = camera.root_T_forward_view.to_np()
    root_V_forward = (
        root_T_camera.rotation_matrix @ camera.forward_facing_axis
    ).to_np()

    assert np.allclose(root_T_view[:3, 0], root_V_forward.flatten()[:3], atol=1e-9)
    assert np.allclose(root_T_view[:3, 3], root_T_camera.to_np()[:3, 3], atol=1e-9)


def test_visibility_follows_camera_orientation(pr2_world_copy: World):
    """
    A body off to the side is visible exactly when the head is turned towards it.
    """
    body = Body(name=PrefixedName("test_body"))
    body.collision = ShapeCollection(
        [
            Box(
                scale=Scale(1.0, 1.0, 1.0),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=2.0, z=1.0, reference_frame=body
                ),
            )
        ]
    )

    with pr2_world_copy.modify_world():
        pr2_world_copy.add_connection(
            Connection6DoF.create_with_dofs(
                parent=pr2_world_copy.root,
                child=body,
                world=pr2_world_copy,
            )
        )

    camera = pr2_world_copy.get_semantic_annotations_by_type(Camera)[0]
    head_pan = pr2_world_copy.get_degree_of_freedom_by_name("head_pan_joint")

    assert not VisibleTo(entity=body, camera=camera)()

    pr2_world_copy.state[head_pan.id].position = np.pi / 2
    pr2_world_copy.notify_state_change()

    assert VisibleTo(entity=body, camera=camera)()


def test_occluding_bodies(pr2_world_state_reset: World):
    world = deepcopy(pr2_world_state_reset)
    world.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(0, 0, 0)
    )

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


def test_occluding_bodies_follows_camera_orientation(pr2_world_state_reset: World):
    """
    Occlusion is judged along the direction the camera looks, not along a fixed world
    axis, so a pair off to the side is only resolved once the head is turned towards it.
    """
    world = deepcopy(pr2_world_state_reset)
    world.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(0, 0, 0)
    )

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
        world.add_connection(
            FixedConnection(
                parent=root,
                child=obstacle,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=root, y=3, z=0.8
                ),
            )
        )
        world.add_connection(
            FixedConnection(
                parent=root,
                child=occluded_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=root, y=10, z=0.5
                ),
            )
        )

    camera = world.get_semantic_annotations_by_type(Camera)[0]
    head_pan = world.get_degree_of_freedom_by_name("head_pan_joint")
    world.state[head_pan.id].position = np.pi / 2
    world.notify_state_change()

    assert obstacle in occluding_bodies(camera, occluded_body)


def test_above_and_below(two_block_world):
    center, top = two_block_world

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=-3)
    assert Above(top.center_of_mass, center.center_of_mass, pov)()
    assert Below(center.center_of_mass, top.center_of_mass, pov)()

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=3, yaw=np.pi)
    assert Above(top.center_of_mass, center.center_of_mass, pov)()
    assert Below(center.center_of_mass, top.center_of_mass, pov)()

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=3, roll=np.pi)
    assert Above(center.center_of_mass, top.center_of_mass, pov)()
    assert Below(top.center_of_mass, center.center_of_mass, pov)()


def test_left_and_right(two_block_world):
    center, top = two_block_world

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=3, roll=np.pi / 2)
    assert LeftOf(top.center_of_mass, center.center_of_mass, pov)()
    assert RightOf(center.center_of_mass, top.center_of_mass, pov)()

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(x=3, roll=-np.pi / 2)
    assert RightOf(top.center_of_mass, center.center_of_mass, pov)()
    assert LeftOf(center.center_of_mass, top.center_of_mass, pov)()


def test_behind_and_in_front_of(two_block_world):
    center, top = two_block_world

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(z=-5, pitch=np.pi / 2)
    assert Behind(top.center_of_mass, center.center_of_mass, pov)()
    assert InFrontOf(center.center_of_mass, top.center_of_mass, pov)()

    pov = HomogeneousTransformationMatrix.from_xyz_rpy(z=5, pitch=-np.pi / 2)
    assert InFrontOf(top.center_of_mass, center.center_of_mass, pov)()
    assert Behind(center.center_of_mass, top.center_of_mass, pov)()


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
    assert InsideRegion(center, region).compute_contained_fraction() == 0.5
    assert InsideRegion(top, region).compute_contained_fraction() == 0.0
    assert InsideRegion(center, region)()
    assert not InsideRegion(top, region)()


def test_supporting(two_block_world):
    center, top = two_block_world

    with center._world.modify_world():
        top.parent_connection.parent_T_connection_expression = (
            HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=center, z=1.0)
        )
    assert SupportedBy(top, center)()
    assert not SupportedBy(center, top)()


def test_is_body_in_gripper(pr2_world_copy):
    pr2 = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]

    gripper = pr2_world_copy.get_semantic_annotations_by_type(EndEffector)

    left_gripper = (
        gripper[0]
        if LeftOf(
            gripper[0].root.center_of_mass,
            gripper[1].root.center_of_mass,
            pr2.root.global_transform,
        )()
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
    finger1_pos = left_gripper.finger.tip.collision.center_of_mass_in_world().vector3
    finger2_pos = left_gripper.thumb.tip.collision.center_of_mass_in_world().vector3
    between_fingers = (finger1_pos + finger2_pos) / 2.0

    # Add box to world
    with pr2_world_copy.modify_world():
        root = pr2_world_copy.root
        connection = Connection6DoF.create_with_dofs(
            parent=root,
            child=test_box,
            world=pr2_world_copy,
        )
        pr2_world_copy.add_connection(connection)
        connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=between_fingers[0],
            y=between_fingers[1],
            z=between_fingers[2],
            reference_frame=root,
        )

    assert is_body_in_gripper(test_box, left_gripper) > 0
    assert robot_holds_body(pr2, test_box)
    connection.origin = HomogeneousTransformationMatrix(reference_frame=root)
    assert is_body_in_gripper(test_box, left_gripper) == 0


def test_reachable(pr2_world_state_reset, rclpy_node):
    pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]

    tool_frame_T_reachable_goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-0.2,
        y=0.3,
        reference_frame=pr2.left_arm.end_effector.tool_frame,
    )

    assert Reachable(
        tool_frame_T_reachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.end_effector.tool_frame,
    )()
    assert not blocking(
        tool_frame_T_reachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.end_effector.tool_frame,
    )
    tool_frame_T_unreachable_goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=10, y=10, reference_frame=pr2.left_arm.end_effector.tool_frame
    )
    assert not Reachable(
        tool_frame_T_unreachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.end_effector.tool_frame,
    )()

    tool_frame_T_rotated_reachable_goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-0.2,
        y=0.3,
        yaw=np.pi / 2,
        reference_frame=pr2.left_arm.end_effector.tool_frame,
    )
    assert Reachable(
        tool_frame_T_rotated_reachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.end_effector.tool_frame,
    )()

    tool_frame_T_rotated_unreachable_goal = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-0.2,
            y=0.3,
            yaw=-np.pi / 2,
            reference_frame=pr2.left_arm.end_effector.tool_frame,
        )
    )
    assert not Reachable(
        tool_frame_T_rotated_unreachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.end_effector.tool_frame,
    )()


def test_blocking(pr2_world_copy):
    pr2 = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    obstacle = Body(name=PrefixedName("obstacle"))
    collision = Box(
        scale=Scale(3.0, 1.0, 1.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.0, z=0.5, reference_frame=obstacle
        ),
    )
    obstacle.collision = ShapeCollection([collision])
    obstacle.visual = ShapeCollection([collision])

    with pr2_world_copy.modify_world():
        pr2_world_copy.add_connection(
            Connection6DoF.create_with_dofs(
                parent=pr2_world_copy.root,
                child=obstacle,
                world=pr2_world_copy,
            )
        )

    assert obstacle not in pr2.bodies
    assert robot_in_collision(pr2)

    tool_frame_T_reachable_goal = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-0.2,
        y=0.3,
        reference_frame=pr2.left_arm.end_effector.tool_frame,
    )
    assert blocking(
        tool_frame_T_reachable_goal,
        pr2.left_arm.root,
        pr2.left_arm.end_effector.tool_frame,
    )


def test_region_is_occupied(pr2_world_state_reset):
    view = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]

    target_box = VolumetricBoundingBox(
        0, 0, 0, 1, 1, 1, HomogeneousTransformationMatrix()
    )
    assert not PlaceIsOccupied(
        target_box,
        Pose.from_xyz_rpy(2.5, 2, 0, reference_frame=pr2_world_state_reset.root),
        pr2_world_state_reset,
    )()

    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        3.5, 2.5, 0
    )
    pr2_world_state_reset.notify_state_change()

    assert PlaceIsOccupied(target_box, view.root.global_pose, pr2_world_state_reset)()

    assert not PlaceIsOccupied(
        target_box,
        Pose.from_xyz_rpy(3.5, 2.5, 1, 0, reference_frame=pr2_world_state_reset.root),
        pr2_world_state_reset,
        view.bodies_with_collision,
    )()


def test_is_pose_free_for_robot(pr2_apartment_state_reset):
    view = pr2_apartment_state_reset.get_semantic_annotations_by_type(PR2)[0]
    assert is_pose_free_for_robot(
        view,
        Pose.from_xyz_rpy(2, -2, 0, reference_frame=pr2_apartment_state_reset.root),
    )

    assert not is_pose_free_for_robot(
        view,
        Pose.from_xyz_rpy(3, 2, 0, reference_frame=pr2_apartment_state_reset.root),
    )

    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, -2, 0
    )

    assert is_pose_free_for_robot(
        view,
        Pose.from_xyz_rpy(2, -2, 0, reference_frame=pr2_apartment_state_reset.root),
    )

    assert is_pose_free_for_robot(
        view,
        Pose.from_xyz_rpy(2.1, -2.1, 0, reference_frame=pr2_apartment_state_reset.root),
    )


def test_is_pose_free_for_robot_with_robot_pose(pr2_apartment_state_reset):
    view = pr2_apartment_state_reset.get_semantic_annotations_by_type(PR2)[0]
    assert is_pose_free_for_robot(
        view,
        Pose.from_xyz_rpy(2, -2, 0, reference_frame=pr2_apartment_state_reset.root),
    )

    assert is_pose_free_for_robot(
        view,
        view.root.global_pose,
    )


def test_bodies_in_gripper(pr2_apartment_world):
    world = deepcopy(pr2_apartment_world)
    tcp = world.get_body_by_name("l_gripper_tool_frame")
    pr2 = world.get_semantic_annotations_by_type(PR2)[0]

    with world.modify_world():
        body = Body(
            name=PrefixedName("mock_milk"),
            collision=ShapeCollection([Box(scale=Scale(0.05, 0.05, 0.3))]),
        )

        connection = FixedConnection(tcp, body)
        world.add_connection(connection)

    pr2.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, -2, 0
    )

    bodies = bodies_in_gripper(pr2.left_arm.end_effector)

    assert len(bodies) == 1
    assert bodies[0].name.name == "mock_milk"
    assert bodies[0] == body


def test_empty_gripper_is_not_holding_something():

    @dataclass(eq=False)
    class ReviewEndEffector(EndEffector):
        """
        Minimal concrete EndEffector for predicate tests.
        """

        @property
        def approach_axis(self) -> Vector3:
            return Vector3.X(reference_frame=self.tool_frame)

        @property
        def closing_axis(self) -> Vector3:
            return Vector3.Y(reference_frame=self.tool_frame)

        def setup_hardware_interfaces(self):
            pass

        def setup_joint_states(self) -> List[JointState]:
            return []

        @classmethod
        def setup_default_configuration_in_world_below_robot_root(
            cls, robot_root: KinematicStructureEntity
        ):
            raise NotImplementedError

    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    palm = Body(name=PrefixedName("palm", prefix="review"))
    collision = Box(
        scale=Scale(),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=palm),
    )
    palm.collision = ShapeCollection([collision], reference_frame=palm)
    tool_frame = Body(name=PrefixedName("tool_frame", prefix="review"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(palm)
        world.add_kinematic_structure_entity(tool_frame)
        world.add_connection(FixedConnection(parent=root, child=palm))
        world.add_connection(FixedConnection(parent=palm, child=tool_frame))
        gripper = ReviewEndEffector(
            name=PrefixedName("gripper", prefix="review"),
            root=palm,
            tool_frame=tool_frame,
        )
        world.add_semantic_annotation(gripper)

    # nothing is attached below the tool frame -> the gripper holds nothing
    assert is_gripper_holding_something(gripper) is False


@dataclass(eq=False)
class ReviewCamera(Camera):
    """
    Minimal concrete Camera for predicate tests.
    """

    @property
    def forward_facing_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.root)

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ):
        raise NotImplementedError


def test_nothing_occludes_a_body_in_clear_line_of_sight():

    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    camera_body = Body(name=PrefixedName("camera_body", prefix="review"))
    target = Body(name=PrefixedName("target", prefix="review"))
    collision = Box(
        scale=Scale(),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=target),
    )
    target.collision = ShapeCollection([collision], reference_frame=target)
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(camera_body)
        world.add_kinematic_structure_entity(target)
        world.add_connection(
            FixedConnection(
                parent=root,
                child=camera_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=1.0, reference_frame=root
                ),
            )
        )
        world.add_connection(
            FixedConnection(
                parent=root,
                child=target,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=3.0, z=1.0, reference_frame=root
                ),
            )
        )
        camera = ReviewCamera(
            name=PrefixedName("camera", prefix="review"),
            root=camera_body,
            field_of_view=FieldOfView(horizontal_angle=0.99, vertical_angle=0.75),
        )
        world.add_semantic_annotation(camera)

    assert occluding_bodies(camera, target) == []


# %% a body resting on a surface without sinking into it


def _stand_on(center: Body, top: Body, gap: float = 0.0) -> None:
    """
    Stand ``top`` on ``center``, leaving ``gap`` between the faces that meet.
    """
    with center._world.modify_world():
        top.parent_connection.parent_T_connection_expression = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=center, z=1.0 + gap
            )
        )


def test_a_body_resting_within_the_contact_tolerance_is_supported(two_block_world):
    """
    A body set down on a surface comes to rest a hair above it, so a support judged by
    overlapping volume alone would never hold.
    """
    center, top = two_block_world
    _stand_on(center, top, gap=RESTING_CONTACT_TOLERANCE / 2)

    assert SupportedBy(top, center)()


def test_a_body_hovering_beyond_the_contact_tolerance_is_not_supported(two_block_world):
    center, top = two_block_world
    _stand_on(center, top, gap=RESTING_CONTACT_TOLERANCE * 2)

    assert not SupportedBy(top, center)()


def test_a_body_inside_another_s_bounding_box_but_not_touching_it_is_not_supported():
    """
    A body rests on what it touches. A large or hollow shape, such as a wall, has a
    bounding box enclosing a great deal of empty space, and a body standing in that
    space is held up by nothing.
    """
    world = World()
    ball = Body(name=PrefixedName("ball"))
    ball.collision = ShapeCollection(
        [
            Sphere(
                radius=BALL_RADIUS,
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=ball
                ),
            )
        ],
        reference_frame=ball,
    )
    beside_the_ball = Body(name=PrefixedName("beside_the_ball"))
    beside_the_ball.collision = ShapeCollection(
        [
            Box(
                scale=Scale(0.05, 0.05, 0.05),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=beside_the_ball
                ),
            )
        ],
        reference_frame=beside_the_ball,
    )
    corner = BALL_RADIUS * 0.9
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=ball,
                child=beside_the_ball,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    corner, corner, corner, reference_frame=ball
                ),
            )
        )

    assert not SupportedBy(beside_the_ball, ball)()


def test_a_body_standing_in_a_container_is_supported_by_it():
    """
    A body put inside a container rests on its floor, though the container's walls rise
    above the body and carry the container's own middle higher than the body's.
    """
    world = World()
    container = Body(name=PrefixedName("container"))
    container.collision = ShapeCollection(
        [
            Box(
                scale=Scale(1.0, 1.0, CONTAINER_FLOOR_THICKNESS),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=container
                ),
            ),
            *(
                Box(
                    scale=Scale(0.05, 1.0, CONTAINER_WALL_HEIGHT),
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        side * 0.5,
                        0.0,
                        CONTAINER_WALL_HEIGHT / 2,
                        reference_frame=container,
                    ),
                )
                for side in (-1, 1)
            ),
        ],
        reference_frame=container,
    )
    content = Body(name=PrefixedName("content"))
    content.collision = ShapeCollection(
        [
            Box(
                scale=Scale(0.05, 0.05, 0.05),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=content
                ),
            )
        ],
        reference_frame=content,
    )
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=container,
                child=content,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=CONTAINER_FLOOR_THICKNESS / 2 + 0.025, reference_frame=container
                ),
            )
        )

    assert SupportedBy(content, container)()


def test_bodies_a_gap_apart_are_in_contact_within_a_threshold_that_spans_it(
    two_block_world,
):
    """
    The threshold says how close counts as touching, so a gap narrower than it is
    contact.
    """
    center, top = two_block_world
    _stand_on(center, top, gap=RESTING_CONTACT_TOLERANCE / 2)

    assert InContactWith(center, top, maximum_distance=RESTING_CONTACT_TOLERANCE)()


def test_a_body_does_not_support_itself(two_block_world):
    """
    Asking whether a body rests on itself is asking a collision detector to check a body
    against itself, which it refuses.
    """
    center, _ = two_block_world

    assert not SupportedBy(center, center)()


# %% a body rests on what pushes it up


def _box_body(name: str, scale: Scale, identifier: UUID | None = None) -> Body:
    """
    A body shaped as one box of ``scale`` about its own origin.
    """
    body = (
        Body(name=PrefixedName(name))
        if identifier is None
        else Body(name=PrefixedName(name), id=identifier)
    )
    body.collision = ShapeCollection(
        [
            Box(
                scale=scale,
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=body
                ),
            )
        ],
        reference_frame=body,
    )
    return body


def _fix_to(
    parent: Body, child: Body, parent_T_child: HomogeneousTransformationMatrix
) -> None:
    """
    Fix ``child`` to ``parent`` at ``parent_T_child``.
    """
    with parent._world.modify_world():
        parent._world.add_connection(
            FixedConnection(
                parent=parent,
                child=child,
                parent_T_connection_expression=parent_T_child,
            )
        )


def _steepest_supporting_slope() -> float:
    """
    How steeply what a body rests on may slope, as the predicate defaults it.
    """
    return _supported_by_default("maximum_slope")


def test_a_body_touched_only_on_its_side_is_not_supported_by_what_touches_it():
    """
    A kerb touching the lower half of a crate's side pushes it sideways, not up, so the
    crate does not rest on it.
    """
    world = World()
    table = _box_body("table", Scale(2.0, 2.0, 0.1))
    crate = _box_body("crate", Scale(0.2, 0.2, 0.2))
    kerb = _box_body("kerb", Scale(0.1, 2.0, 0.06))
    with world.modify_world():
        world.add_kinematic_structure_entity(table)
    _fix_to(
        table,
        crate,
        HomogeneousTransformationMatrix.from_xyz_rpy(z=0.15, reference_frame=table),
    )
    _fix_to(
        table,
        kerb,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.15, z=0.08, reference_frame=table
        ),
    )

    assert not SupportedBy(crate, kerb)()


def _box_on_a_slope(slope: float) -> tuple[Body, Body]:
    """
    A box lying flat on a slab tilted by ``slope`` radians.

    :return: The box and the slab.
    """
    world = World()
    ground = _box_body("ground", Scale(0.01, 0.01, 0.01))
    slab = _box_body("slab", Scale(1.0, 1.0, 0.05))
    box = _box_body("box", Scale(0.1, 0.1, 0.1))
    with world.modify_world():
        world.add_kinematic_structure_entity(ground)
    _fix_to(
        ground,
        slab,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            z=1.0, pitch=slope, reference_frame=ground
        ),
    )
    _fix_to(
        slab,
        box,
        HomogeneousTransformationMatrix.from_xyz_rpy(z=0.075, reference_frame=slab),
    )
    return box, slab


def test_a_body_on_a_slope_gentler_than_the_steepest_allowed_is_supported():
    box, slab = _box_on_a_slope(_steepest_supporting_slope() / 2)

    assert SupportedBy(box, slab)()


def test_a_body_on_a_slope_steeper_than_the_steepest_allowed_is_not_supported():
    box, slab = _box_on_a_slope(_steepest_supporting_slope() * 1.5)

    assert not SupportedBy(box, slab)()


def test_a_body_standing_upside_down_is_supported_by_what_it_stands_on():
    """
    Up is the world's up: a body turned over rests on what is underneath it all the same.
    """
    world = World()
    table = _box_body("table", Scale(2.0, 2.0, 0.1))
    crate = _box_body("crate", Scale(0.2, 0.2, 0.2))
    with world.modify_world():
        world.add_kinematic_structure_entity(table)
    _fix_to(
        table,
        crate,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            z=0.15, roll=np.pi, reference_frame=table
        ),
    )

    assert SupportedBy(crate, table)()


@pytest.mark.parametrize(
    "crate_identifier, table_identifier",
    [(UUID(int=1), UUID(int=2)), (UUID(int=2), UUID(int=1))],
)
def test_a_body_rests_on_a_surface_whichever_of_the_two_is_checked_first(
    crate_identifier: UUID, table_identifier: UUID
):
    """
    A collision check lists the two bodies in an order of its own, which does not
    change what rests on what.
    """
    world = World()
    table = _box_body("table", Scale(2.0, 2.0, 0.1), table_identifier)
    crate = _box_body("crate", Scale(0.2, 0.2, 0.2), crate_identifier)
    with world.modify_world():
        world.add_kinematic_structure_entity(table)
    _fix_to(
        table,
        crate,
        HomogeneousTransformationMatrix.from_xyz_rpy(z=0.15, reference_frame=table),
    )

    assert SupportedBy(crate, table)()


def test_a_body_taller_than_the_deepest_allowed_clipping_is_supported_by_the_floor_of_a_container():
    """
    A tall body standing on a container's floor sinks into nothing, though the
    container's walls rise along the whole of it.
    """
    world = World()
    container = Body(name=PrefixedName("container"))
    container.collision = ShapeCollection(
        [
            Box(
                scale=Scale(1.0, 1.0, CONTAINER_FLOOR_THICKNESS),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=container
                ),
            ),
            *(
                Box(
                    scale=Scale(0.05, 1.0, CONTAINER_WALL_HEIGHT),
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        side * 0.5,
                        0.0,
                        CONTAINER_WALL_HEIGHT / 2,
                        reference_frame=container,
                    ),
                )
                for side in (-1, 1)
            ),
        ],
        reference_frame=container,
    )
    height = _supported_by_default("maximum_intersection_height") * 3
    bottle = _box_body("bottle", Scale(0.05, 0.05, height))
    with world.modify_world():
        world.add_kinematic_structure_entity(container)
    _fix_to(
        container,
        bottle,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            z=CONTAINER_FLOOR_THICKNESS / 2 + height / 2, reference_frame=container
        ),
    )

    assert SupportedBy(bottle, container)()


def test_a_body_sunk_deeper_than_the_deepest_allowed_clipping_is_not_supported():
    """
    Bodies sunk that far into each other are a clipping the simulation did not resolve,
    not one resting on the other.
    """
    world = World()
    table = _box_body("table", Scale(2.0, 2.0, 1.0))
    crate = _box_body("crate", Scale(0.5, 0.5, 0.5))
    deepest = _supported_by_default("maximum_intersection_height")
    with world.modify_world():
        world.add_kinematic_structure_entity(table)
    _fix_to(
        table,
        crate,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            z=0.5 + 0.25 - deepest * 1.5, reference_frame=table
        ),
    )

    assert not SupportedBy(crate, table)()


# %% a spatial relation is something a query can state


def test_a_view_dependent_spatial_relation_is_a_predicate():
    """
    A relation a statement asserts has to be a predicate rather than a bare symbol, so a
    query can state it and a search can be devised from it rather than only evaluating
    it after the fact.
    """
    assert issubclass(ViewDependentSpatialRelation, Predicate)


@pytest.mark.parametrize(
    "relation",
    [
        InContactWith,
        VisibleTo,
        Reachable,
        SupportedBy,
        Supports,
        InsideOf,
        InsideRegion,
        PlaceIsOccupied,
        ViewDependentSpatialRelation,
    ],
)
def test_every_relation_is_a_predicate(relation):
    """
    A relation belongs in the vocabulary a statement can assert, so it is a predicate
    even where what it reads is a measurement.
    """
    assert issubclass(relation, Predicate)


def test_containment_answers_whether_it_holds_rather_than_by_how_much(two_block_world):
    center, top = two_block_world

    assert InsideOf(top, center)() in (True, False)


def test_containment_reports_the_fraction_it_measured(two_block_world):
    """
    The judgement is a threshold over a measurement, and the measurement stays readable
    on its own for the callers that compare it against a threshold of their own.
    """
    center, top = two_block_world
    relation = InsideOf(top, center)

    assert relation.compute_containment_ratio() == pytest.approx(
        InsideOf(top, center).compute_containment_ratio()
    )


def test_the_threshold_is_what_turns_the_measurement_into_a_verdict(two_block_world):
    """
    The same pair reads either way depending only on how much containment is asked for,
    which is what makes the threshold the statement of intent rather than a tuned
    constant hidden in the caller.
    """
    center, top = two_block_world
    measured = InsideOf(top, center).compute_containment_ratio()

    assert InsideOf(top, center, minimum_containment_ratio=measured)()
    assert not InsideOf(top, center, minimum_containment_ratio=measured + 0.01)()


def test_support_relates_the_supported_thing_to_what_holds_it_up():
    supported = Body(name=PrefixedName("supported"))
    supporting = Body(name=PrefixedName("supporting"))

    relation = SupportedBy(supported=supported, supporting=supporting)

    assert relation.subject is supported
    assert relation.object is supporting


@pytest.mark.parametrize(
    "relation, type_parameters",
    [
        (InContactWith, [TBody]),
        (SupportedBy, [TBody]),
        (InsideRegion, [TBody, TRegion]),
        (VisibleTo, [TKinematicStructureEntity, TCamera]),
    ],
)
def test_a_relation_is_generic_in_the_kinds_of_thing_it_relates(
    relation, type_parameters
):
    """
    A relation leaves open which kind of body, region or camera it relates, up to the
    bound of each type parameter, so a narrower relation can bind them.
    """
    assert relation.get_generic_type_parameters() == type_parameters


# %% how a relation reads


def test_reachability_reads_the_pose_as_what_is_reachable():
    """
    Reachability is stated about the pose, by the tip that has to arrive at it, so the
    pose is the subject of the sentence rather than the chain that reaches for it.
    """
    operands = placeholder_operands(Reachable)
    operands.update(Reachable._example_operand_values_())

    assert (
        verbalize_expression(Reachable(**operands))
        == "a HomogeneousTransformationMatrix is reachable by a Body"
    )


@pytest.mark.parametrize(
    "relation, sentence",
    [
        (SupportedBy, "a Body is supported by another Body"),
        (VisibleTo, "a Body or a Region is visible to a Camera"),
        (InContactWith, "a Body is in contact with another Body"),
        (Supports, "a Body is supporting a body"),
    ],
)
def test_a_relation_named_for_its_object_still_reads_as_a_sentence(relation, sentence):
    """
    A relation whose name is not verb-first cannot have its verb read off that name, so
    it states its own clause rather than inheriting one that renders ungrammatically.
    """
    operands = placeholder_operands(relation)
    operands.update(relation._example_operand_values_())

    assert verbalize_expression(relation(**operands)) == sentence
