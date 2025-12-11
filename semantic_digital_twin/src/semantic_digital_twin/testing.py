import os
import threading
import time

import pytest
from krrood.entity_query_language.symbol_graph import SymbolGraph
from typing_extensions import Tuple

from .adapters.urdf import URDFParser
from .datastructures.prefixed_name import PrefixedName
from .spatial_types import HomogeneousTransformationMatrix
from .spatial_types.derivatives import DerivativeMap
from .spatial_types.spatial_types import Vector3
from .utils import rclpy_installed, tracy_installed, hsrb_installed
from .world import World
from .world_description.connections import (
    Connection6DoF,
    PrismaticConnection,
    RevoluteConnection,
    FixedConnection,
    OmniDrive,
)
from .world_description.degree_of_freedom import DegreeOfFreedom
from .world_description.geometry import Box, Scale, Sphere
from .world_description.shape_collection import ShapeCollection
from .world_description.world_entity import Body


@pytest.fixture
def world_setup() -> Tuple[
    World,
    Body,
    Body,
    Body,
    Body,
    Body,
]:
    world = World()
    root = Body(name=PrefixedName(name="root", prefix="world"))
    l1 = Body(name=PrefixedName("l1"))
    l2 = Body(name=PrefixedName("l2"))
    bf = Body(name=PrefixedName("bf"))
    r1 = Body(name=PrefixedName("r1"))
    r2 = Body(name=PrefixedName("r2"))

    with world.modify_world():
        [world.add_kinematic_structure_entity(b) for b in [root, l1, l2, bf, r1, r2]]
        lower_limits = DerivativeMap()
        lower_limits.velocity = -1
        upper_limits = DerivativeMap()
        upper_limits.velocity = 1
        dof = DegreeOfFreedom(
            name=PrefixedName("dof"),
            lower_limits=lower_limits,
            upper_limits=upper_limits,
        )
        world.add_degree_of_freedom(dof)

        c_l1_l2 = PrismaticConnection(
            parent=l1, child=l2, dof_id=dof.id, axis=Vector3.X(reference_frame=l1)
        )
        c_r1_r2 = RevoluteConnection(
            parent=r1, child=r2, dof_id=dof.id, axis=Vector3.Z(reference_frame=r1)
        )
        bf_root_l1 = FixedConnection(parent=bf, child=l1)
        bf_root_r1 = FixedConnection(parent=bf, child=r1)
        world.add_connection(c_l1_l2)
        world.add_connection(c_r1_r2)
        world.add_connection(bf_root_l1)
        world.add_connection(bf_root_r1)
        c_root_bf = Connection6DoF.create_with_dofs(parent=root, child=bf, world=world)
        world.add_connection(c_root_bf)

    return world, l1, l2, bf, r1, r2


@pytest.fixture
def world_setup_simple():
    world = World()
    root = Body(name=PrefixedName(name="root", prefix="world"))
    body1 = Body(
        name=PrefixedName("name1", prefix="test"),
        collision=ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(),
                    scale=Scale(0.25, 0.25, 0.25),
                )
            ]
        ),
    )
    body2 = Body(
        name=PrefixedName("name2", prefix="test"),
        collision=ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(),
                    scale=Scale(0.25, 0.25, 0.25),
                )
            ]
        ),
    )
    body3 = Body(
        name=PrefixedName("name3", prefix="test"),
        collision=ShapeCollection(
            [Sphere(origin=HomogeneousTransformationMatrix.from_xyz_rpy(), radius=0.01)]
        ),
    )

    body4 = Body(
        name=PrefixedName("name4", prefix="test"),
        collision=ShapeCollection(
            [Sphere(origin=HomogeneousTransformationMatrix.from_xyz_rpy(), radius=0.01)]
        ),
    )

    with world.modify_world():
        world.add_kinematic_structure_entity(body1)
        world.add_kinematic_structure_entity(body2)
        world.add_kinematic_structure_entity(body3)
        world.add_kinematic_structure_entity(body4)

        c_root_body1 = Connection6DoF.create_with_dofs(
            parent=root, child=body1, world=world
        )
        c_root_body2 = Connection6DoF.create_with_dofs(
            parent=root, child=body2, world=world
        )
        c_root_body3 = Connection6DoF.create_with_dofs(
            parent=root, child=body3, world=world
        )
        c_root_body4 = Connection6DoF.create_with_dofs(
            parent=root, child=body4, world=world
        )

        world.add_connection(c_root_body1)
        world.add_connection(c_root_body2)
        world.add_connection(c_root_body3)
        world.add_connection(c_root_body4)
    return world, body1, body2, body3, body4


@pytest.fixture
def two_arm_robot_world():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "urdf"
    )
    robot = os.path.join(urdf_dir, "simple_two_arm_robot.urdf")
    world = World()
    with world.modify_world():
        localization_body = Body(name=PrefixedName("odom_combined"))
        world.add_kinematic_structure_entity(localization_body)

        robot_parser = URDFParser.from_file(file_path=robot)
        world_with_robot = robot_parser.parse()
        # world_with_pr2.plot_kinematic_structure()
        root = world_with_robot.root
        c_root_bf = OmniDrive.create_with_dofs(
            parent=localization_body, child=root, world=world
        )
        world.merge_world(world_with_robot, root_connection=c_root_bf)
    return world


@pytest.fixture
def pr2_world():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "urdf"
    )
    pr2 = os.path.join(urdf_dir, "pr2_kinematic_tree.urdf")
    pr2_parser = URDFParser.from_file(file_path=pr2)
    world_with_pr2 = pr2_parser.parse()
    with world_with_pr2.modify_world():
        pr2_root = world_with_pr2.root
        localization_body = Body(name=PrefixedName("odom_combined"))
        world_with_pr2.add_kinematic_structure_entity(localization_body)
        c_root_bf = OmniDrive.create_with_dofs(
            parent=localization_body, child=pr2_root, world=world_with_pr2
        )
        world_with_pr2.add_connection(c_root_bf)

    return world_with_pr2


@pytest.fixture
def tracy_world():
    if not tracy_installed():
        pytest.skip("Tracy not installed")
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "urdf"
    )
    tracy = os.path.join(urdf_dir, "tracy.urdf")
    world = World()
    with world.modify_world():
        localization_body = Body(name=PrefixedName("odom_combined"))
        world.add_kinematic_structure_entity(localization_body)

        tracy_parser = URDFParser.from_file(file_path=tracy)
        world_with_tracy = tracy_parser.parse()
        # world_with_tracy.plot_kinematic_structure()
        tracy_root = world_with_tracy.root
        c_root_bf = Connection6DoF.create_with_dofs(
            parent=localization_body, child=tracy_root, world=world
        )
        world.merge_world(world_with_tracy, c_root_bf)

    return world


@pytest.fixture
def hsrb_world():
    if not hsrb_installed():
        pytest.skip("HSRB not installed")
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "urdf"
    )
    hsrb = os.path.join(urdf_dir, "hsrb.urdf")
    world = World()
    with world.modify_world():
        localization_body = Body(name=PrefixedName("odom_combined"))
        world.add_kinematic_structure_entity(localization_body)

        hsrb_parser = URDFParser.from_file(file_path=hsrb)
        world_with_hsrb = hsrb_parser.parse()
        hsrb_root = world_with_hsrb.root
        c_root_bf = Connection6DoF.create_with_dofs(
            parent=localization_body, child=hsrb_root, world=world
        )
        world.merge_world(world_with_hsrb, c_root_bf)

    return world


@pytest.fixture
def apartment_world() -> World:
    """
    Return the apartment world parsed from the URDF file.
    """
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "urdf"
    )
    apartment = os.path.join(urdf_dir, "apartment.urdf")
    parser = URDFParser.from_file(file_path=apartment)
    world = parser.parse()
    world.validate()
    return world


@pytest.fixture(scope="function")
def rclpy_node():
    if not rclpy_installed():
        pytest.skip("ROS not installed")
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init()
    node = rclpy.create_node("test_node")

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
    thread.start()
    time.sleep(0.1)
    try:
        yield node
    finally:
        # Stop executor cleanly and wait for the thread to exit
        executor.shutdown()
        thread.join(timeout=2.0)

        # Remove the node from the executor and destroy it
        # (executor.shutdown() takes care of spinning; add_node is safe to keep as-is)
        node.destroy_node()

        # Shut down the ROS client library
        rclpy.shutdown()


@pytest.fixture()
def kitchen_world():
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "urdf", "kitchen-small.urdf"
    )
    parser = URDFParser.from_file(file_path=path)
    world = parser.parse()
    world.validate()
    return world
