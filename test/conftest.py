import os
import threading
import time
from copy import deepcopy

import numpy as np
import pytest
from typing_extensions import Type

from krrood.class_diagrams import ClassDiagram
from krrood.entity_query_language.predicate import Symbol
from krrood.entity_query_language.symbol_graph import SymbolGraph
from krrood.ontomatic.property_descriptor.attribute_introspector import (
    DescriptorAwareIntrospector,
)
from krrood.utils import recursive_subclasses
from pycram.datastructures.dataclasses import Context  # type: ignore
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.utils import rclpy_installed, tracy_installed
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    DiffDrive,
    FixedConnection,
    Connection6DoF,
)
from semantic_digital_twin.world_description.geometry import Box, Scale, Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    CollisionCheckingConfig,
)

###############################
### Fixture Usage Guide #######
###############################

"""
This file composes the Pytest fixtures for all tests in the monorepo. 
Some basic facts about fixtures:
 * conftest.py and test directories inherit the fixtures from contests from upper directories 
 * Fixtures can be scoped, possible values are: function, class, module, package, session 
 * Fixtures can use other fixtures 
    * This only works the requesting fixture is scoped less or equal the the requested fixture (function requesting session works)
 * The return/yield value of fixtures is being cached. 
    * In case of worlds this is important since always the same world is returned by a session scoped fixture
    
General Remarks:
    * Apparently generating the robot semantic view takes some time so it should be done in the session scoped setup
    
The structure of fixtures in this conftest: 
    * World setup fixtures: 
        These setup a world and return it, they are scoped for a whole session
         THEY SHOULD NOT BE USED DIRECTLY! 
         This is the case since the returned world will always be the same and changing the state or model of it will 
         cause side effects in other tests. They can only be used if the test does not change the state or model
    * Merging setup fixtures: 
        These combine one or more different worlds from other fixtures. 
        THEY SHOULD NOT BE USED DIRECTLY 
        for the same reason as the world setup fixtures. 
    * State reset fixture:
        This fixtures are scoped for functions and reset the state between tests runs. 
        THEY CAN BE USED FOR TESTS THAT ONLY CHANGE THE STATE
        Since resetting the state between test runs is easy this is the most common fixture for tests 
    * Model changing fixtures: 
        This fixtures are scoped for functions, they are a copy of the world from the world setup and are discarded 
        after the test since there is no good method to reset the model after a test has changed it. 

"""


@pytest.fixture(autouse=True, scope="function")
def cleanup_after_test():
    # We need to pass the class diagram, since otherwise some names are not found anymore after clearing the symbol graph
    # for the first time, since World is not a symbol
    SymbolGraph().clear()
    class_diagram = ClassDiagram(
        recursive_subclasses(Symbol) + [World],
        introspector=DescriptorAwareIntrospector(),
    )
    SymbolGraph(_class_diagram=class_diagram)
    # runs BEFORE each test
    yield
    # runs AFTER each test (even if the test fails or errors)
    SymbolGraph().clear()


@pytest.fixture(autouse=True, scope="session")
def cleanup_ros():
    """
    Fixture to ensure that ROS is properly cleaned up after all tests.
    """
    if os.environ.get("ROS_VERSION") == "2":
        import rclpy

        if not rclpy.ok():
            rclpy.init()
    yield
    if os.environ.get("ROS_VERSION") == "2":
        if rclpy.ok():
            rclpy.shutdown()


#############################################
############### Worlds ######################
#############################################


@pytest.fixture()
def cylinder_bot_world():
    robot_world = World()
    with robot_world.modify_world():
        robot = Body(
            name=PrefixedName("bot"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.1, height=0.5)]),
            collision_config=CollisionCheckingConfig(
                buffer_zone_distance=0.05, violated_distance=0.0, max_avoided_bodies=3
            ),
        )
        robot_world.add_body(robot)
        MinimalRobot.from_world(robot_world)
    world = World()
    with world.modify_world():
        body = Body(
            name=PrefixedName("map"),
        )
        environment = Body(
            name=PrefixedName("environment"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.5)]),
        )
        env_connection = FixedConnection(
            parent=body,
            child=environment,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                1
            ),
        )
        world.add_connection(env_connection)

        connection = OmniDrive.create_with_dofs(
            world=world, parent=body, child=robot_world.root
        )
        world.merge_world(robot_world, connection)
        connection.has_hardware_interface = True

    return world


@pytest.fixture()
def cylinder_bot_diff_world():
    robot_world = World()
    with robot_world.modify_world():
        robot = Body(
            name=PrefixedName("bot"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.1, height=0.5)]),
            collision_config=CollisionCheckingConfig(
                buffer_zone_distance=0.05, violated_distance=0.0, max_avoided_bodies=3
            ),
        )
        robot_world.add_body(robot)
        MinimalRobot.from_world(robot_world)
    world = World()
    with world.modify_world():
        body = Body(
            name=PrefixedName("map"),
        )
        environment = Body(
            name=PrefixedName("environment"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.5)]),
        )
        env_connection = FixedConnection(
            parent=body,
            child=environment,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                1
            ),
        )
        world.add_connection(env_connection)

        connection = DiffDrive.create_with_dofs(
            world=world, parent=body, child=robot_world.root
        )
        world.merge_world(robot_world, connection)
        connection.has_hardware_interface = True

    return world


def world_with_urdf_factory(
    urdf_path: str,
    robot_semantic_annotation: Type[AbstractRobot] | None,
    drive_connection_type: Type[OmniDrive],
):
    """
    Builds this tree:
    map -> odom_combined -> "urdf tree"
    """
    urdf_parser = URDFParser.from_file(file_path=urdf_path)
    world_with_urdf = urdf_parser.parse()
    if robot_semantic_annotation is not None:
        robot_semantic_annotation.from_world(world_with_urdf)

    with world_with_urdf.modify_world():
        map = Body(name=PrefixedName("map"))
        localization_body = Body(name=PrefixedName("odom_combined"))

        map_C_localization = Connection6DoF.create_with_dofs(
            world_with_urdf, map, localization_body
        )
        world_with_urdf.add_connection(map_C_localization)

        if drive_connection_type is OmniDrive:
            c_root_bf = OmniDrive.create_with_dofs(
                parent=localization_body,
                child=world_with_urdf.root,
                world=world_with_urdf,
            )
        world_with_urdf.add_connection(c_root_bf)

    return world_with_urdf


@pytest.fixture(scope="session")
def pr2_world_setup():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "pycram",
        "resources",
        "robots",
    )
    pr2 = os.path.join(urdf_dir, "pr2_calibrated_with_ft.urdf")
    return world_with_urdf_factory(pr2, PR2, OmniDrive)


@pytest.fixture(scope="session")
def hsr_world_setup():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "pycram",
        "resources",
        "robots",
    )
    hsr = os.path.join(urdf_dir, "hsrb.urdf")
    return world_with_urdf_factory(hsr, HSRB, OmniDrive)


@pytest.fixture(scope="session")
def tracy_world():
    if not tracy_installed():
        pytest.skip("Tracy not installed")
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "semantic_digital_twin",
        "resources",
        "urdf",
    )
    tracy = os.path.join(urdf_dir, "tracy.urdf")
    tracy_parser = URDFParser.from_file(file_path=tracy)
    world_with_tracy = tracy_parser.parse()
    Tracy.from_world(world_with_tracy)
    return world_with_tracy


@pytest.fixture(scope="session")
def stretch_world():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "pycram",
        "resources",
        "robots",
    )
    stretch = os.path.join(urdf_dir, "stretch_description.urdf")
    return world_with_urdf_factory(stretch, Stretch, OmniDrive)


@pytest.fixture(scope="session")
def tiago_world():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "pycram",
        "resources",
        "robots",
    )
    tiago = os.path.join(urdf_dir, "tiago_dual.urdf")
    return world_with_urdf_factory(tiago, Tiago, OmniDrive)


@pytest.fixture(scope="session")
def apartment_world_setup():
    apartment_world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "pycram",
            "resources",
            "worlds",
            "apartment.urdf",
        )
    ).parse()
    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "pycram",
            "resources",
            "objects",
            "milk.stl",
        )
    ).parse()
    cereal_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "pycram",
            "resources",
            "objects",
            "breakfast_cereal.stl",
        )
    ).parse()
    apartment_world.merge_world_at_pose(
        milk_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.05, reference_frame=apartment_world.root
        ),
    )
    apartment_world.merge_world_at_pose(
        cereal_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 1.8, 1.05, reference_frame=apartment_world.root
        ),
    )
    milk_view = Milk(root=apartment_world.get_body_by_name("milk.stl"))
    with apartment_world.modify_world():
        apartment_world.add_semantic_annotation(milk_view)

    return apartment_world


@pytest.fixture(scope="session")
def simple_apartment_setup():
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)

        box = Body(
            name=PrefixedName("box"),
            collision=ShapeCollection([Box(scale=Scale(1, 1, 1))]),
            visual=ShapeCollection([Box(scale=Scale(1, 1, 1))]),
        )

        box_2 = Body(
            name=PrefixedName("box_2"),
            collision=ShapeCollection([Box(scale=Scale(1, 1, 1))]),
            visual=ShapeCollection([Box(scale=Scale(1, 1, 1))]),
        )

        box_1_connection = FixedConnection(
            parent=world.root,
            child=box,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                2, 0, 0.5, reference_frame=world.root
            ),
        )
        box_2_connection = FixedConnection(
            parent=root,
            child=box_2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                -2, 0, 0.5
            ),
        )

        wall1 = Body(
            name=PrefixedName("wall_1"),
            collision=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
            visual=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
        )
        wall2 = Body(
            name=PrefixedName("wall_2"),
            collision=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
            visual=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
        )
        wall3 = Body(
            name=PrefixedName("wall_3"),
            collision=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
            visual=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
        )
        wall4 = Body(
            name=PrefixedName("wall_4"),
            collision=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
            visual=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
        )

        wall_1_connection = FixedConnection(
            parent=root,
            child=wall1,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                0, -4, 1
            ),
        )
        wall_2_connection = FixedConnection(
            parent=root,
            child=wall2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                0, 4, 1
            ),
        )
        wall_3_connection = FixedConnection(
            parent=root,
            child=wall3,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                -4, 0, 1, yaw=np.pi / 2
            ),
        )
        wall_4_connection = FixedConnection(
            parent=root,
            child=wall4,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                4, 0, 1, yaw=np.pi / 2
            ),
        )

        world.add_connection(box_1_connection)
        world.add_connection(box_2_connection)
        world.add_connection(wall_1_connection)
        world.add_connection(wall_2_connection)
        world.add_connection(wall_3_connection)
        world.add_connection(wall_4_connection)

    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "pycram",
            "resources",
            "objects",
            "milk.stl",
        )
    ).parse()
    world.merge_world_at_pose(
        milk_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi),
    )
    return world


@pytest.fixture(scope="session")
def kitchen_world():
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "semantic_digital_twin",
        "resources",
        "urdf",
        "kitchen-small.urdf",
    )
    parser = URDFParser.from_file(file_path=path)
    world = parser.parse()
    world.validate()
    return world


@pytest.fixture(scope="session")
def pr2_apartment_world(pr2_world_setup, apartment_world_setup):
    """
    Builds this tree:
    map -> odom_combined -> pr2 urdf tree
        -> apartment urdf
    """
    pr2_copy = deepcopy(pr2_world_setup)
    PR2.from_world(pr2_copy)  # semantic annotations are lost on copy

    apartment_copy = deepcopy(apartment_world_setup)

    pr2_copy.merge_world(apartment_copy)
    pr2_copy.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(1.3, 2, 0)
    )
    return pr2_copy


@pytest.fixture(scope="session")
def simple_pr2_world_setup(pr2_world_setup, simple_apartment_setup):
    apartment_world = deepcopy(simple_apartment_setup)
    pr2_copy = deepcopy(pr2_world_setup)
    robot_view = PR2.from_world(pr2_copy)
    pr2_copy.merge_world(apartment_world)

    return pr2_copy, robot_view, Context(pr2_copy, robot_view)


@pytest.fixture(scope="session")
def hsr_apartment_world(hsr_world_setup, apartment_world_setup):
    apartment_copy = deepcopy(apartment_world_setup)
    hsr_copy = deepcopy(hsr_world_setup)
    robot_view = HSRB.from_world(hsr_copy)

    apartment_copy.merge_world_at_pose(
        hsr_copy, HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
    )

    return apartment_copy, robot_view, Context(apartment_copy, robot_view)


@pytest.fixture(scope="session")
def stretch_apartment_world(stretch_world_setup, apartment_world_setup):
    apartment_copy = deepcopy(apartment_world_setup)
    stretch_copy = deepcopy(stretch_world_setup)

    apartment_copy.merge_world_at_pose(
        stretch_copy, HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
    )

    return apartment_copy


###############################
######## World with reset #####
###############################


@pytest.fixture
def pr2_world_state_reset(pr2_world_setup):
    world = deepcopy(pr2_world_setup)
    PR2.from_world(world)
    state = deepcopy(world.state.data)
    yield world
    world.state.data = state


###############################
######### Utils ###############
###############################


@pytest.fixture(scope="function")
def rclpy_node():
    """
    You can use this fixture if you want to use the marker visualizer of semDT and need a ros node.
    ..warning::
        This fixture can not be used in multiple tests at once
    """
    if not rclpy_installed():
        pytest.skip("ROS not installed")
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    if not rclpy.ok():
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
