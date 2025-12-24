import logging
import os
import threading
import time
from copy import deepcopy

import numpy as np
import pytest

from krrood.class_diagrams import ClassDiagram
from krrood.entity_query_language.predicate import Symbol
from krrood.entity_query_language.symbol_graph import SymbolGraph
from krrood.ontomatic.property_descriptor.attribute_introspector import (
    DescriptorAwareIntrospector,
)
from krrood.utils import recursive_subclasses
from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.utils import rclpy_installed, tracy_installed
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    FixedConnection,
    Connection6DoF,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


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

#
# def pytest_configure(conf):
#     logging.getLogger("pycram").setLevel(logging.DEBUG)
#     logging.getLogger("giskardpy").setLevel(logging.DEBUG)
#     logging.getLogger("krrood").setLevel(logging.DEBUG)
#     logging.getLogger("semantic_digital_twin").setLevel(logging.DEBUG)


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
        PR2.from_world(world_with_pr2)

    return world_with_pr2


#############################################
############### Worlds ######################
#############################################


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
    hsr_parser = URDFParser.from_file(file_path=hsr)
    world_with_hsr = hsr_parser.parse()
    with world_with_hsr.modify_world():
        hsr_root = world_with_hsr.root
        localization_body = Body(name=PrefixedName("odom_combined"))
        world_with_hsr.add_kinematic_structure_entity(localization_body)
        c_root_bf = OmniDrive.create_with_dofs(
            parent=localization_body, child=hsr_root, world=world_with_hsr
        )
        world_with_hsr.add_connection(c_root_bf)

    return world_with_hsr


@pytest.fixture
def tracy_world():
    if not tracy_installed():
        pytest.skip("Tracy not installed")
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "resources", "urdf"
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
    milk_view = Milk(body=apartment_world.get_body_by_name("milk.stl"))
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
        )

        box_2 = Body(
            name=PrefixedName("box_2"),
            collision=ShapeCollection([Box(scale=Scale(1, 1, 1))]),
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
        )
        wall2 = Body(
            name=PrefixedName("wall_2"),
            collision=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
        )
        wall3 = Body(
            name=PrefixedName("wall_3"),
            collision=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
        )
        wall4 = Body(
            name=PrefixedName("wall_4"),
            collision=ShapeCollection([Box(scale=Scale(8, 0.1, 2))]),
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
        milk_world, HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07)
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
    pr2_copy = deepcopy(pr2_world_setup)
    apartment_copy = deepcopy(apartment_world_setup)

    apartment_copy.merge_world_at_pose(
        pr2_copy,
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            1.3, 2, 0, reference_frame=apartment_copy.root
        ),
    )
    return apartment_copy


@pytest.fixture(scope="session")
def simple_pr2_world_setup(pr2_world_setup, simple_apartment_setup):
    world = deepcopy(pr2_world_setup)
    apartment_world = deepcopy(simple_apartment_setup)
    world.merge_world(apartment_world)

    robot_view = PR2.from_world(world)
    return world, robot_view, Context(world, robot_view)


@pytest.fixture(scope="session")
def hsr_apartment_world(hsr_world_setup, apartment_world_setup):
    apartment_copy = deepcopy(apartment_world_setup)
    hsr_copy = deepcopy(hsr_world_setup)

    apartment_copy.merge_world_at_pose(
        hsr_copy, HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
    )

    robot_view = HSRB.from_world(hsr_copy)
    return apartment_copy, robot_view, Context(hsr_copy, robot_view)


###############################
######## World with reset #####
###############################


@pytest.fixture
def pr2_world_state_reset(pr2_world_setup):
    world = deepcopy(pr2_world_setup)
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
