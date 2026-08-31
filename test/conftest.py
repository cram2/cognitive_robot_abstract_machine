import gc
import os
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import objgraph
import pytest
from semantic_digital_twin.callbacks.callback import Callback
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)

from .orm_interface_build import ORM_BUILD_OPTION, OrmBuild
from .pytest_environment import PytestEnvironmentVariable

try:
    from semantic_digital_twin.robots.garmi import Garmi
except ImportError:
    Garmi = None

try:
    from coraplex.datastructures.dataclasses import Context
except ModuleNotFoundError:
    # ROS dependencies.
    Context = None

try:
    from giskardpy.middleware.ros2 import rospy
except ModuleNotFoundError:
    # ROS dependencies.
    rospy = None
from semantic_digital_twin.adapters.package_resolver import PathResolver
from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from typing_extensions import List, Type, TypeVar

CallbackT = TypeVar("CallbackT", bound=Callback)
"""
The kind of publisher a test started.
"""


from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.symbol_graph.symbol_graph import SymbolGraph, Symbol
from krrood.ontomatic.property_descriptor.attribute_introspector import (
    DescriptorAwareIntrospector,
)
from krrood.utils import recursive_subclasses
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.predetermined_maps.apartment_environment import (
    ApartmentEnvironment,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.robots.icub3 import ICub3
from semantic_digital_twin.robots.justin import Justin
from semantic_digital_twin.robots.mmp_dresden import MMPDresden
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    Table,
    Apple,
    Orange,
    Carrot,
    Lettuce,
    Banana,
    Spoon,
    Drawer,
    Handle,
    Elevator,
    Slider,
    Door,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
)
from semantic_digital_twin.utils import (
    rclpy_installed,
    tracy_installed,
    daisy_installed,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    DifferentialDrive,
    FixedConnection,
    Connection6DoF,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
    Cylinder,
    Sphere,
    Color,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
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


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Let a run state when it builds the ORM interfaces it reads.

    ..note:: Registering the option is what lets a run state it and read it in ``--help``.
        The build itself happens while pytest is still importing the conftests, before it
        has parsed anything, so :meth:`OrmBuild.requested` reads the choice off the
        arguments as they were given rather than off the parsed configuration.
    """
    parser.addoption(
        ORM_BUILD_OPTION,
        choices=OrmBuild.choices(),
        default=None,
        help=(
            "when to build the generated ORM interfaces; "
            f"'{OrmBuild.AUTO}' builds only what the checkout has not built since its "
            f"sources changed, '{OrmBuild.ALWAYS}' builds every run, whatever the "
            f"checkout holds, '{OrmBuild.NEVER}' builds nothing, and reads whatever the "
            "checkout holds"
        ),
    )


def pytest_configure(config):
    worker = os.environ.get(PytestEnvironmentVariable.XDIST_WORKER)

    if worker:
        worker_num = int(worker.removeprefix("gw"))
        os.environ["ROS_DOMAIN_ID"] = str(100 + worker_num)


@pytest.fixture(scope="session")
def _session_class_diagram() -> ClassDiagram:
    # We need to pass the class diagram, since otherwise some names are not found anymore
    # after clearing the symbol graph, since World is not a symbol. Built once per
    # session: the set of Symbol subclasses is static after collection, and the
    # SymbolGraph singleton reset below only needs to drop per-test instance state, not
    # this class-level graph.
    return ClassDiagram(
        recursive_subclasses(Symbol) + [World],
        introspector=DescriptorAwareIntrospector(),
    )


@pytest.fixture(autouse=True, scope="function")
def cleanup_after_test(_session_class_diagram):
    SymbolGraph.clear_instance()
    SymbolGraph(_class_diagram=_session_class_diagram)
    # runs BEFORE each test
    yield
    # runs AFTER each test (even if the test fails or errors)
    SymbolGraph.clear_instance()


@pytest.fixture(autouse=True, scope="module")
def count_worlds():
    yield
    gc.collect()
    world_in_mem = objgraph.count("World")
    if world_in_mem > 30:
        raise MemoryError(
            "Something is leaking worlds, there are more than 20 worlds in memory after the test"
        )


#############################################
############### Worlds ######################
#############################################


@pytest.fixture()
def world_with_two_bodies() -> tuple[World, Body, Body]:
    """A fresh world with an unconnected parent and child body, ready to receive a connection."""
    world = World()
    parent = Body(name=PrefixedName("parent"))
    child = Body(name=PrefixedName("child"))
    return world, parent, child


@pytest.fixture()
def cylinder_bot_world():
    robot_world = World()
    with robot_world.modify_world():
        robot = Body(
            name=PrefixedName("bot"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.1, height=0.5)]),
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

        environment2 = Body(
            name=PrefixedName("environment2"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.5, height=0.5)]),
        )
        env_connection2 = FixedConnection(
            parent=body,
            child=environment2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                y=0.5
            ),
        )
        world.add_connection(env_connection2)

        connection = OmniDrive.create_with_dofs(
            world=world, parent=body, child=robot_world.root
        )
        world.merge_world(robot_world, connection)
        connection.has_hardware_interface = True

        world.collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(2, {robot})
        )

    return world


@pytest.fixture()
def self_collision_bot_world():
    world = World()
    with world.modify_world():
        robot = Body(
            name=PrefixedName("map"),
            collision=ShapeCollection(shapes=[Sphere(radius=0.1)]),
            visual=ShapeCollection(shapes=[Sphere(radius=0.1)]),
        )
        l_shoulder = Body(
            name=PrefixedName("l_shoulder"),
            collision=ShapeCollection(shapes=[Sphere(radius=0.1)]),
            visual=ShapeCollection(shapes=[Sphere(radius=0.1)]),
        )
        l_tip = Body(
            name=PrefixedName("l_tip"),
            collision=ShapeCollection(shapes=[Sphere(radius=0.1)]),
            visual=ShapeCollection(shapes=[Sphere(radius=0.1)]),
        )
        l_thumb = Body(
            name=PrefixedName("l_thumb"),
            collision=ShapeCollection(shapes=[Sphere(radius=0.1)]),
            visual=ShapeCollection(shapes=[Sphere(radius=0.1)]),
        )
        r_shoulder = Body(
            name=PrefixedName("r_shoulder"),
            collision=ShapeCollection(shapes=[Sphere(radius=0.1)]),
            visual=ShapeCollection(shapes=[Sphere(radius=0.1)]),
        )
        r_tip = Body(
            name=PrefixedName("r_tip"),
            collision=ShapeCollection(shapes=[Sphere(radius=0.1)]),
            visual=ShapeCollection(shapes=[Sphere(radius=0.1)]),
        )
        r_thumb = Body(
            name=PrefixedName("r_thumb"),
            collision=ShapeCollection(shapes=[Sphere(radius=0.1)]),
            visual=ShapeCollection(shapes=[Sphere(radius=0.1)]),
        )

        world.add_connection(
            RevoluteConnection.create_with_dofs(
                parent=robot,
                child=l_shoulder,
                axis=Vector3.Z(),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.2, y=0.2
                ),
                world=world,
            )
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                parent=l_shoulder,
                child=l_tip,
                axis=Vector3.Z(),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.2
                ),
                world=world,
            )
        )
        world.add_connection(
            FixedConnection(
                parent=l_tip,
                child=l_thumb,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=-0.05, z=0.1
                ),
            )
        )

        world.add_connection(
            RevoluteConnection.create_with_dofs(
                parent=robot,
                child=r_shoulder,
                axis=Vector3.Z(),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.2,
                    y=-0.2,
                ),
                world=world,
            )
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                parent=r_shoulder,
                child=r_tip,
                axis=Vector3.Z(),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.2
                ),
                world=world,
            )
        )
        world.add_connection(
            FixedConnection(
                parent=r_tip,
                child=r_thumb,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=0.05, z=0.1
                ),
            )
        )
        MinimalRobot.from_world(world)

    return world


@pytest.fixture()
def supported_abstract_robots():
    return [
        PR2,
        Tiago,
        Justin,
        HSRB,
        Tracy,
        Stretch,
        Armar7,
        ICub3,
        UnitreeG1,
        MMPDresden,
        DAiSy,
        # Garmi, We dont have the ROS Package yet
    ]


@pytest.fixture()
def cylinder_bot_diff_world():
    robot_world = World()
    with robot_world.modify_world():
        robot = Body(
            name=PrefixedName("bot"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.1, height=0.5)]),
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

        connection = DifferentialDrive.create_with_dofs(
            world=world, parent=body, child=robot_world.root
        )
        world.merge_world(robot_world, connection)
        connection.has_hardware_interface = True

    return world


def world_with_urdf_factory(
    robot_semantic_annotation: Type[AbstractRobot],
    robot_starting_pose: HomogeneousTransformationMatrix | None = None,
    urdf_path_resolver: PathResolver | None = None,
    robot_localization_pose: HomogeneousTransformationMatrix | None = None,
):
    """
    Builds this tree:
    map -> odom_combined -> "urdf tree"
    """
    drive_connection_type = robot_semantic_annotation.get_drive_connection_type()
    urdf_parser = URDFParser.from_file(
        file_path=robot_semantic_annotation.get_ros_file_path(),
        path_resolver=urdf_path_resolver,
    )
    world_with_urdf = urdf_parser.parse()
    if robot_semantic_annotation is not None:
        robot_semantic_annotation.from_world(world_with_urdf)

    with world_with_urdf.modify_world():
        old_root = world_with_urdf.root
        map = Body(name=PrefixedName("map"))
        localization_body = Body(name=PrefixedName("odom_combined"))

        map_C_localization = Connection6DoF.create_with_dofs(
            world_with_urdf, map, localization_body
        )
        world_with_urdf.add_connection(map_C_localization)

        c_root_bf = drive_connection_type.create_with_dofs(
            parent=localization_body,
            child=old_root,
            world=world_with_urdf,
        )
        world_with_urdf.add_connection(c_root_bf)
        c_root_bf.has_hardware_interface = True
    if robot_localization_pose is not None:
        map_C_localization.origin = robot_localization_pose

    if robot_starting_pose is not None:
        c_root_bf.origin = robot_starting_pose

    return world_with_urdf


@pytest.fixture(scope="session")
def _pr2_world_setup():
    return world_with_urdf_factory(PR2)


@pytest.fixture(scope="function")
def pr2_world_copy(_pr2_world_setup):
    result = deepcopy(_pr2_world_setup)
    return result


@pytest.fixture(scope="session")
def _hsr_world_setup():
    return world_with_urdf_factory(HSRB)


@pytest.fixture(scope="function")
def hsr_world_copy(_hsr_world_setup):
    result = deepcopy(_hsr_world_setup)
    HSRB.from_world(result)
    return result


@pytest.fixture(scope="session")
def _garmi_world_setup():
    if Garmi is None:
        pytest.skip("GARMI semantic annotation not installed")
    try:
        return world_with_urdf_factory(Garmi)
    except ParsingError as error:
        pytest.skip(f"GARMI URDF not available: {error}")


@pytest.fixture(scope="session")
def tracy_world():
    if not tracy_installed():
        pytest.skip("Tracy not installed")
    tracy_parser = URDFParser.from_file(file_path=Tracy.get_ros_file_path())
    world_with_tracy = tracy_parser.parse()
    Tracy.from_world(world_with_tracy)
    return world_with_tracy


@pytest.fixture(scope="session")
def daisy_world():
    if not daisy_installed():
        pytest.skip("DAiSy not installed")
    daisy = "package://iai_daisy_description/robots/daisy.urdf.xacro"
    daisy_parser = URDFParser.from_file(file_path=daisy)
    world_with_daisy = daisy_parser.parse()
    DAiSy.from_world(world_with_daisy)
    return world_with_daisy


@pytest.fixture(scope="session")
def _stretch_world_setup():
    return world_with_urdf_factory(Stretch)


@pytest.fixture(scope="function")
def stretch_world_copy(_stretch_world_setup):
    return deepcopy(_stretch_world_setup)


@pytest.fixture(scope="session")
def _tiago_world_setup():
    return world_with_urdf_factory(Tiago)


@pytest.fixture(scope="session")
def _apartment_world_setup():
    apartment_world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "coraplex",
            "resources",
            "worlds",
            "apartment.urdf",
        )
    ).parse()
    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "coraplex",
            "resources",
            "objects",
            "milk.stl",
        )
    ).parse()
    cereal_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "coraplex",
            "resources",
            "objects",
            "breakfast_cereal.stl",
        )
    ).parse()
    spoon_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "coraplex",
            "resources",
            "objects",
            "spoon.stl",
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
            2.37, 2.5, 1.05, reference_frame=apartment_world.root
        ),
    )
    apartment_world.merge_world(
        spoon_world,
        FixedConnection(
            parent=apartment_world.get_body_by_name("cabinet10_drawer_top"),
            child=spoon_world.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.05, -0.05, -0.02
            ),
        ),
    )

    with apartment_world.modify_world():

        apartment_world.add_semantic_annotations(
            [
                Milk(root=apartment_world.get_body_by_name("milk.stl")),
                Spoon(root=apartment_world.get_body_by_name("spoon.stl")),
            ]
        )
        apartment_world.add_semantic_annotation_recursively(
            Drawer(
                root=apartment_world.get_body_by_name("cabinet10_drawer_top"),
                handle=Handle(root=apartment_world.get_body_by_name("handle_cab10_t")),
            )
        )

    return apartment_world


@pytest.fixture(scope="session")
def _elevator_world_setup():

    world = World()

    with world.modify_world():
        world.add_body(Body(name=PrefixedName("root")))

        wall_thickness = 0.05
        scale = Scale(1, 1, 1)
        name = PrefixedName("elevator")
        elevator = Elevator.create_with_new_body_in_world(
            name=PrefixedName("Elevator"),
            world=world,
            scale=Scale(1, 1, 1),
            wall_thickness=0.05,
        )

        vertical_drive = Slider.create_with_new_body_in_world(
            name=PrefixedName(f"{name.name}_drive", name.prefix),
            world=world,
            active_axis=Vector3.Z(),
        )
        elevator.add(vertical_drive)

        door_scale = Scale(wall_thickness, scale.y / 2, scale.z)
        door1 = Door.create_with_new_body_in_world(
            name=PrefixedName(f"{name.name}_door0", name.prefix),
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                Point3(-scale.x / 2, -scale.y / 4, 0),
                reference_frame=world.root,
            ),
            scale=door_scale,
        )
        door2 = Door.create_with_new_body_in_world(
            name=PrefixedName(f"{name.name}_door1", name.prefix),
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                Point3(-scale.x / 2, scale.y / 4, 0),
                reference_frame=world.root,
            ),
            scale=door_scale,
        )

        elevator.add(door1)
        elevator.add(door2)

        door_travel = door_scale.y
        door_slider_configs = (
            (
                door1,
                DerivativeMap(position=0.0),
                DerivativeMap(position=door_travel),
            ),
            (
                door2,
                DerivativeMap(position=0.0),
                DerivativeMap(position=door_travel),
            ),
        )
        for i, (current_door, lower, upper) in enumerate(door_slider_configs):
            door_slider = Slider.create_with_new_body_in_world(
                name=PrefixedName(f"{name.name}_door{i}_drive", name.prefix),
                world=world,
                active_axis=(Vector3.Y() * ((-1) ** (i + 1))),
                connection_limits=DegreeOfFreedomLimits(lower=lower, upper=upper),
            )
            current_door.add(door_slider)

        world.add_semantic_annotation(elevator)
    return world


@pytest.fixture(scope="function")
def apartment_world_copy(_apartment_world_setup):
    result = deepcopy(_apartment_world_setup)
    return result


@pytest.fixture(scope="function")
def apartment_world_pr2_copy_with_context(_apartment_world_setup, _pr2_world_setup):
    result = deepcopy(_apartment_world_setup)
    pr2_copy = deepcopy(_pr2_world_setup)
    result.merge_world(pr2_copy)
    return (
        result,
        result.get_semantic_annotations_by_type(AbstractRobot)[0],
        Context(
            result,
            result.get_semantic_annotations_by_type(AbstractRobot)[0],
        ),
    )


@pytest.fixture(scope="session")
def _simple_apartment_setup():
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)

        box = Body(
            name=PrefixedName("box"),
            collision=ShapeCollection([Box(scale=Scale(1, 1, 0.95))]),
            visual=ShapeCollection([Box(scale=Scale(1, 1, 0.95))]),
        )

        box_2 = Body(
            name=PrefixedName("box_2"),
            collision=ShapeCollection([Box(scale=Scale(1, 1, 0.95))]),
            visual=ShapeCollection([Box(scale=Scale(1, 1, 0.95))]),
        )

        box_1_connection = FixedConnection(
            parent=world.root,
            child=box,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                2, 0, 0.375, reference_frame=world.root
            ),
        )
        box_2_connection = FixedConnection(
            parent=root,
            child=box_2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                -2, 0, 0.375
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
            "coraplex",
            "resources",
            "objects",
            "milk.stl",
        )
    ).parse()
    world.merge_world_at_pose(
        milk_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.02, yaw=np.pi),
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
def apartment_meshes():
    """
    Skip tests that need the visual meshes of the ``iai_apartment`` package.
    """
    try:
        walls_mesh = ApartmentEnvironment.mesh_path("walls.dae")
    except ParsingError as error:
        pytest.skip(f"apartment meshes not available: {error}")
    if not os.path.isfile(walls_mesh):
        pytest.skip(f"apartment meshes not available: {walls_mesh} is missing")


@pytest.fixture(scope="session")
def apartment_environment_world(apartment_meshes):
    """
    A world holding nothing but the apartment of :class:`ApartmentEnvironment`.
    """
    return ApartmentEnvironment().get_world()


@pytest.fixture(scope="session")
def pr2_apartment_world(_pr2_world_setup, _apartment_world_setup):
    """
    Builds this tree:
    map -> odom_combined -> pr2 urdf tree
        -> apartment urdf
    """
    pr2_copy = deepcopy(_pr2_world_setup)
    apartment_copy = deepcopy(_apartment_world_setup)

    pr2_copy.merge_world(apartment_copy)
    pr2_copy.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(1.3, 2, 0)
    )
    return pr2_copy


@pytest.fixture(scope="session")
def simple_pr2_world_setup(_pr2_world_setup, _simple_apartment_setup):
    apartment_world = deepcopy(_simple_apartment_setup)
    pr2_copy = deepcopy(_pr2_world_setup)
    pr2_copy.merge_world(apartment_world)
    robot_view = pr2_copy.get_semantic_annotations_by_type(PR2)[0]
    return pr2_copy, robot_view, Context(pr2_copy, robot_view)


@pytest.fixture(scope="session")
def hsr_apartment_world(_hsr_world_setup, _apartment_world_setup):
    apartment_copy = deepcopy(_apartment_world_setup)
    hsr_copy = deepcopy(_hsr_world_setup)
    robot_view = hsr_copy.get_semantic_annotations_by_type(HSRB)[0]

    apartment_copy.merge_world_at_pose(
        hsr_copy, HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
    )

    return apartment_copy, robot_view, Context(apartment_copy, robot_view)


@pytest.fixture(scope="session")
def stretch_apartment_world(_stretch_world_setup, _apartment_world_setup):
    apartment_copy = deepcopy(_apartment_world_setup)
    stretch_copy = deepcopy(_stretch_world_setup)

    apartment_copy.merge_world_at_pose(
        stretch_copy, HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
    )

    return apartment_copy


@pytest.fixture(scope="session")
def tiago_apartment_world(_tiago_world_setup, _apartment_world_setup):
    apartment_copy = deepcopy(_apartment_world_setup)
    tiago_copy = deepcopy(_tiago_world_setup)
    apartment_copy.merge_world(tiago_copy)

    return apartment_copy, Tiago.from_world(apartment_copy)


###############################
######## World with reset #####
###############################


@pytest.fixture
def pr2_world_state_reset(_pr2_world_setup):
    world = deepcopy(_pr2_world_setup)
    state = world.state._data.copy()
    yield world
    world.state._data[:] = state


@pytest.fixture
def pr2_apartment_state_reset(pr2_apartment_world):
    world = deepcopy(pr2_apartment_world)
    state = deepcopy(world.state._data)
    yield world
    world.state._data = state


###############################
######### Utils ###############
###############################


@dataclass
class RosPublishers:
    """
    Keeps the ros publishers a test started and stops them when the test ends.

    A publisher stays registered on the world it publishes, so one that is left running
    on a world outliving the test publishes on a node that is already destroyed.
    """

    started: List[Callback] = field(default_factory=list)
    """
    The publishers started so far, in the order they were started.
    """

    def adopt(self, publisher: CallbackT) -> CallbackT:
        """
        :param publisher: The publisher whose lifetime ends with the test.
        :return: the publisher itself, so it can be adopted where it is created.
        """
        self.started.append(publisher)
        return publisher

    def stop_all(self) -> None:
        """
        Stop every adopted publisher, latest first.
        """
        for publisher in reversed(self.started):
            publisher.stop()
        self.started.clear()


@pytest.fixture(scope="function")
def ros_publishers():
    """
    Hands out the owner of every publisher a test starts on a world it shares with other
    tests.
    """
    publishers = RosPublishers()
    yield publishers
    publishers.stop_all()


@pytest.fixture(scope="function")
def init_rospy():
    """
    Gives a test the global Giskard ros node.

    ..warning::
        This fixture drives the same global ros context as :func:`rclpy_node`, so the
        two cannot be used together.
    """
    if rospy is None:
        pytest.skip("ROS not installed")

    rospy.init_node("giskard")

    try:
        yield None
    finally:
        # Cleanly reset TF and shutdown ROS2 node/executor
        rospy.shutdown()


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


@pytest.fixture(scope="session")
def kitchen_environment_fixture():
    world = World()
    all_elements_connections = []
    root = Body(name=PrefixedName("root"))

    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        fruit_table = Table.create_with_new_body_in_world(
            world=world,
            name="fruit_table",
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, y=1, z=0
            ),
            scale=Scale(2, 2, 1),
        )

        vegetable_table = Table.create_with_new_body_in_world(
            world=world,
            name="vegetable_table",
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, y=1, z=2
            ),
            scale=Scale(2, 2, 1),
        )

        empty_table = Table.create_with_new_body_in_world(
            world=world,
            name="empty_table",
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, y=1, z=4
            ),
            scale=Scale(2, 2, 1),
        )

        empty_table2 = Table.create_with_new_body_in_world(
            world=world,
            name="empty_table2",
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, y=1, z=6
            ),
            scale=Scale(2, 2, 1),
        )

        apple = Apple.create_with_new_body_in_world(
            world=world,
            name="apple",
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, y=1, z=0.55
            ),
            scale=Scale(0.10, 0.10, 0.10),
        )
        for color in apple.bodies[0].visual.shapes:
            color.color = Color.RED()

        orange = Orange.create_with_new_body_in_world(
            world=world,
            name="orange",
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, y=0.5, z=0.55
            ),
            scale=Scale(0.10, 0.10, 0.10),
        )
        for color in orange.bodies[0].visual.shapes:
            color.color = Color.ORANGE()

        banana1 = Banana.create_with_new_body_in_world(
            world=world,
            name="banana1",
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, y=0.6, z=0.75
            ),
            scale=Scale(0.10, 0.10, 0.60),
        )
        for color in banana1.bodies[0].visual.shapes:
            color.color = Color.YELLOW()

        carrot = Carrot.create_with_new_body_in_world(
            world=world,
            name="carrot",
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, y=1, z=2.6
            ),
            scale=Scale(0.05, 0.05, 0.20),
        )
        for color in carrot.bodies[0].visual.shapes:
            color.color = Color.ORANGE()

        lettuce = Lettuce.create_with_new_body_in_world(
            world=world,
            name="lettuce",
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, y=1.5, z=2.55
            ),
            scale=Scale(0.15, 0.15, 0.10),
        )
        for color in lettuce.bodies[0].visual.shapes:
            color.color = Color.GREEN()

        banana = Banana.create_with_new_body_in_world(
            world=world,
            name="banana",
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=10, y=10, z=10
            ),
            scale=Scale(0.20, 0.05, 0.05),
        )
        for color in banana.bodies[0].visual.shapes:
            color.color = Color.YELLOW()

    fake_robot = Cylinder(width=0.45, height=1.5)
    shape_geometry = ShapeCollection([fake_robot])
    fake_robot_body = Body(
        name=PrefixedName("base_link_body"),
        collision=shape_geometry,
        visual=shape_geometry,
    )

    root_C_fake_robot = FixedConnection(
        parent=root,
        child=fake_robot_body,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(),
    )
    all_elements_connections.append(root_C_fake_robot)

    with world.modify_world():
        for conn in all_elements_connections:
            world.add_connection(conn)

    return world
