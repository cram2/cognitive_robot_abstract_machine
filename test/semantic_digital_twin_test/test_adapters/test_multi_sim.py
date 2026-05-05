import logging
import os
import time

import mujoco
import pytest

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Box, Scale, Color, Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Region, Actuator

from physics_simulators.mujoco_simulator import MujocoSimulator
from physics_simulators.base_simulator import SimulatorState
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSim, MujocoActuator

urdf_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "semantic_digital_twin",
    "resources",
    "urdf",
)
mjcf_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "semantic_digital_twin",
    "resources",
    "mjcf",
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

headless = os.environ.get("CI", "false").lower() == "true"
only_run_test_in_CI = os.environ.get("CI", "false").lower() == "false"
# only_run_test_in_CI = False

pytestmark = pytest.mark.skipif(
    only_run_test_in_CI,
    reason="Only run test in CI or multisim could not be imported.",
)

TEST_URDF_1 = os.path.normpath(os.path.join(urdf_dir, "simple_two_arm_robot.urdf"))
TEST_URDF_2 = os.path.normpath(os.path.join(urdf_dir, "hsrb.urdf"))
TEST_URDF_TRACY = os.path.normpath(os.path.join(urdf_dir, "tracy.urdf"))
TEST_MJCF_1 = os.path.normpath(os.path.join(mjcf_dir, "mjx_single_cube_no_mesh.xml"))
TEST_MJCF_2 = os.path.normpath(os.path.join(mjcf_dir, "jeroen_cups.xml"))
STEP_SIZE = 1e-3


def stop_multisim_if_running(multi_sim: MujocoSim) -> None:
    simulator = getattr(multi_sim, "simulator", None)
    if simulator is None:
        return
    if getattr(simulator, "state", None) is SimulatorState.STOPPED:
        return
    multi_sim.stop_simulation()


@pytest.fixture
def test_urdf_1_world():
    return URDFParser.from_file(file_path=TEST_URDF_1).parse()


@pytest.fixture
def test_mjcf_1_world():
    return MJCFParser(TEST_MJCF_1).parse()


@pytest.fixture
def test_mjcf_2_world():
    return MJCFParser(TEST_MJCF_2).parse()


def test_empty_multi_sim_in_5s():
    world = World()
    multi_sim = MujocoSim(world=world, headless=headless)

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.file_path == "/tmp/scene.xml"
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def test_world_multi_sim_in_5s(test_urdf_1_world):
    multi_sim = MujocoSim(world=test_urdf_1_world, headless=headless)

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.file_path == "/tmp/scene.xml"
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def test_apartment_multi_sim_in_5s():
    try:
        test_urdf_2_world = URDFParser.from_file(file_path=TEST_URDF_2).parse()
    except ParsingError:
        pytest.skip("Skipping HSRB krrood_test due to URDF parsing error.")

    multi_sim = MujocoSim(world=test_urdf_2_world, headless=headless)

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.file_path == "/tmp/scene.xml"
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def test_world_multi_sim_with_change(test_urdf_1_world):
    multi_sim = MujocoSim(world=test_urdf_1_world, headless=headless)

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.file_path == "/tmp/scene.xml"
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        time.sleep(1.0)

        start_time = time.time()

        new_body = Body(name=PrefixedName("test_body"))
        box_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.2, y=0.4, z=3.0, roll=0, pitch=0.5, yaw=0, reference_frame=new_body
        )
        box = Box(
            origin=box_origin,
            scale=Scale(1.0, 1.5, 0.5),
            color=Color(1.0, 0.0, 0.0, 1.0),
        )
        new_body.collision = ShapeCollection([box], reference_frame=new_body)

        logger.debug(f"Time before adding new body: {time.time() - start_time}s")
        with test_urdf_1_world.modify_world():
            test_urdf_1_world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=test_urdf_1_world,
                    parent=test_urdf_1_world.root,
                    child=new_body,
                )
            )
        logger.debug(f"Time after adding new body: {time.time() - start_time}s")

        assert new_body.name.name in multi_sim.simulator.get_all_body_names().result

        time.sleep(0.5)

        region = Region(name=PrefixedName("test_region"))
        region_box = Box(
            scale=Scale(0.1, 0.5, 0.2),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=region),
            color=Color(0.0, 1.0, 0.0, 0.8),
        )
        region.area = ShapeCollection([region_box], reference_frame=region)

        logger.debug(f"Time before add adding region: {time.time() - start_time}s")
        with test_urdf_1_world.modify_world():
            test_urdf_1_world.add_connection(
                FixedConnection(
                    parent=test_urdf_1_world.root,
                    child=region,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=0.5
                    ),
                )
            )
        logger.debug(f"Time after add adding region: {time.time() - start_time}s")

        assert region.name.name in multi_sim.simulator.get_all_body_names().result

        time.sleep(0.5)

        T_const = 0.1
        kp = 100
        kv = 10
        actuator = Actuator()
        dof = test_urdf_1_world.get_degree_of_freedom_by_name(name="r_joint_1")
        actuator.add_dof(dof=dof)
        actuator.simulator_additional_properties.append(
            MujocoActuator(
                dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                dynamics_parameters=[T_const] + [0.0] * 9,
                gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                gain_parameters=[kp] + [0.0] * 9,
                bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                bias_parameters=[0, -kp, -kv] + [0.0] * 7,
            )
        )

        logger.debug(f"Time before adding new actuator: {time.time() - start_time}s")
        with test_urdf_1_world.modify_world():
            test_urdf_1_world.add_actuator(actuator=actuator)
        logger.debug(f"Time after adding new actuator: {time.time() - start_time}s")

        assert actuator.name.name in multi_sim.simulator.get_all_actuator_names().result

        time.sleep(4.0)
        multi_sim.stop_simulation()
    finally:
        stop_multisim_if_running(multi_sim)


def test_multi_sim_in_5s(test_mjcf_1_world):
    multi_sim = MujocoSim(
        world=test_mjcf_1_world,
        headless=headless,
        step_size=STEP_SIZE,
    )

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def test_mesh_scale_and_equality(test_mjcf_2_world):
    multi_sim = MujocoSim(
        world=test_mjcf_2_world,
        headless=headless,
        step_size=STEP_SIZE,
    )

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def test_mujoco_with_tracy_dae_files():
    try:
        dae_world = URDFParser.from_file(file_path=TEST_URDF_TRACY).parse()
    except ParsingError:
        pytest.skip("Skipping tracy test due to URDF parsing error.")

    multi_sim = MujocoSim(world=dae_world, headless=headless)

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.file_path == "/tmp/scene.xml"
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def test_mujocosim_world_with_added_objects(test_urdf_1_world):
    milk_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "semantic_digital_twin",
        "resources",
        "stl",
        "milk.stl",
    )
    stl_parser = STLParser(milk_path)
    mesh_world = stl_parser.parse()
    transformation = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.5, reference_frame=test_urdf_1_world.root
    )

    with test_urdf_1_world.modify_world():
        test_urdf_1_world.merge_world_at_pose(mesh_world, transformation)

    multi_sim = MujocoSim(world=test_urdf_1_world, headless=headless)

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.file_path == "/tmp/scene.xml"
        assert multi_sim.simulator.headless is headless
        assert multi_sim.simulator.step_size == STEP_SIZE

        multi_sim.start_simulation()
        start_time = time.time()
        time.sleep(5.0)
        multi_sim.stop_simulation()

        assert time.time() - start_time >= 5.0
    finally:
        stop_multisim_if_running(multi_sim)


def test_spawn_body_with_connections():
    def spawn_robot_body(spawn_world: World) -> Body:
        spawn_body = Body(name=PrefixedName("robot"))
        box_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0, y=0, z=0.5, roll=0, pitch=0, yaw=0, reference_frame=spawn_body
        )
        box = Box(
            origin=box_origin,
            scale=Scale(0.4, 0.4, 1.0),
            color=Color(0.9, 0.9, 0.9, 1.0),
        )
        spawn_body.collision = ShapeCollection([box], reference_frame=spawn_body)

        with spawn_world.modify_world():
            spawn_world.add_connection(
                FixedConnection(parent=spawn_world.root, child=spawn_body)
            )

        return spawn_body

    def spawn_shoulder_bodies(spawn_world: World, root_body: Body) -> tuple[Body, Body]:
        spawn_left_shoulder_body = Body(name=PrefixedName("left_shoulder"))
        cylinder = Cylinder(
            width=0.2,
            height=0.1,
            color=Color(0.9, 0.1, 0.1, 1.0),
        )
        spawn_left_shoulder_body.collision = ShapeCollection(
            [cylinder], reference_frame=spawn_left_shoulder_body
        )
        dof = DegreeOfFreedom(name=PrefixedName("left_shoulder_joint"))
        left_shoulder_origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0,
            pos_y=0.3,
            pos_z=0.9,
            quat_w=0.707,
            quat_x=0.707,
            quat_y=0,
            quat_z=0,
        )

        with spawn_world.modify_world():
            spawn_world.add_degree_of_freedom(dof)
            spawn_world.add_connection(
                RevoluteConnection(
                    name=dof.name,
                    parent=root_body,
                    child=spawn_left_shoulder_body,
                    axis=Vector3.Z(reference_frame=spawn_left_shoulder_body),
                    dof_id=dof.id,
                    parent_T_connection_expression=left_shoulder_origin,
                )
            )

        spawn_right_shoulder_body = Body(name=PrefixedName("right_shoulder"))
        cylinder = Cylinder(
            width=0.2,
            height=0.1,
            color=Color(0.9, 0.1, 0.1, 1.0),
        )
        spawn_right_shoulder_body.collision = ShapeCollection(
            [cylinder], reference_frame=spawn_right_shoulder_body
        )
        dof = DegreeOfFreedom(name=PrefixedName("right_shoulder_joint"))
        right_shoulder_origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0,
            pos_y=-0.3,
            pos_z=0.9,
            quat_w=0.707,
            quat_x=0.707,
            quat_y=0,
            quat_z=0,
        )

        with spawn_world.modify_world():
            spawn_world.add_degree_of_freedom(dof)
            spawn_world.add_connection(
                RevoluteConnection(
                    name=dof.name,
                    parent=root_body,
                    child=spawn_right_shoulder_body,
                    axis=Vector3.Z(reference_frame=spawn_right_shoulder_body),
                    dof_id=dof.id,
                    parent_T_connection_expression=right_shoulder_origin,
                )
            )

        return spawn_left_shoulder_body, spawn_right_shoulder_body

    world = World()
    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=0.001,
    )

    try:
        multi_sim.start_simulation()
        time.sleep(1)

        robot_body = spawn_robot_body(spawn_world=world)
        spawn_shoulder_bodies(spawn_world=world, root_body=robot_body)

        time.sleep(1)

        assert set(multi_sim.simulator.get_all_body_names().result) == {
            "world",
            "robot",
            "left_shoulder",
            "right_shoulder",
        }

        multi_sim.stop_simulation()
    finally:
        stop_multisim_if_running(multi_sim)
