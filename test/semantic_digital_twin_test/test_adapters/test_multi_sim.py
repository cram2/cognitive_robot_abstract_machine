import logging
import os
import time

import mujoco
import pytest
import numpy

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
    Color,
    Cylinder,
    Mesh,
    Texture,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Region, Actuator

from physics_simulators.mujoco_simulator import MujocoSimulator
from physics_simulators.base_simulator import SimulatorState
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import (
    MujocoSim,
    MujocoActuator,
    MujocoBuilder,
    MujocoLight,
)

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

pytestmark = pytest.mark.skipif(
    only_run_test_in_CI,
    reason="Only run test in CI or multisim could not be imported.",
)

TEST_URDF_1 = os.path.normpath(os.path.join(urdf_dir, "simple_two_arm_robot.urdf"))
TEST_URDF_2 = HSRB.get_ros_file_path()
TEST_URDF_TRACY = Tracy.get_ros_file_path()
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
        assert multi_sim.simulator.file_path == MujocoSim.default_file_path
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
        assert multi_sim.simulator.file_path == MujocoSim.default_file_path
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
        assert multi_sim.simulator.file_path == MujocoSim.default_file_path
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
        assert multi_sim.simulator.file_path == MujocoSim.default_file_path
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


def _write_textured_tetrahedron(directory, texture_color) -> str:
    """
    Writes a minimal textured OBJ+MTL+PNG mesh (a tetrahedron, so its convex hull is
    non-degenerate) into ``directory``, textured with a solid ``texture_color``, and returns
    the OBJ file's path. Always named "tetra.obj"/"tetra.mtl"/"wood.png", so callers writing
    into different directories can reproduce a texture basename collision between them.
    """
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=texture_color).save(directory / "wood.png")
    (directory / "tetra.mtl").write_text("newmtl wood\nmap_Kd wood.png\n")
    mesh_file = directory / "tetra.obj"
    mesh_file.write_text(
        "mtllib tetra.mtl\n"
        "o tetra\n"
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "v 0.0 0.0 1.0\n"
        "vt 0.0 0.0\n"
        "vt 1.0 0.0\n"
        "vt 0.0 1.0\n"
        "vt 0.5 0.5\n"
        "usemtl wood\n"
        "f 1/1 2/2 3/3\n"
        "f 1/1 2/2 4/4\n"
        "f 1/1 3/3 4/4\n"
        "f 2/2 3/3 4/4\n"
    )
    return str(mesh_file)


def _build_world_with_two_textured_bodies(
    tmp_path, mesh_file_a: str, mesh_file_b: str
) -> MujocoBuilder:
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)
        for name, mesh_file in [("quad_0", mesh_file_a), ("quad_1", mesh_file_b)]:
            mesh_shape = Mesh(filename=mesh_file, scale=Scale(1, 1, 1))
            body = Body(
                name=PrefixedName(name),
                visual=ShapeCollection([mesh_shape]),
                collision=ShapeCollection([mesh_shape]),
            )
            world.add_kinematic_structure_entity(body)
            world.add_connection(FixedConnection(parent=root, child=body))

    builder = MujocoBuilder()
    builder.build_world(world=world, file_path=str(tmp_path / "scene.xml"))
    return builder


def test_builder_assigns_material_to_every_geom_sharing_a_texture(tmp_path):
    """
    Regression test: MujocoBuilder._parse_geom used to return early - without ever setting
    geom_props["material"] - whenever a geom's texture was already registered by an earlier
    geom. Since most textures in a scene are shared across many geoms (a real RoboCasa
    kitchen reuses a handful of textures across ~90 meshes), this meant only the first geom
    to use a given texture ever got a material; every later reuse silently rendered with
    MuJoCo's default (untextured, gray) material instead.
    """
    mesh_file = _write_textured_tetrahedron(tmp_path, texture_color=(120, 60, 20))

    builder = _build_world_with_two_textured_bodies(tmp_path, mesh_file, mesh_file)

    materials = {
        body.name: geom.material for body in builder.spec.bodies for geom in body.geoms
    }
    assert materials["quad_0"] == materials["quad_1"]
    assert materials["quad_0"] != ""


def test_builder_does_not_confuse_different_textures_sharing_a_basename(tmp_path):
    """
    Regression test: RoboCasa's asset pipeline reuses generic texture basenames (e.g.
    "T_BC001.png") across many unrelated fixtures' own distinct texture files - a real
    kitchen had 14 different fixtures (sink, stove, fridge, dishwasher, ...) all using a
    texture file named exactly "T_BC001.png" in their own directories. Deduplicating by
    basename alone collapsed all of them onto whichever fixture's texture was registered
    first, so most fixtures rendered with the wrong (borrowed) texture image instead of
    their own.
    """
    mesh_file_a = _write_textured_tetrahedron(
        tmp_path / "fixture_a", texture_color=(200, 0, 0)
    )
    mesh_file_b = _write_textured_tetrahedron(
        tmp_path / "fixture_b", texture_color=(0, 200, 0)
    )

    builder = _build_world_with_two_textured_bodies(tmp_path, mesh_file_a, mesh_file_b)

    materials = {
        body.name: geom.material for body in builder.spec.bodies for geom in body.geoms
    }
    assert materials["quad_0"] != materials["quad_1"]
    texture_files = {texture.name: texture.file for texture in builder.spec.textures}
    assert len(texture_files) == 2


def test_builder_writes_a_light_attached_to_a_body(tmp_path):
    """
    Regression test: MujocoBuilder had no handling for MujocoLight additional properties at
    all, so a world's lights were silently dropped when built into a MuJoCo scene - every
    recorded/simulated world fell back to MuJoCo's minimal default camera headlight instead
    of the scene's own intended lighting.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)
        root.simulator_additional_properties.append(
            MujocoLight(
                name="overview_light",
                body=root,
                directional=True,
                position=[2.0, -2.0, 2.0],
                direction=[0.0, 0.0, -1.0],
                ambient=[0.3, 0.3, 0.3],
                diffuse=[0.5, 0.5, 0.5],
                specular=[0.3, 0.3, 0.3],
            )
        )

    builder = MujocoBuilder()
    builder.build_world(world=world, file_path=str(tmp_path / "scene.xml"))

    [light] = [light for body in builder.spec.bodies for light in body.lights]
    assert light.name == "overview_light"
    assert list(light.pos) == pytest.approx([2.0, -2.0, 2.0])
    assert list(light.ambient) == pytest.approx([0.3, 0.3, 0.3])
    assert list(light.diffuse) == pytest.approx([0.5, 0.5, 0.5])


def test_builder_assigns_material_to_a_textured_primitive_shape(tmp_path):
    """
    Regression test: Box/Sphere/Cylinder shapes never carried any texture reference, only a
    flat Color - RoboCasa's countertops and cabinet doors are actual MJCF box geoms with a
    material referencing a marble/wood texture, so this whole texture reference was silently
    discarded on every round-trip and they rendered flat-colored instead of textured.
    """
    from PIL import Image

    texture_directory = tmp_path / "textures"
    texture_directory.mkdir()
    texture_file = texture_directory / "marble.png"
    Image.new("RGB", (4, 4), color=(200, 200, 200)).save(texture_file)

    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        world.add_body(root)
        box_shape = Box(
            scale=Scale(1, 1, 1),
            texture=Texture(
                file_path=str(texture_file), repeat=(3.0, 3.0), uniform=True
            ),
        )
        counter = Body(
            name=PrefixedName("counter"),
            visual=ShapeCollection([box_shape]),
            collision=ShapeCollection([box_shape]),
        )
        world.add_kinematic_structure_entity(counter)
        world.add_connection(FixedConnection(parent=root, child=counter))

    builder = MujocoBuilder()
    builder.build_world(world=world, file_path=str(tmp_path / "scene.xml"))

    [geom] = [
        geom
        for body in builder.spec.bodies
        for geom in body.geoms
        if body.name == "counter"
    ]
    assert geom.material != ""
    [material] = [
        material
        for material in builder.spec.materials
        if material.name == geom.material
    ]
    assert list(material.texrepeat) == pytest.approx([3.0, 3.0])
    assert bool(material.texuniform) is True
    texture_name = material.textures[0]
    assert texture_name != ""
    [texture] = [
        texture for texture in builder.spec.textures if texture.name == texture_name
    ]
    assert texture.file == str(texture_file)


def test_mujoco_with_tracy_dae_files():
    try:
        dae_world = URDFParser.from_file(file_path=TEST_URDF_TRACY).parse()
    except ParsingError:
        pytest.skip("Skipping tracy test due to URDF parsing error.")

    multi_sim = MujocoSim(world=dae_world, headless=headless)

    try:
        assert isinstance(multi_sim.simulator, MujocoSimulator)
        assert multi_sim.simulator.file_path == MujocoSim.default_file_path
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
        assert multi_sim.simulator.file_path == MujocoSim.default_file_path
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
                    raw_dof=dof,
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
                    raw_dof=dof,
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


def test_world_sim_state_sync():
    plane_half_thickness = 0.05
    box_half_size = 0.1
    init_pos = numpy.array([0.3, 0.2, 5.0])
    target_pos = numpy.array(
        [init_pos[0], init_pos[1], plane_half_thickness + box_half_size]
    )

    def spawn_state_sync_scene(
        spawn_world: World,
    ) -> tuple[Body, Connection6DoF]:
        plane_body = Body(name=PrefixedName("ground_plane"))
        plane_body.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        reference_frame=plane_body
                    ),
                    scale=Scale(2.0, 2.0, plane_half_thickness * 2),
                    color=Color(1.0, 1.0, 0.0, 1.0),
                )
            ],
            reference_frame=plane_body,
        )

        falling_box = Body(name=PrefixedName("falling_box"))
        falling_box.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        reference_frame=falling_box
                    ),
                    scale=Scale(
                        box_half_size * 2, box_half_size * 2, box_half_size * 2
                    ),
                    color=Color(1.0, 0.0, 0.0, 1.0),
                )
            ],
            reference_frame=falling_box,
        )

        with spawn_world.modify_world():
            spawn_world.add_connection(
                FixedConnection(parent=spawn_world.root, child=plane_body)
            )
            box_connection = Connection6DoF.create_with_dofs(
                world=spawn_world,
                parent=spawn_world.root,
                child=falling_box,
            )
            spawn_world.add_connection(box_connection)
        return falling_box, box_connection

    world = World()
    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=STEP_SIZE,
    )

    try:
        multi_sim.start_simulation()
        time.sleep(1)

        falling_box, box_connection = spawn_state_sync_scene(world)

        body_names = multi_sim.simulator.get_all_body_names().result
        assert {"ground_plane", "falling_box"}.issubset(
            body_names
        ), f"scene bodies were not spawned in the simulator; bodies={body_names}"

        box_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=float(init_pos[0]),
            y=float(init_pos[1]),
            z=float(init_pos[2]),
            reference_frame=falling_box,
        )
        time.sleep(2.5)

        final_pos = numpy.asarray(
            multi_sim.simulator.get_body_position("falling_box").result[:3],
            dtype=float,
        )

        multi_sim.stop_simulation()

        assert numpy.allclose(final_pos, target_pos, atol=1e-1), (
            f"Box did not settle at target: final_pos={final_pos}, "
            f"expected≈{target_pos}"
        )
    finally:
        stop_multisim_if_running(multi_sim)
