import os.path

import pytest

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoLight
from semantic_digital_twin.world_description.connections import FixedConnection

MJCF_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "semantic_digital_twin",
    "resources",
    "mjcf",
)


@pytest.fixture
def table_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "table.xml"))


@pytest.fixture
def kitchen_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "kitchen-small.xml"))


@pytest.fixture
def apartment_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "iai_apartment.xml"))


@pytest.fixture
def pr2_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "pr2_kinematic_tree.xml"))


def test_table_parsing(table_xml_parser):
    body_num = 7
    world = table_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) == body_num

    origin_left_front_leg_joint = world.get_connection(
        world.root, world.kinematic_structure_entities[1]
    )
    assert isinstance(origin_left_front_leg_joint, FixedConnection)


def test_kitchen_parsing(kitchen_xml_parser):
    world = kitchen_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0


def test_apartment_parsing(apartment_xml_parser):
    world = apartment_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0


def test_pr2_parsing(pr2_xml_parser):
    world = pr2_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0
    assert world.root.name.name == "world"


HINGED_BODY_MJCF = """
<mujoco>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="door">
        <joint name="hinge" type="hinge" axis="0 0 1" range="-1.57 0"/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_joint_position_limits_are_python_floats():
    """
    Parsed joint position limits must be plain Python floats.

    MuJoCo reports them as numpy scalars, which do not interoperate with the symbolic-math layer (``numpy_scalar - symbol`` makes numpy
    try to arrayify the symbol) and break motion planning on the joint.
    """
    world = MJCFParser.from_xml_string(HINGED_BODY_MJCF).parse()
    limits = world.get_degree_of_freedom_by_name("hinge").limits
    assert type(limits.lower.position) is float
    assert type(limits.upper.position) is float


LIT_WORLD_MJCF = """
<mujoco>
  <worldbody>
    <light pos="2.0 -2.0 2.0" dir="0.01 0.01 -1" specular="0.3 0.3 0.3" ambient="0.3 0.3 0.3"
           diffuse="0.3 0.3 0.3" directional="true" castshadow="false"/>
    <body name="base">
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_light_is_parsed_and_attached_to_its_parent_body():
    """
    Regression test: MJCFParser used to have no handling for <light> elements at all, so every
    world built through the parser -> World -> MujocoBuilder round-trip silently lost all
    lighting information, falling back to MuJoCo's minimal default camera headlight instead of
    the scene's own intended lights.
    """
    world = MJCFParser.from_xml_string(LIT_WORLD_MJCF).parse()

    lights = [
        light_property
        for light_property in world.root.simulator_additional_properties
        if isinstance(light_property, MujocoLight)
    ]
    assert len(lights) == 1
    light = lights[0]
    assert light.position == pytest.approx([2.0, -2.0, 2.0])
    assert light.direction == pytest.approx([0.01, 0.01, -1.0], abs=1e-3)
    assert light.ambient == pytest.approx([0.3, 0.3, 0.3])
    assert light.diffuse == pytest.approx([0.3, 0.3, 0.3])
    assert light.specular == pytest.approx([0.3, 0.3, 0.3])
    assert light.directional is True
    assert light.cast_shadow is False


TEXTURED_BOX_MJCF_TEMPLATE = """
<mujoco>
  <asset>
    <texture name="marble_tex" type="2d" file="{texture_file_path}"/>
    <material name="marble" texture="marble_tex" texrepeat="3 3" texuniform="true"/>
  </asset>
  <worldbody>
    <body name="counter">
      <geom type="box" size="0.5 0.5 0.05" material="marble"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_primitive_box_geom_resolves_its_material_texture(tmp_path):
    """
    Regression test: Box/Sphere/Cylinder shapes never carried any texture reference, only a
    flat Color. RoboCasa's countertops and cabinet doors are actual MJCF box geoms whose
    material references a marble/wood texture, so this reference was silently discarded on
    every round-trip and they rendered flat-colored instead of textured.
    """
    from PIL import Image

    texture_file_path = tmp_path / "marble.png"
    Image.new("RGB", (4, 4), color=(200, 200, 200)).save(texture_file_path)
    mjcf_file_path = tmp_path / "scene.xml"
    mjcf_file_path.write_text(
        TEXTURED_BOX_MJCF_TEMPLATE.format(texture_file_path=texture_file_path)
    )

    world = MJCFParser(str(mjcf_file_path)).parse()

    [counter] = [
        body
        for body in world.kinematic_structure_entities
        if body.name.name == "counter"
    ]
    [box_shape] = counter.visual.shapes
    assert box_shape.texture is not None
    assert box_shape.texture.file_path == str(texture_file_path)
    assert box_shape.texture.repeat == pytest.approx([3.0, 3.0])
    assert box_shape.texture.uniform is True
