"""
Tests for the onboarder's pure post-processing and the URDF asset bundler.

Recording itself needs a running coraplex demo, but everything that turns a recording
into a scene bundle is plain data work: deciding when an object moved, finding the
attach/detach window of each transport, labelling the resulting segments, and making a
URDF self-contained. Those are covered here against hand-built recordings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import inspect
import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest
from semantic_digital_twin.adapters.gazebo import GazeboParser
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.package_resolver import PackageUriResolver
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    OmniDrive,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, Dict, List, Optional

from cramera import paths
from cramera.onboard import bundle_urdf as bundler
from cramera.onboard.bundle_world import BundledWorld
from cramera.onboard.world_to_urdf import UrdfDocument
from cramera.onboard.demo import (
    BundledModel,
    SceneBuilder,
    SceneIndexEntry,
    Recorder,
    RecordingAnalysis,
    SpawnedBox,
    split_passthrough_arguments,
)

RESTING = [0.0, 0.0, 1.0, 0, 0, 0, 1]
"""
A pose that stays put, used wherever a frame's value must not matter.
"""

ONE_MESH_URDF_TEXT = (
    '<robot name="demo">\n'
    '  <link name="base_link"/>\n'
    '  <link name="cup_link">\n'
    "    <visual><geometry>\n"
    '      <mesh filename="meshes/cup.stl"/>\n'
    "    </geometry></visual>\n"
    "  </link>\n"
    '  <joint name="cup_joint" type="fixed">\n'
    '    <parent link="base_link"/><child link="cup_link"/>\n'
    "  </joint>\n"
    "</robot>\n"
)
"""
A URDF referencing exactly one mesh, shared by the URDF- and xacro-source bundling
tests.
"""


def pose_at(x: float, y: float, z: float = 1.0) -> List[float]:
    """
    A pose with the given position and no rotation.
    """
    return [x, y, z, 0, 0, 0, 1]


def recording(
    object_frames: List[Dict[str, List[float]]],
    base_frames: List[List[float]] = None,
    actions: List[Dict[str, Any]] = None,
) -> Recorder:
    """
    A recorder holding a finished recording, without having run a demo.
    """
    recorder = Recorder()
    recorder.object_frames = object_frames
    recorder.frames = [{} for _ in object_frames]
    recorder.base_frames = base_frames or [RESTING for _ in object_frames]
    recorder.actions = actions or []
    return recorder


# %% recorder field defaults
class TestRecorderMutableDefaults:
    """
    Each ``Recorder()`` must own its own containers, never a class-shared one.
    """

    def test_two_recorders_do_not_share_their_mutable_fields(self):
        first = Recorder()
        second = Recorder()

        for field_name in (
            "resolutions",
            "urdf_sources",
            "mesh_sources",
            "frames",
            "base_frames",
            "object_frames",
            "actions",
            "plan_nodes",
        ):
            assert getattr(first, field_name) is not getattr(second, field_name)


# %% CLI argument passthrough
class TestSplitPassthroughArguments:
    """
    ``cramera-onboard`` forwards a demo file's own CLI arguments unchanged, so a demo
    that parses its own ``sys.argv`` can still be onboarded.
    """

    def test_no_separator_keeps_everything_as_cramera_onboards_own(self):
        split = split_passthrough_arguments(["demo.py", "--name", "kitchen"])

        assert split.own == ["demo.py", "--name", "kitchen"]
        assert split.passthrough == []

    def test_separator_splits_into_own_and_passthrough(self):
        split = split_passthrough_arguments(
            ["demo.py", "--name", "kitchen", "--", "--robot", "pr2"]
        )

        assert split.own == ["demo.py", "--name", "kitchen"]
        assert split.passthrough == ["--robot", "pr2"]

    def test_separator_at_the_end_leaves_passthrough_empty(self):
        split = split_passthrough_arguments(["demo.py", "--name", "kitchen", "--"])

        assert split.own == ["demo.py", "--name", "kitchen"]
        assert split.passthrough == []


# %% asset and tick hooks
class TestAssetHookMethods:
    """
    The methods ``install_asset_hooks``/``install_tick_hook`` patch in.

    Exercised directly with fake ``original`` callables, so none of the real
    semantic_digital_twin/giskardpy classes need to be monkeypatched here.
    """

    def test_a_resolution_is_remembered_and_returned(self):
        recorder = Recorder()

        result = recorder._remember_resolution(
            lambda resolver, uri: "/opt/pkg/cup.stl",
            "the-resolver",
            "package://pkg/cup.stl",
        )

        assert result == "/opt/pkg/cup.stl"
        assert recorder.resolutions == {"package://pkg/cup.stl": "/opt/pkg/cup.stl"}

    def test_a_urdf_source_is_recorded_once(self):
        recorder = Recorder()
        original = lambda cls, file_path, **kwargs: "parsed"

        first = recorder._remember_urdf_source(original, "the-cls", "robot.urdf")
        recorder._remember_urdf_source(original, "the-cls", "robot.urdf")

        assert first == "parsed"
        assert recorder.urdf_sources == ["robot.urdf"]

    def test_a_mesh_source_is_recorded_once(self):
        recorder = Recorder()
        original = lambda stl_parser, file_path, *args, **kwargs: None

        recorder._remember_mesh_source(original, "the-parser", "cup.stl")
        recorder._remember_mesh_source(original, "the-parser", "cup.stl")

        assert recorder.mesh_sources == ["cup.stl"]

    def test_the_tick_hook_forwards_to_the_original_and_records_the_frame(self):
        recorder = Recorder()
        recorded_executors = []
        recorder.record_frame = recorded_executors.append

        result = recorder._record_tick(lambda executor: "ticked", "the-executor")

        assert result == "ticked"
        assert recorded_executors == ["the-executor"]


# %% movement detection


# %% spawned primitive boxes
@dataclass
class ShapeSpecification:
    """
    The shape collection a body specification carries.
    """

    shapes: List[Any] = field(default_factory=list)


@dataclass
class BodyBlueprint:
    """
    A body specification, of which the recorder reads only name and shapes.
    """

    name: str
    shapes: ShapeSpecification


def box_specification(name: str = "crate", scale=(0.4, 0.3, 0.2)) -> BodyBlueprint:
    """
    A specification describing exactly one box shape.

    :param name: The specification's name.
    :param scale: The box extents in metres.
    """
    return BodyBlueprint(
        name=name,
        shapes=ShapeSpecification(
            shapes=[Box(scale=Scale(*scale), color=Color(R=1.0, G=0.5, B=0.0))]
        ),
    )


class TestSpawnedBox:
    def test_a_single_box_shape_is_recordable(self):
        spawned = SpawnedBox.of_specification(box_specification())

        assert spawned == SpawnedBox(
            name="crate", scale=[0.4, 0.3, 0.2], color="#ff8000"
        )

    def test_the_spawn_time_name_override_wins(self):
        spawned = SpawnedBox.of_specification(box_specification(), name="crate_2")

        assert spawned.name == "crate_2"

    def test_a_specification_that_is_not_one_box_is_not_recordable(self):
        """
        Only a lone box has a geometry the bundle can describe with three numbers.
        """
        two_boxes = box_specification()
        two_boxes.shapes.shapes.append(
            Box(scale=Scale(1, 1, 1), color=Color(R=0.0, G=0.0, B=0.0))
        )

        assert SpawnedBox.of_specification(two_boxes) is None
        assert (
            SpawnedBox.of_specification(BodyBlueprint("empty", ShapeSpecification()))
            is None
        )


class TestRememberSpawnedBox:
    def test_a_spawned_box_is_remembered_and_materialized(self):
        recorder = Recorder()
        materialized = object()

        result = recorder._remember_spawned_box(
            lambda specification, name: materialized, box_specification()
        )

        assert result is materialized
        assert [spawned.name for spawned in recorder.spawned_boxes] == ["crate"]

    def test_the_same_box_spawned_twice_is_recorded_once(self):
        recorder = Recorder()
        original = lambda specification, name: None

        recorder._remember_spawned_box(original, box_specification())
        recorder._remember_spawned_box(original, box_specification())

        assert len(recorder.spawned_boxes) == 1

    def test_two_boxes_spawned_from_one_specification_are_both_recorded(self):
        """
        The same specification is routinely materialized several times under different
        names, and each body needs its own recorded pose.
        """
        recorder = Recorder()
        original = lambda specification, name: None

        recorder._remember_spawned_box(original, box_specification(), "crate_a")
        recorder._remember_spawned_box(original, box_specification(), "crate_b")

        assert [spawned.name for spawned in recorder.spawned_boxes] == [
            "crate_a",
            "crate_b",
        ]

    def test_a_mesh_body_is_not_recorded_as_a_box(self):
        recorder = Recorder()

        recorder._remember_spawned_box(
            lambda specification, name: None,
            BodyBlueprint("milk", ShapeSpecification()),
        )

        assert recorder.spawned_boxes == []


# %% remembering non-URDF model sources
class TestModelSourceHooks:
    def test_a_gazebo_source_is_recorded_once(self):
        recorder = Recorder()
        original = lambda cls, file_path, **kwargs: "parsed"

        first = recorder._remember_gazebo_source(original, "the-cls", "world.sdf")
        recorder._remember_gazebo_source(original, "the-cls", "world.sdf")

        assert first == "parsed"
        assert recorder.gazebo_sources == ["world.sdf"]

    def test_an_mjcf_source_is_recorded_once(self):
        recorder = Recorder()
        original = lambda parser, file_path, *args, **kwargs: None

        recorder._remember_mjcf_source(original, "the-parser", "lab.xml")
        recorder._remember_mjcf_source(original, "the-parser", "lab.xml")

        assert recorder.mjcf_sources == ["lab.xml"]


PATCHED_METHODS = (
    (PackageUriResolver, "resolve"),
    (URDFParser, "from_file"),
    (GazeboParser, "from_file"),
    (MJCFParser, "__init__"),
    (STLParser, "__init__"),
    (BodySpecification, "to_domain_object"),
)
"""
Every method ``install_asset_hooks`` replaces, as ``(owner, name)``.
"""


def patched_methods_now() -> dict:
    """
    The methods currently installed on the patched classes.

    Read statically, since ``getattr`` on a classmethod builds a fresh bound method
    every time and would never compare equal to itself.
    """
    return {
        (owner, name): inspect.getattr_static(owner, name)
        for owner, name in PATCHED_METHODS
    }


class TestAssetHookLifecycle:
    def test_uninstalling_restores_every_patched_method(self):
        """
        Bundling re-parses the recorded sources, so the hooks must be gone by then or
        the re-parse is recorded as another source to bundle.
        """
        before = patched_methods_now()
        recorder = Recorder()
        recorder.install_asset_hooks()
        try:
            assert patched_methods_now() != before
        finally:
            recorder.uninstall_asset_hooks()

        assert patched_methods_now() == before

    def test_uninstalling_twice_is_harmless(self):
        recorder = Recorder()
        recorder.install_asset_hooks()
        recorder.uninstall_asset_hooks()
        recorder.uninstall_asset_hooks()

        assert recorder._asset_hook_uninstallers == []


class TestBundledModel:
    def test_a_bundled_model_becomes_a_scene_model_entry(self, tmp_path):
        """
        The ``models`` entry a bundled model contributes to ``scene.json``.
        """
        report = BundledWorld.of_mjcf_source(
            str(_written(tmp_path / "lab.xml", MJCF_SOURCE)),
            "lab",
            str(tmp_path / "bundle"),
        )
        bundled = BundledModel(
            name="lab", prefix="lab_1", is_robot=False, report=report
        )

        assert bundled.to_payload() == {
            "name": "lab",
            "urdf": "lab.urdf",
            "prefix": "lab_1",
            "robot": False,
            "links": len(report.links),
            "movableJoints": report.movable_joints,
        }


class TestSceneIndexEntry:
    def test_a_bundle_is_indexed_with_its_robot_and_environment(self, tmp_path):
        """
        The viewer's pickers resolve a (robot, environment) pair back to a bundle, so
        the index has to carry both per scene.
        """
        bundle = tmp_path / "lab_scene"
        bundle.mkdir()
        (bundle / "scene.json").write_text(
            json.dumps(
                {
                    "robot": {"name": "pr2"},
                    "models": [
                        {"name": "pr2", "robot": True},
                        {"name": "kitchen", "robot": False},
                        {"name": "table", "robot": False},
                    ],
                }
            )
        )

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.to_payload() == {
            "name": "lab_scene",
            "robot": "pr2",
            "environment": "kitchen+table",
        }

    def test_a_bench_only_bundle_has_no_environment(self, tmp_path):
        bundle = tmp_path / "bench"
        bundle.mkdir()
        (bundle / "scene.json").write_text(
            json.dumps(
                {
                    "robot": {"name": "tracy"},
                    "models": [{"name": "tracy", "robot": True}],
                }
            )
        )

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.environment is None

    def test_a_directory_without_a_scene_file_is_skipped(self, tmp_path):
        (tmp_path / "not_a_bundle").mkdir()

        assert SceneIndexEntry.of_directory(tmp_path) == []

    def test_the_reserved_live_scene_name_is_skipped(self, tmp_path):
        """
        A live-attach snapshot (:mod:`cramera.live.live_bundle`) is a throwaway bundle
        rebuilt on every attach, never something a user onboarded — it must never show
        up as a robot/environment choice in the real picker.
        """
        bundle = tmp_path / paths.LIVE_SCENE_NAME
        bundle.mkdir()
        (bundle / "scene.json").write_text(
            json.dumps({"robot": {"name": "pr2"}, "models": []})
        )

        assert SceneIndexEntry.of_directory(tmp_path) == []


# %% bundling a parsed world
def _written(path: Path, text: str) -> Path:
    """
    Write a source file and return its path.

    :param path: Where the file goes.
    :param text: Its content.
    """
    path.write_text(text)
    return path


MJCF_SOURCE = """<mujoco model="lab">
  <worldbody>
    <body name="table" pos="1 0 0">
      <geom name="top" type="box" size="0.5 0.3 0.02"/>
      <body name="lid" pos="0 0 0.1">
        <joint name="hinge" type="hinge" axis="0 0 1" range="0 1.57"/>
        <geom name="lid_geom" type="box" size="0.1 0.1 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""
"""
An MJCF scene with one fixed and one hinged body, and no mesh references.
"""


class TestBundleParsedWorld:
    @pytest.fixture()
    def mjcf_source(self, tmp_path) -> str:
        source = tmp_path / "lab.xml"
        source.write_text(MJCF_SOURCE)
        return str(source)

    def test_an_mjcf_scene_becomes_a_loadable_urdf(self, mjcf_source, tmp_path):
        """
        The viewer only knows how to load URDF, so an MJCF source has to come out the
        other side as one, with its kinematics preserved.
        """
        report = BundledWorld.of_mjcf_source(
            mjcf_source, "lab", str(tmp_path / "bundle")
        )

        assert report.links == ["world", "table", "lid"]
        assert report.movable_joints == ["hinge"]
        urdf = Path(report.urdf).read_text()
        assert bundler.BundleReport.LINK_PATTERN.findall(urdf) == report.links
        assert (
            dict(bundler.BundleReport.JOINT_PATTERN.findall(urdf))["hinge"]
            == "revolute"
        )

    def test_the_report_names_the_source_it_was_built_from(self, mjcf_source, tmp_path):
        report = BundledWorld.of_mjcf_source(
            mjcf_source, "lab", str(tmp_path / "bundle")
        )

        assert report.source == mjcf_source

    def test_a_missing_source_is_reported_as_such(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="MJCF source not found"):
            BundledWorld.of_mjcf_source(
                str(tmp_path / "gone.xml"), "lab", str(tmp_path / "bundle")
            )

    def test_a_missing_gazebo_source_names_its_own_format(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Gazebo source not found"):
            BundledWorld.of_gazebo_source(
                str(tmp_path / "gone.sdf"), "world", str(tmp_path / "bundle")
            )


# %% serializing bodies that no source claimed
class TestSerializeUnclaimedBodies:
    """
    A world built in code -- bodies constructed directly instead of parsed out of a
    URDF, MJCF or SDF file -- leaves no source to bundle, so the bodies themselves have
    to become a model.

    Only the bodies no parsed model already claims are serialized, which
    is what makes a root of their own necessary: their parent may be a body this document
    does not contain.
    """

    @pytest.fixture()
    def hand_built_world(self) -> World:
        """
        A world whose floor belongs to a parsed model, with a table and its drawer built
        in code on top of it.
        """
        world = World()
        floor = Body(name=PrefixedName("floor"))
        table = Body(
            name=PrefixedName("table"),
            visual=ShapeCollection(shapes=[Box(scale=Scale(1.0, 0.6, 0.7))]),
        )
        drawer = Body(
            name=PrefixedName("drawer"),
            visual=ShapeCollection(shapes=[Box(scale=Scale(0.3, 0.3, 0.2))]),
        )
        drawer_dof = DegreeOfFreedom(name=PrefixedName("drawer_dof"))
        with world.modify_world():
            world.add_kinematic_structure_entity(floor)
            world.add_kinematic_structure_entity(table)
            world.add_kinematic_structure_entity(drawer)
            world.add_degree_of_freedom(drawer_dof)
            world.add_connection(
                FixedConnection(
                    parent=floor,
                    child=table,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=0.4, y=0.2, z=0.0
                    ),
                )
            )
            world.add_connection(
                PrismaticConnection(
                    parent=table,
                    child=drawer,
                    axis=Vector3.from_iterable([1, 0, 0]),
                    raw_dof=drawer_dof,
                )
            )
        return world

    def serialize(self, world: World, tmp_path) -> bundler.BundleReport:
        """
        Serialize everything but the floor, as a bundler would for a world whose only
        parsed source claimed the floor.
        """
        return UrdfDocument.of_bodies(
            bodies=[
                world.get_body_by_name("table"),
                world.get_body_by_name("drawer"),
            ],
            name="environment",
            output_directory=str(tmp_path / "bundle"),
            mesh_subdirectory="environment",
        )

    def test_the_subset_becomes_a_urdf_rooted_in_one_link(
        self, hand_built_world, tmp_path
    ):
        report = self.serialize(hand_built_world, tmp_path)

        urdf = Path(report.urdf).read_text()
        assert bundler.BundleReport.LINK_PATTERN.findall(urdf) == [
            UrdfDocument.SYNTHESIZED_ROOT_LINK,
            "table",
            "drawer",
        ]

    def test_a_connection_inside_the_subset_keeps_animating(
        self, hand_built_world, tmp_path
    ):
        """
        The drawer's joint is what the recorded connection positions drive, so it has to
        survive as a prismatic joint under the name the recording keys it by.
        """
        report = self.serialize(hand_built_world, tmp_path)

        # the key Recorder.record_frame writes the drawer's position under
        recorded_key = str(
            hand_built_world.get_body_by_name("drawer").parent_connection.name
        )
        assert report.movable_joints == [recorded_key]
        urdf = Path(report.urdf).read_text()
        assert dict(bundler.BundleReport.JOINT_PATTERN.findall(urdf))[recorded_key] == (
            "prismatic"
        )

    def test_a_body_whose_parent_is_absent_is_grafted_on_at_its_world_pose(
        self, hand_built_world, tmp_path
    ):
        """
        The table's parent is the floor, which this document does not contain, so the
        table hangs off the synthesized root -- and has to keep the place it stood in.
        """
        report = self.serialize(hand_built_world, tmp_path)

        urdf = ElementTree.fromstring(Path(report.urdf).read_text())
        graft = [
            joint
            for joint in urdf.findall("joint")
            if joint.find("child").attrib["link"] == "table"
        ]
        assert len(graft) == 1
        assert graft[0].attrib["type"] == "fixed"
        assert graft[0].find("parent").attrib["link"] == (
            UrdfDocument.SYNTHESIZED_ROOT_LINK
        )
        table_position = hand_built_world.get_body_by_name("table").global_pose.to_np()[
            :3, 3
        ]
        assert [
            float(value) for value in graft[0].find("origin").attrib["xyz"].split()
        ] == pytest.approx(list(table_position))

    def test_the_written_urdf_parses_back_into_a_world(
        self, hand_built_world, tmp_path
    ):
        """
        The viewer only ever loads URDF, so the document has to be a well-formed one:

        a single root, and no joint naming a link it does not contain.
        """
        report = self.serialize(hand_built_world, tmp_path)

        reparsed = URDFParser.from_file(report.urdf).parse()

        assert sorted(str(body.name).split("/")[-1] for body in reparsed.bodies) == (
            sorted(["table", "drawer", UrdfDocument.SYNTHESIZED_ROOT_LINK])
        )

    def test_a_joint_at_a_nonzero_position_is_written_at_its_zero(
        self, hand_built_world, tmp_path
    ):
        """
        URDF reads a joint as ``origin`` followed by the joint's own displacement, and
        the viewer supplies that displacement from the recording.

        So a world whose joints are already displaced when it is bundled -- an MJCF
        keyframe puts the Panda's arm in a home pose, for instance -- must still be
        written at its zero, or the recorded value is applied on top of the displacement
        that is already baked in and the joint ends up moving from the wrong place.
        """
        drawer = hand_built_world.get_body_by_name("drawer")
        drawer.parent_connection.position = 0.25
        zero_origin = drawer.parent_connection.parent_T_connection_expression.to_np()

        report = self.serialize(hand_built_world, tmp_path)

        urdf = ElementTree.fromstring(Path(report.urdf).read_text())
        [joint] = [
            element
            for element in urdf.findall("joint")
            if element.find("child").attrib["link"] == "drawer"
        ]
        written = [float(value) for value in joint.find("origin").attrib["xyz"].split()]
        assert written == pytest.approx(list(zero_origin[:3, 3]), abs=1e-6)

    def test_the_report_claims_the_real_bodies_only(self, hand_built_world, tmp_path):
        """
        The report's links say which bodies are now covered by a model, so the
        synthesized root -- which is no body of the world -- must not appear among them.
        """
        report = self.serialize(hand_built_world, tmp_path)

        assert report.links == ["table", "drawer"]


# %% loose objects of a world built in code
class TestFreeFloatingObjects:
    """
    A world built in code loads no mesh file and spawns no box, so nothing tells the
    recorder which of its bodies are the loose ones.

    The world itself does: a body it
    holds by a ``Connection6DoF`` is one it lets move freely.
    """

    def world_with(self, *, mobile_robot: bool) -> World:
        """
        A world with a fixed table, a free-floating cube, and a robot base that is
        either bolted down or free to drive.
        """
        world = World()
        floor = Body(name=PrefixedName("floor"))
        table = Body(
            name=PrefixedName("table"),
            visual=ShapeCollection(shapes=[Box(scale=Scale(1.0, 0.6, 0.7))]),
        )
        cube = Body(
            name=PrefixedName("cube", prefix="montessori"),
            visual=ShapeCollection(shapes=[Box(scale=Scale(0.05, 0.05, 0.05))]),
        )
        robot_base = Body(name=PrefixedName("base_link"))
        with world.modify_world():
            for body in (floor, table, cube, robot_base):
                world.add_kinematic_structure_entity(body)
            world.add_connection(FixedConnection(parent=floor, child=table))
            world.add_connection(
                Connection6DoF.create_with_dofs(parent=floor, child=cube, world=world)
            )
            if mobile_robot:
                world.add_connection(
                    Connection6DoF.create_with_dofs(
                        parent=floor, child=robot_base, world=world
                    )
                )
            else:
                world.add_connection(FixedConnection(parent=floor, child=robot_base))
        return world

    def test_a_freely_connected_body_is_a_loose_object(self):
        recorder = Recorder()
        recorder.world = self.world_with(mobile_robot=False)

        assert [str(body.name) for body in recorder.free_floating_bodies()] == [
            "montessori/cube"
        ]

    def test_the_key_drops_the_world_prefix(self):
        """
        Poses are filed under the same bare key a mesh file or a spawned box would use,
        so the viewer looks every object up the same way.
        """
        recorder = Recorder()
        recorder.world = self.world_with(mobile_robot=False)

        [cube] = recorder.free_floating_bodies()
        assert recorder.object_key(cube) == "cube"

    def test_a_driving_robot_is_not_mistaken_for_an_object(self):
        """
        A mobile base is free-floating too, and it is recorded as the robot rather than
        as something the robot manipulates.
        """
        world = self.world_with(mobile_robot=True)
        recorder = Recorder()
        recorder.world = world

        class StationaryAnnotation:
            """
            Stands in for the robot annotation, which only has to name its bodies.
            """

            bodies = [world.get_body_by_name("base_link")]

        recorder.robot = StationaryAnnotation()

        assert [str(body.name) for body in recorder.free_floating_bodies()] == [
            "montessori/cube"
        ]


# %% the model synthesized for bodies no source described
class TestBundleUnclaimedBodies:
    """
    Every model comes from a recorded source, so a world built in code produces none and
    the viewer draws an empty scene.

    The bodies left over after the recorded sources have claimed theirs become one model
    of their own.
    """

    @pytest.fixture()
    def builder(self, tmp_path) -> SceneBuilder:
        world = World()
        floor = Body(name=PrefixedName("floor"))
        table = Body(
            name=PrefixedName("table"),
            visual=ShapeCollection(shapes=[Box(scale=Scale(1.0, 0.6, 0.7))]),
        )
        cube = Body(
            name=PrefixedName("cube"),
            visual=ShapeCollection(shapes=[Box(scale=Scale(0.05, 0.05, 0.05))]),
        )
        with world.modify_world():
            for body in (floor, table, cube):
                world.add_kinematic_structure_entity(body)
            world.add_connection(FixedConnection(parent=floor, child=table))
            world.add_connection(FixedConnection(parent=floor, child=cube))
        recorder = recording([{}])
        recorder.world = world
        return SceneBuilder(recorder, "scene", str(tmp_path / "bundle"), 1)

    def parsed_model(self, links: List[str]) -> BundledModel:
        """
        A model as bundling a recorded source would report it.
        """
        return BundledModel(
            name="robot",
            prefix="",
            is_robot=True,
            report=bundler.BundleReport(
                name="robot",
                urdf="robot.urdf",
                source="robot.urdf",
                links=links,
                joints=[],
                movable_joints=[],
                meshes_copied=0,
                mesh_suffixes=[],
                references_rewritten=0,
                missing=[],
            ),
        )

    def test_the_bodies_no_source_described_become_one_model(self, builder):
        environment = builder._bundle_unclaimed_bodies(
            [self.parsed_model(["floor"])], objects=[]
        )

        assert environment is not None
        assert environment.name == SceneBuilder.ENVIRONMENT_MODEL_NAME
        assert environment.is_robot is False
        # the URDF states its own tree through joints, so link order carries no meaning
        assert sorted(environment.report.links) == ["cube", "table"]

    def test_a_world_a_source_fully_described_gets_no_extra_model(self, builder):
        """
        A demo that loads its world from files must keep bundling exactly the models it
        loaded, with nothing synthesized beside them.
        """
        assert (
            builder._bundle_unclaimed_bodies(
                [self.parsed_model(["floor", "table", "cube"])], objects=[]
            )
            is None
        )

    def test_a_tracked_object_is_left_out_of_the_environment(self, builder):
        """
        A tracked object is drawn from its own geometry and moved every frame, so
        leaving it in the environment as well would draw a second, motionless copy of
        it.
        """
        environment = builder._bundle_unclaimed_bodies(
            [self.parsed_model(["floor"])], objects=[{"key": "cube"}]
        )

        assert environment.report.links == ["table"]


# %% the executed plan tree
@dataclass
class RecordedStatus:
    """
    A plan node's status, of which the serializer reads only the name.
    """

    name: str


@dataclass
class RecordedPlanNode:
    """
    A plan node as the serializer walks it: a status, a parent and ordered children.
    """

    status: RecordedStatus = field(default_factory=lambda: RecordedStatus("SUCCEEDED"))
    parent: Optional["RecordedPlanNode"] = None
    children: List["RecordedPlanNode"] = field(default_factory=list)

    def with_children(self, *children: "RecordedPlanNode") -> "RecordedPlanNode":
        for child in children:
            child.parent = self
        self.children = list(children)
        return self


class TestSerializePlans:
    def test_a_tree_is_serialized_from_the_root_of_any_recorded_node(self):
        """
        Recording a leaf is enough: the serializer walks up to the root and emits the
        whole tree from there, once.
        """
        leaf = RecordedPlanNode()
        root = RecordedPlanNode().with_children(RecordedPlanNode().with_children(leaf))
        recorder = Recorder(plan_nodes=[leaf, root])

        [tree] = recorder.serialize_plans()

        assert tree["kind"] == "RecordedPlanNode"
        assert tree["status"] == "SUCCEEDED"
        assert len(tree["children"]) == 1
        assert len(tree["children"][0]["children"]) == 1

    def test_serialization_stops_at_the_node_cap(self):
        root = RecordedPlanNode().with_children(*(RecordedPlanNode() for _ in range(5)))
        recorder = Recorder(plan_nodes=[root])

        [tree] = recorder.serialize_plans(max_nodes=3)

        assert len(tree["children"]) == 2  # the root itself counts towards the cap

    def test_the_cap_defaults_to_the_recorders_own_limit(self):
        root = RecordedPlanNode().with_children(
            *(
                RecordedPlanNode()
                for _ in range(Recorder.MAX_SERIALIZED_PLAN_NODES + 10)
            )
        )
        recorder = Recorder(plan_nodes=[root])

        [tree] = recorder.serialize_plans()

        assert len(tree["children"]) == Recorder.MAX_SERIALIZED_PLAN_NODES - 1


class TestMovementDetection:
    def test_a_pose_is_unmoved_within_the_tolerance(self):
        assert RecordingAnalysis.has_moved(pose_at(0, 0), pose_at(0.01, 0.0)) is False

    def test_planar_travel_counts_as_movement(self):
        assert RecordingAnalysis.has_moved(pose_at(0, 0), pose_at(0.5, 0.0)) is True

    def test_vertical_travel_counts_as_movement(self):
        assert (
            RecordingAnalysis.has_moved(pose_at(0, 0, 1.0), pose_at(0, 0, 1.5)) is True
        )

    def test_the_tolerance_is_configurable(self):
        assert (
            RecordingAnalysis.has_moved(pose_at(0, 0), pose_at(0.5, 0.0), tolerance=1.0)
            is False
        )


# %% transport windows
class TestObjectWindows:
    def test_an_object_that_never_moves_has_no_window(self):
        recorder = recording([{"milk.stl": RESTING} for _ in range(5)])
        assert RecordingAnalysis(recorder).object_windows() == []

    def test_a_transported_object_reports_its_travel_window(self):
        """
        The window starts at the first frame that differs from the spawn pose and ends
        one past the last frame that differs from the final pose.
        """
        frames = [
            {"milk.stl": pose_at(0, 0)},
            {"milk.stl": pose_at(0, 0)},
            {"milk.stl": pose_at(1, 0)},
            {"milk.stl": pose_at(2, 0)},
            {"milk.stl": pose_at(2, 0)},
        ]
        window = RecordingAnalysis(recording(frames)).object_windows()[0]
        assert window["object"] == "milk.stl"
        assert window["attach"] == 2
        assert window["detach"] == 3
        assert window["place"] == [2, 0, 1.0]

    def test_an_instant_jump_yields_no_window(self):
        """
        An object that is already at its destination the frame after it leaves the spawn
        has an empty window, so it is not reported as a transport.
        """
        frames = [{"milk.stl": pose_at(0, 0)} for _ in range(3)]
        frames += [{"milk.stl": pose_at(2, 0)} for _ in range(3)]
        assert RecordingAnalysis(recording(frames)).object_windows() == []

    def test_windows_are_ordered_by_when_they_start(self):
        early = [pose_at(0, 0), pose_at(1, 0), pose_at(2, 0), pose_at(3, 0)]
        early += [pose_at(4, 0), pose_at(4, 0)]
        late = [pose_at(0, 0)] * 3 + [pose_at(0, 1.5), pose_at(0, 3), pose_at(0, 3)]
        frames = [
            {"early.stl": early[index], "late.stl": late[index]} for index in range(6)
        ]
        windows = RecordingAnalysis(recording(frames)).object_windows()
        assert [window["object"] for window in windows] == ["early.stl", "late.stl"]
        assert [window["attach"] for window in windows] == [1, 3]


class TestFirstBaseMotion:
    def test_a_standing_base_reports_the_upper_bound(self):
        recorder = recording([{} for _ in range(5)])
        assert RecordingAnalysis(recorder).first_base_motion(4) == 4

    def test_the_frame_the_base_leaves_its_spawn_is_found(self):
        recorder = recording([{} for _ in range(5)])
        recorder.base_frames = [
            RESTING,
            RESTING,
            pose_at(1, 0),
            pose_at(2, 0),
            pose_at(2, 0),
        ]
        assert RecordingAnalysis(recorder).first_base_motion(5) == 2

    def test_motion_after_the_bound_is_not_reported(self):
        recorder = recording([{} for _ in range(5)])
        recorder.base_frames = [RESTING, RESTING, RESTING, pose_at(3, 0), pose_at(3, 0)]
        assert RecordingAnalysis(recorder).first_base_motion(2) == 2


# %% segment derivation
class TestDeriveSegments:
    def test_a_recording_without_transports_is_one_segment(self):
        recorder = recording(
            [{"milk.stl": RESTING} for _ in range(4)],
            actions=[{"action": "ParkArmsAction", "arm": None, "target": None}],
        )
        segments = RecordingAnalysis(recorder).derive_segments()
        assert [segment["step"] for segment in segments] == ["parkarms"]
        assert segments[0]["start"] == 0

    def test_an_unlabelled_recording_falls_back_to_one_plan_segment(self):
        recorder = recording([{"milk.stl": RESTING} for _ in range(4)])
        assert [
            segment["step"] for segment in RecordingAnalysis(recorder).derive_segments()
        ] == ["plan"]

    def test_a_transport_is_named_after_its_action_and_object(self):
        milk = [pose_at(0, 0), pose_at(0, 0), pose_at(1, 0)]
        milk += [pose_at(2, 0), pose_at(2, 0), pose_at(2, 0)]
        recorder = recording(
            [{"milk.stl": pose} for pose in milk],
            actions=[
                {"action": "TransportAction", "arm": "LEFT", "target": "milk.stl"}
            ],
        )
        transport = RecordingAnalysis(recorder).derive_segments()[-1]
        assert transport["step"] == "transport_milk"
        assert transport["picks"] == "milk"
        assert transport["arm"] == "LEFT"

    def test_segments_cover_the_recording_without_gaps(self):
        """
        Playback walks the segments in order, so each must start where the last ended.
        """
        milk = [pose_at(0, 0), pose_at(1, 0), pose_at(2, 0)] + [pose_at(2, 0)] * 5
        cup = [pose_at(5, 0)] * 4 + [pose_at(5, 1), pose_at(5, 2)] + [pose_at(5, 2)] * 2
        recorder = recording(
            [{"milk.stl": milk[index], "cup.stl": cup[index]} for index in range(8)],
            actions=[
                {"action": "TransportAction", "arm": None, "target": "milk.stl"},
                {"action": "TransportAction", "arm": None, "target": "cup.stl"},
            ],
        )
        segments = RecordingAnalysis(recorder).derive_segments()
        assert len(segments) == 2
        for earlier, later in zip(segments, segments[1:]):
            assert earlier["end"] == later["start"]


# %% URDF reference resolution
class TestResolveUri:
    def test_a_recorded_resolution_wins(self, tmp_path):
        target = tmp_path / "cup.stl"
        target.write_text("solid cup\nendsolid cup\n")
        resolved = bundler.MeshReference("package://demo/cup.stl").resolve(
            hints={"package://demo/cup.stl": str(target)}
        )
        assert resolved == str(target)

    def test_a_relative_reference_resolves_against_the_urdf(self, tmp_path):
        mesh = tmp_path / "meshes" / "cup.stl"
        mesh.parent.mkdir()
        mesh.write_text("solid cup\nendsolid cup\n")
        assert bundler.MeshReference("meshes/cup.stl").resolve(
            base_directory=str(tmp_path)
        ) == str(mesh)

    def test_a_missing_relative_reference_is_unresolved(self, tmp_path):
        assert (
            bundler.MeshReference("meshes/gone.stl").resolve(
                base_directory=str(tmp_path)
            )
            is None
        )

    def test_a_file_uri_resolves_to_its_path(self, tmp_path):
        mesh = tmp_path / "cup.stl"
        mesh.write_text("solid cup\nendsolid cup\n")
        assert bundler.MeshReference("file://" + str(mesh)).resolve() == str(mesh)

    def test_an_absolute_path_that_exists_resolves_to_itself(self, tmp_path):
        mesh = tmp_path / "cup.stl"
        mesh.write_text("solid cup\nendsolid cup\n")
        assert bundler.MeshReference(str(mesh)).resolve() == str(mesh)

    def test_an_unresolvable_package_uri_is_unresolved(self, monkeypatch):
        """
        Without a recorded hint and with no ROS installation to ask,
        :class:`PackageUriResolver` fails to resolve the package, and the URI comes back
        unresolved rather than raising.
        """
        monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
        monkeypatch.delenv("ROS_PACKAGE_PATH", raising=False)
        monkeypatch.delenv("CMAKE_PREFIX_PATH", raising=False)
        assert (
            bundler.MeshReference("package://no_such_package/cup.stl").resolve() is None
        )


class TestReferenceLayout:
    def test_a_package_reference_keeps_its_package_directory(self):
        assert bundler.MeshReference(
            "package://demo/meshes/cup.stl"
        ).bundled_relative_path() == ("demo/meshes/cup.stl")

    def test_a_local_reference_lands_in_one_flat_directory(self):
        assert (
            bundler.MeshReference("../far/away/cup.stl").bundled_relative_path()
            == "_local/cup.stl"
        )


# %% copying assets into the bundle
class TestBundledAssets:
    def test_an_asset_is_copied_once_however_often_it_is_referenced(self, tmp_path):
        source = tmp_path / "cup.stl"
        source.write_text("solid cup endsolid")
        assets = bundler.BundledAssets()

        assert assets.copy(str(source), str(tmp_path / "out" / "cup.stl")) is True
        assert assets.copy(str(source), str(tmp_path / "elsewhere" / "cup.stl")) is True

        assert assets.copied == {str(source): str(tmp_path / "out" / "cup.stl")}
        assert not (tmp_path / "elsewhere").exists()

    def test_an_unresolved_reference_is_recorded_as_missing(self, tmp_path):
        assets = bundler.BundledAssets()
        assert assets.copy(None, str(tmp_path / "out" / "cup.stl")) is False
        assert assets.missing == [bundler.BundledAssets.UNRESOLVED_REFERENCE]

    def test_a_resolved_path_that_is_not_a_file_is_recorded_as_missing(self, tmp_path):
        assets = bundler.BundledAssets()
        gone = str(tmp_path / "gone.stl")
        assert assets.copy(gone, str(tmp_path / "out" / "gone.stl")) is False
        assert assets.missing == [gone]

    def test_the_textures_a_collada_mesh_names_are_copied_beside_it(self, tmp_path):
        source_directory = tmp_path / "src"
        source_directory.mkdir()
        (source_directory / "wood.png").write_bytes(b"png")
        mesh = source_directory / "table.dae"
        mesh.write_text(
            "<library_images><init_from>wood.png</init_from></library_images>"
        )
        bundled = tmp_path / "out" / "table.dae"

        assets = bundler.BundledAssets()
        assets.copy(str(mesh), str(bundled))
        assets.copy_side_assets(str(mesh), str(bundled))

        assert (tmp_path / "out" / "wood.png").read_bytes() == b"png"
        assert assets.missing == []

    def test_an_object_meshs_material_library_and_its_textures_are_copied(
        self, tmp_path
    ):
        source_directory = tmp_path / "src"
        source_directory.mkdir()
        (source_directory / "cup.mtl").write_text("newmtl body\nmap_Kd glaze.jpg\n")
        (source_directory / "glaze.jpg").write_bytes(b"jpg")
        mesh = source_directory / "cup.obj"
        mesh.write_text("mtllib cup.mtl\nv 0 0 0\n")
        bundled = tmp_path / "out" / "cup.obj"

        assets = bundler.BundledAssets()
        assets.copy(str(mesh), str(bundled))
        assets.copy_side_assets(str(mesh), str(bundled))

        assert (tmp_path / "out" / "cup.mtl").exists()
        assert (tmp_path / "out" / "glaze.jpg").read_bytes() == b"jpg"

    def test_a_stereolithography_mesh_has_no_side_assets(self, tmp_path):
        mesh = tmp_path / "cup.stl"
        mesh.write_text("solid cup endsolid")
        bundled = tmp_path / "out" / "cup.stl"

        assets = bundler.BundledAssets()
        assets.copy(str(mesh), str(bundled))
        assets.copy_side_assets(str(mesh), str(bundled))

        assert list(assets.copied) == [str(mesh)]

    def test_a_texture_beside_the_mesh_keeps_its_relative_location(self, tmp_path):
        """
        Gazebo model trees reference textures from a sibling directory, e.g.
        ``../materials/textures/wall.png``.

        The reference has to be resolved against the mesh and mirrored at the same
        relative place next to the bundled copy, or the browser asks for a file that is
        not there.
        """
        model = tmp_path / "model"
        (model / "meshes").mkdir(parents=True)
        (model / "materials" / "textures").mkdir(parents=True)
        (model / "materials" / "textures" / "wall.png").write_bytes(b"png")
        mesh = model / "meshes" / "wall.dae"
        mesh.write_text("<init_from>../materials/textures/wall.png</init_from>")
        bundled = tmp_path / "bundle" / "meshes" / "model" / "wall.dae"

        assets = bundler.BundledAssets(bundle_root=str(tmp_path / "bundle"))
        assets.copy(str(mesh), str(bundled))
        assets.copy_side_assets(str(mesh), str(bundled))

        texture = tmp_path / "bundle" / "meshes" / "materials" / "textures" / "wall.png"
        assert texture.read_bytes() == b"png"

    def test_a_reference_escaping_the_bundle_is_skipped(self, tmp_path):
        """
        A mesh sitting at the top of the bundle's mesh tree could otherwise write
        outside the bundle entirely.
        """
        source_directory = tmp_path / "src"
        source_directory.mkdir()
        (tmp_path / "outside.png").write_bytes(b"png")
        mesh = source_directory / "wall.dae"
        mesh.write_text("<init_from>../outside.png</init_from>")
        bundled = tmp_path / "bundle" / "wall.dae"

        assets = bundler.BundledAssets(bundle_root=str(tmp_path / "bundle"))
        assets.copy(str(mesh), str(bundled))
        assets.copy_side_assets(str(mesh), str(bundled))

        assert list(assets.copied) == [str(mesh)]

    def test_the_mesh_suffixes_are_sorted_and_deduplicated(self, tmp_path):
        assets = bundler.BundledAssets()
        for name in ("b.STL", "a.stl", "c.dae"):
            source = tmp_path / name
            source.write_text("x")
            assets.copy(str(source), str(tmp_path / "out" / name))
        assert assets.mesh_suffixes == [".dae", ".stl"]


# %% bundling a URDF
class TestBundleUrdf:
    @pytest.fixture()
    def source_tree(self, tmp_path):
        """
        A URDF referencing one mesh, both on disk next to each other.
        """
        (tmp_path / "meshes").mkdir()
        (tmp_path / "meshes" / "cup.stl").write_text("solid cup\nendsolid cup\n")
        urdf = tmp_path / "robot.urdf"
        urdf.write_text(ONE_MESH_URDF_TEXT)
        return urdf

    @pytest.fixture()
    def xacro_source_tree(self, tmp_path):
        """
        The same URDF content as :attr:`source_tree`, saved with a ``.xacro`` extension.
        """
        (tmp_path / "meshes").mkdir()
        (tmp_path / "meshes" / "cup.stl").write_text("solid cup\nendsolid cup\n")
        xacro = tmp_path / "robot.xacro"
        xacro.write_text(ONE_MESH_URDF_TEXT)
        return xacro

    def test_the_mesh_is_copied_next_to_the_rewritten_urdf(self, source_tree, tmp_path):
        output_directory = tmp_path / "bundle"
        report = bundler.BundleReport.of_source(
            str(source_tree), "demo", str(output_directory)
        )
        assert (output_directory / "demo.urdf").is_file()
        assert (output_directory / "meshes" / "_local" / "cup.stl").is_file()
        assert report.meshes_copied == 1
        assert report.missing == []

    def test_the_reference_is_rewritten_to_the_bundled_copy(
        self, source_tree, tmp_path
    ):
        output_directory = tmp_path / "bundle"
        bundler.BundleReport.of_source(str(source_tree), "demo", str(output_directory))
        rewritten = (output_directory / "demo.urdf").read_text()
        assert 'filename="meshes/_local/cup.stl"' in rewritten
        assert 'filename="meshes/cup.stl"' not in rewritten

    def test_links_and_joints_are_reported(self, source_tree, tmp_path):
        report = bundler.BundleReport.of_source(
            str(source_tree), "demo", str(tmp_path / "bundle")
        )
        assert report.links == ["base_link", "cup_link"]
        assert report.joints == ["cup_joint"]
        assert report.movable_joints == []

    def test_a_xacro_source_is_bundled_like_a_urdf_source(
        self, xacro_source_tree, tmp_path
    ):
        """
        Bundling a xacro source produces the same links, joints and mesh copy as
        bundling the equivalent URDF - the ElementTree round-trip
        :meth:`URDFParser.from_xacro` performs does not break the regex-based mesh
        rewriting.
        """
        report = bundler.BundleReport.of_source(
            str(xacro_source_tree), "demo", str(tmp_path / "bundle")
        )
        assert report.links == ["base_link", "cup_link"]
        assert report.joints == ["cup_joint"]
        assert report.meshes_copied == 1
        assert report.missing == []

    def test_an_unresolvable_mesh_is_reported_as_missing(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text(
            '<robot name="demo">\n'
            '  <link name="base_link">\n'
            '    <visual><geometry><mesh filename="meshes/gone.stl"/></geometry></visual>\n'
            "  </link>\n"
            "</robot>\n"
        )
        report = bundler.BundleReport.of_source(
            str(urdf), "demo", str(tmp_path / "bundle")
        )
        assert report.missing == [bundler.BundledAssets.UNRESOLVED_REFERENCE]
        assert report.meshes_copied == 0

    def test_a_missing_source_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bundler.BundleReport.of_source(
                str(tmp_path / "gone.urdf"), "demo", str(tmp_path)
            )


class TestUnsupportedConnections:
    """
    A body behind a connection URDF cannot express must survive serialization: it is
    grafted onto the document root at its world pose instead of crashing the bundle.
    """

    def drive_world(self):
        """
        A world whose robot base hangs on an omnidirectional drive.
        """
        world = World()
        root = Body(name=PrefixedName("root", prefix="world"))
        base = Body(name=PrefixedName("base_link", prefix="pr2"))
        with world.modify_world():
            world.add_body(root)
            world.add_connection(
                OmniDrive.create_with_dofs(parent=root, child=base, world=world)
            )
        return world, root, base

    def test_an_omnidirectional_drive_becomes_a_floating_joint(self, tmp_path):
        world, root, base = self.drive_world()

        report = UrdfDocument.of_bodies(
            bodies=[root, base],
            name="environment",
            output_directory=str(tmp_path / "bundle"),
            mesh_subdirectory="environment",
        )

        urdf = Path(report.urdf).read_text()
        assert 'type="floating"' in urdf
        assert report.movable_joints == [str(base.parent_connection.name)]

    def test_a_connection_without_a_joint_type_grafts_the_child(
        self, tmp_path, monkeypatch
    ):
        world, root, base = self.drive_world()
        connection_types = dict(UrdfDocument.CONNECTION_JOINT_TYPES)
        del connection_types[OmniDrive]
        monkeypatch.setattr(
            UrdfDocument, "CONNECTION_JOINT_TYPES", connection_types
        )

        report = UrdfDocument.of_bodies(
            bodies=[root, base],
            name="environment",
            output_directory=str(tmp_path / "bundle"),
            mesh_subdirectory="environment",
        )

        urdf = Path(report.urdf).read_text()
        graft_name = "%s_to_%s" % (UrdfDocument.SYNTHESIZED_ROOT_LINK, "pr2/base_link")
        assert graft_name in urdf
