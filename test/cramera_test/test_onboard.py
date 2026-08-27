"""
Tests for the onboarder's pure post-processing and the URDF asset bundler.

Recording itself needs a running coraplex demo, but everything that turns a recording
into a scene bundle is plain data work: deciding when an object moved, finding the
attach/detach window of each transport, labelling the resulting segments, and making a
URDF self-contained. Those are covered here against hand-built recordings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from typing_extensions import Any, Dict, List, Optional

from cramera.onboard import bundle_urdf as bundler
from cramera.onboard.demo import Recorder, RecordingAnalysis

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
