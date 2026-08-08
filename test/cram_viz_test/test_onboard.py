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
from typing_extensions import Any, Dict, List

from cram_viz.onboard import bundle_urdf as bundler
from cram_viz.onboard.demo import (
    Recorder,
    derive_segments,
    first_base_motion,
    link_set,
    moved,
    object_windows,
)

#: a pose that stays put, used wherever a frame's value must not matter
RESTING = [0.0, 0.0, 1.0, 0, 0, 0, 1]

#: a URDF referencing exactly one mesh, shared by the URDF- and xacro-source bundling tests
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
class TestMovementDetection:
    def test_a_pose_is_unmoved_within_the_tolerance(self):
        assert moved(pose_at(0, 0), pose_at(0.01, 0.0)) is False

    def test_planar_travel_counts_as_movement(self):
        assert moved(pose_at(0, 0), pose_at(0.5, 0.0)) is True

    def test_vertical_travel_counts_as_movement(self):
        assert moved(pose_at(0, 0, 1.0), pose_at(0, 0, 1.5)) is True

    def test_the_tolerance_is_configurable(self):
        assert moved(pose_at(0, 0), pose_at(0.5, 0.0), tolerance=1.0) is False


# %% transport windows
class TestObjectWindows:
    def test_an_object_that_never_moves_has_no_window(self):
        recorder = recording([{"milk.stl": RESTING} for _ in range(5)])
        assert object_windows(recorder) == []

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
        window = object_windows(recording(frames))[0]
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
        assert object_windows(recording(frames)) == []

    def test_windows_are_ordered_by_when_they_start(self):
        early = [pose_at(0, 0), pose_at(1, 0), pose_at(2, 0), pose_at(3, 0)]
        early += [pose_at(4, 0), pose_at(4, 0)]
        late = [pose_at(0, 0)] * 3 + [pose_at(0, 1.5), pose_at(0, 3), pose_at(0, 3)]
        frames = [
            {"early.stl": early[index], "late.stl": late[index]} for index in range(6)
        ]
        windows = object_windows(recording(frames))
        assert [window["object"] for window in windows] == ["early.stl", "late.stl"]
        assert [window["attach"] for window in windows] == [1, 3]


class TestFirstBaseMotion:
    def test_a_standing_base_reports_the_upper_bound(self):
        recorder = recording([{} for _ in range(5)])
        assert first_base_motion(recorder, 4) == 4

    def test_the_frame_the_base_leaves_its_spawn_is_found(self):
        recorder = recording([{} for _ in range(5)])
        recorder.base_frames = [
            RESTING,
            RESTING,
            pose_at(1, 0),
            pose_at(2, 0),
            pose_at(2, 0),
        ]
        assert first_base_motion(recorder, 5) == 2

    def test_motion_after_the_bound_is_not_reported(self):
        recorder = recording([{} for _ in range(5)])
        recorder.base_frames = [RESTING, RESTING, RESTING, pose_at(3, 0), pose_at(3, 0)]
        assert first_base_motion(recorder, 2) == 2


# %% segment derivation
class TestDeriveSegments:
    def test_a_recording_without_transports_is_one_segment(self):
        recorder = recording(
            [{"milk.stl": RESTING} for _ in range(4)],
            actions=[{"action": "ParkArmsAction", "arm": None, "target": None}],
        )
        segments = derive_segments(recorder)
        assert [segment["step"] for segment in segments] == ["parkarms"]
        assert segments[0]["start"] == 0

    def test_an_unlabelled_recording_falls_back_to_one_plan_segment(self):
        recorder = recording([{"milk.stl": RESTING} for _ in range(4)])
        assert [segment["step"] for segment in derive_segments(recorder)] == ["plan"]

    def test_a_transport_is_named_after_its_action_and_object(self):
        milk = [pose_at(0, 0), pose_at(0, 0), pose_at(1, 0)]
        milk += [pose_at(2, 0), pose_at(2, 0), pose_at(2, 0)]
        recorder = recording(
            [{"milk.stl": pose} for pose in milk],
            actions=[
                {"action": "TransportAction", "arm": "LEFT", "target": "milk.stl"}
            ],
        )
        transport = derive_segments(recorder)[-1]
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
        segments = derive_segments(recorder)
        assert len(segments) == 2
        for earlier, later in zip(segments, segments[1:]):
            assert earlier["end"] == later["start"]


# %% robot parts
@dataclass
class RobotPartWithBodies:
    """
    A robot part exposing the bodies whose link names the onboarder records.
    """

    bodies: List[Any] = field(default_factory=list)


@dataclass
class NamedBody:
    """
    A world body carrying a model-prefixed name.
    """

    name: str


class TestLinkSet:
    def test_the_model_prefix_is_stripped(self):
        part = RobotPartWithBodies(bodies=[NamedBody("pr2/l_wrist_link")])
        assert link_set(part) == ["l_wrist_link"]

    def test_an_unprefixed_name_is_kept(self):
        part = RobotPartWithBodies(bodies=[NamedBody("l_wrist_link")])
        assert link_set(part) == ["l_wrist_link"]

    def test_a_part_without_bodies_has_no_links(self):
        assert link_set(RobotPartWithBodies()) == []


# %% URDF reference resolution
class TestResolveUri:
    def test_a_recorded_resolution_wins(self, tmp_path):
        target = tmp_path / "cup.stl"
        target.write_text("solid cup\nendsolid cup\n")
        resolved = bundler.resolve_uri(
            "package://demo/cup.stl", hints={"package://demo/cup.stl": str(target)}
        )
        assert resolved == str(target)

    def test_a_relative_reference_resolves_against_the_urdf(self, tmp_path):
        mesh = tmp_path / "meshes" / "cup.stl"
        mesh.parent.mkdir()
        mesh.write_text("solid cup\nendsolid cup\n")
        assert bundler.resolve_uri("meshes/cup.stl", base_dir=str(tmp_path)) == str(
            mesh
        )

    def test_a_missing_relative_reference_is_unresolved(self, tmp_path):
        assert bundler.resolve_uri("meshes/gone.stl", base_dir=str(tmp_path)) is None

    def test_a_file_uri_resolves_to_its_path(self, tmp_path):
        mesh = tmp_path / "cup.stl"
        mesh.write_text("solid cup\nendsolid cup\n")
        assert bundler.resolve_uri("file://" + str(mesh)) == str(mesh)

    def test_an_absolute_path_that_exists_resolves_to_itself(self, tmp_path):
        mesh = tmp_path / "cup.stl"
        mesh.write_text("solid cup\nendsolid cup\n")
        assert bundler.resolve_uri(str(mesh)) == str(mesh)

    def test_an_unresolvable_package_uri_is_unresolved(self, monkeypatch):
        """
        Without a recorded hint and with no ROS installation to ask,
        :class:`PackageUriResolver` fails to resolve the package, and the URI comes back
        unresolved rather than raising.
        """
        monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
        monkeypatch.delenv("ROS_PACKAGE_PATH", raising=False)
        monkeypatch.delenv("CMAKE_PREFIX_PATH", raising=False)
        assert bundler.resolve_uri("package://no_such_package/cup.stl") is None


class TestReferenceLayout:
    def test_a_package_reference_keeps_its_package_directory(self):
        assert bundler._bundled_relative_path("package://demo/meshes/cup.stl") == (
            "demo/meshes/cup.stl"
        )

    def test_a_local_reference_lands_in_one_flat_directory(self):
        assert bundler._bundled_relative_path("../far/away/cup.stl") == "_local/cup.stl"


# %% xacro expansion
class TestXacroToUrdfText:
    def test_a_xacro_file_expands_to_urdf_text(self, tmp_path):
        """
        A macro-free xacro file expands to URDF text carrying the same links and joints,
        via :class:`URDFParser.from_xacro` rather than the ``xacro`` CLI.
        """
        source = (
            '<robot name="demo">\n'
            '  <link name="base_link"/>\n'
            '  <link name="cup_link"/>\n'
            '  <joint name="cup_joint" type="fixed">\n'
            '    <parent link="base_link"/><child link="cup_link"/>\n'
            "  </joint>\n"
            "</robot>\n"
        )
        xacro_path = tmp_path / "robot.xacro"
        xacro_path.write_text(source)

        expanded = bundler.xacro_to_urdf_text(str(xacro_path))

        assert bundler.LINK_PATTERN.findall(expanded) == bundler.LINK_PATTERN.findall(
            source
        )
        assert bundler.JOINT_PATTERN.findall(expanded) == bundler.JOINT_PATTERN.findall(
            source
        )


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
        out_dir = tmp_path / "bundle"
        report = bundler.bundle_urdf(str(source_tree), "demo", str(out_dir))
        assert (out_dir / "demo.urdf").is_file()
        assert (out_dir / "meshes" / "_local" / "cup.stl").is_file()
        assert report.meshes_copied == 1
        assert report.missing == []

    def test_the_reference_is_rewritten_to_the_bundled_copy(
        self, source_tree, tmp_path
    ):
        out_dir = tmp_path / "bundle"
        bundler.bundle_urdf(str(source_tree), "demo", str(out_dir))
        rewritten = (out_dir / "demo.urdf").read_text()
        assert 'filename="meshes/_local/cup.stl"' in rewritten
        assert 'filename="meshes/cup.stl"' not in rewritten

    def test_links_and_joints_are_reported(self, source_tree, tmp_path):
        report = bundler.bundle_urdf(str(source_tree), "demo", str(tmp_path / "bundle"))
        assert report.links == ["base_link", "cup_link"]
        assert report.joints == ["cup_joint"]
        assert report.movable_joints == []

    def test_a_xacro_source_is_bundled_like_a_urdf_source(
        self, xacro_source_tree, tmp_path
    ):
        """
        Bundling a xacro source produces the same links, joints and mesh copy as
        bundling the equivalent URDF - :func:`xacro_to_urdf_text`'s ElementTree
        round-trip does not break the regex-based mesh rewriting.
        """
        report = bundler.bundle_urdf(
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
        report = bundler.bundle_urdf(str(urdf), "demo", str(tmp_path / "bundle"))
        assert report.missing == [bundler.UNRESOLVED_REFERENCE]
        assert report.meshes_copied == 0

    def test_a_missing_source_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bundler.bundle_urdf(str(tmp_path / "gone.urdf"), "demo", str(tmp_path))
