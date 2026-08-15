"""
Tests of writing a finalized live recording to disk as a replayable scene bundle.
"""

from __future__ import annotations

import json

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from cramera import paths
from cramera.live.bridge import Bridge
from cramera.live.recording import RecordedFrame, Recording
from cramera.live.recording_bundle import (
    NothingToBundle,
    finalize_recording,
    write_recording_bundle,
)
from cramera.live.recording_segments import derive_segments

from .test_live_bundle import attached_bridge, laboratory_world, shaped

MILK_SPAWN = [0.1, 0.2, 0.3, 0, 0, 0, 1]


def frame_with_milk(objects=None) -> RecordedFrame:
    return RecordedFrame(
        frames={},
        base=None,
        objects=objects if objects is not None else {"milk.stl": MILK_SPAWN},
    )


class TestGeometry:
    def test_the_robot_and_environment_are_bundled(self, tmp_path):
        bridge = attached_bridge(with_robot=True)
        bridge.snapshot()

        scene = write_recording_bundle(
            bridge, [frame_with_milk()], 20.0, tmp_path / "rec", "__recording__"
        )

        names = sorted(model["name"] for model in scene["models"])
        assert names == ["environment", "robotwithsubtree"]
        assert (tmp_path / "rec" / "environment.urdf").is_file()
        assert (tmp_path / "rec" / "robotwithsubtree.urdf").is_file()

    def test_the_scene_carries_the_given_name(self, tmp_path):
        bridge = attached_bridge()

        scene = write_recording_bundle(
            bridge, [frame_with_milk()], 20.0, tmp_path / "rec", "my_run"
        )

        assert scene["name"] == "my_run"

    def test_the_bundle_signature_matches_the_bridge(self, tmp_path):
        bridge = attached_bridge()

        scene = write_recording_bundle(
            bridge, [frame_with_milk()], 20.0, tmp_path / "rec", "__recording__"
        )

        assert scene["bundleSignature"] == bridge.bundle_signature()

    def test_an_existing_directory_is_replaced(self, tmp_path):
        bridge = attached_bridge()
        output_directory = tmp_path / "rec"
        write_recording_bundle(bridge, [frame_with_milk()], 20.0, output_directory, "a")
        marker = output_directory / "marker"
        marker.touch()

        write_recording_bundle(bridge, [frame_with_milk()], 20.0, output_directory, "b")

        assert not marker.exists()


class TestNoFrames:
    def test_an_empty_recording_cannot_be_bundled(self, tmp_path):
        bridge = attached_bridge()

        with pytest.raises(NothingToBundle):
            write_recording_bundle(bridge, [], 20.0, tmp_path / "rec", "__recording__")


class TestLooseObjects:
    def test_a_box_shaped_object_is_an_inline_box(self, tmp_path):
        """
        ``milk.stl`` in the shared laboratory fixture is a Box-shaped body — it must be
        described inline, exactly like the offline onboarding pipeline describes a
        single-box body, without writing a mesh file.
        """
        bridge = attached_bridge()

        scene = write_recording_bundle(
            bridge, [frame_with_milk()], 20.0, tmp_path / "rec", "__recording__"
        )

        [entry] = scene["objects"]
        assert entry["key"] == "milk.stl"
        assert entry["box"] == [0.1, 0.1, 0.1]
        assert entry["spawn"] == MILK_SPAWN
        assert "mesh" not in entry
        assert not (tmp_path / "rec" / "meshes" / "objects").exists()

    def test_a_shapeless_object_falls_back_to_the_catalogs_placeholder_box(
        self, tmp_path
    ):
        """
        Not every published body carries real geometry (e.g. one only ever named, never
        given a shape) — the bridge already renders it as a placeholder box live, and
        the recording must reuse that same size rather than trying to measure or export
        geometry that does not exist.
        """
        world = laboratory_world()
        blob = Body(name=PrefixedName("blob.stl", prefix="world"))
        with world.modify_world():
            world.add_connection(FixedConnection(parent=world.root, child=blob))

        bridge = Bridge()
        bridge.attach(world)
        bridge.snapshot()

        scene = write_recording_bundle(
            bridge,
            [
                frame_with_milk(
                    objects={"milk.stl": MILK_SPAWN, "blob.stl": [1, 2, 3, 0, 0, 0, 1]}
                )
            ],
            20.0,
            tmp_path / "rec",
            "__recording__",
        )

        entry = next(e for e in scene["objects"] if e["key"] == "blob.stl")
        assert entry["box"] == list(Bridge.DEFAULT_OBJECT_SIZE)
        assert "mesh" not in entry

    def test_a_mesh_shaped_object_is_copied_into_the_bundle(self, tmp_path):
        source = tmp_path / "table.obj"
        source.write_text("o table\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        world = laboratory_world()
        table = Body(
            name=PrefixedName("table.obj", prefix="world"),
            visual=ShapeCollection(
                shapes=[Mesh(filename=str(source), scale=Scale(1.0, 1.0, 1.0))]
            ),
        )
        with world.modify_world():
            world.add_connection(FixedConnection(parent=world.root, child=table))

        bridge = Bridge()
        bridge.attach(world)
        bridge.snapshot()

        scene = write_recording_bundle(
            bridge,
            [
                frame_with_milk(
                    objects={"milk.stl": MILK_SPAWN, "table.obj": [1, 2, 3, 0, 0, 0, 1]}
                )
            ],
            20.0,
            tmp_path / "rec",
            "__recording__",
        )

        entry = next(e for e in scene["objects"] if e["key"] == "table.obj")
        assert entry["mesh"] == "meshes/objects/table.obj.obj"
        assert (tmp_path / "rec" / "meshes" / "objects" / "table.obj.obj").is_file()

    def test_an_object_missing_from_the_first_frame_is_skipped(self, tmp_path):
        """
        An object spawned after recording started has no first-frame pose to declare as
        its static spawn — v1 scope cut, see cramera.live.recording_bundle.
        """
        bridge = attached_bridge()

        scene = write_recording_bundle(
            bridge,
            [frame_with_milk(objects={})],
            20.0,
            tmp_path / "rec",
            "__recording__",
        )

        assert scene["objects"] == []


class TestTrajectory:
    def test_frame_arrays_match_the_recorded_ticks(self, tmp_path):
        bridge = attached_bridge()
        frames = [
            frame_with_milk(),
            RecordedFrame(
                frames={"j": 1.0}, base=[0] * 7, objects={"milk.stl": MILK_SPAWN}
            ),
        ]

        write_recording_bundle(bridge, frames, 20.0, tmp_path / "rec", "__recording__")

        trajectory = json.loads((tmp_path / "rec" / "trajectory.json").read_text())
        assert len(trajectory["frames"]) == 2
        assert len(trajectory["base"]) == 2
        assert len(trajectory["objects"]) == 2
        assert trajectory["frames"][1] == {"j": 1.0}
        assert trajectory["base"][1] == [0] * 7

    def test_frames_per_second_is_carried_through(self, tmp_path):
        bridge = attached_bridge()

        write_recording_bundle(
            bridge, [frame_with_milk()], 12.5, tmp_path / "rec", "__recording__"
        )

        trajectory = json.loads((tmp_path / "rec" / "trajectory.json").read_text())
        assert trajectory["framesPerSecond"] == 12.5


class TestSegments:
    """
    A recording's replay timeline is marked from the bundle's segments; a bundle that
    wrote none would replay as one unmarked stretch (see
    :mod:`cramera.live.recording_segments`).
    """

    def carried_milk(self):
        """
        Four ticks over which the milk is carried away from where it spawned.
        """
        return [
            frame_with_milk({"milk.stl": MILK_SPAWN}),
            frame_with_milk({"milk.stl": MILK_SPAWN}),
            frame_with_milk({"milk.stl": [1.1, 0.2, 0.3, 0, 0, 0, 1]}),
            frame_with_milk({"milk.stl": [2.1, 0.2, 0.3, 0, 0, 0, 1]}),
        ]

    def test_a_carried_object_is_written_as_a_segment_with_its_frames(self, tmp_path):
        bridge = attached_bridge()

        scene = write_recording_bundle(
            bridge, self.carried_milk(), 20.0, tmp_path / "rec", "__recording__"
        )

        carried = [segment for segment in scene["segments"] if segment.get("picks")]
        assert len(carried) == 1
        assert carried[0]["picks"] == "milk.stl"
        assert carried[0]["attach"] == 2

    def test_the_written_segments_are_what_the_derivation_yields(self, tmp_path):
        bridge = attached_bridge()
        frames = self.carried_milk()

        scene = write_recording_bundle(
            bridge, frames, 20.0, tmp_path / "rec", "__recording__"
        )

        assert scene["segments"] == [
            segment.to_payload() for segment in derive_segments(frames)
        ]

    def test_a_recording_in_which_nothing_moved_is_still_one_segment(self, tmp_path):
        bridge = attached_bridge()

        scene = write_recording_bundle(
            bridge, [frame_with_milk()] * 3, 20.0, tmp_path / "rec", "__recording__"
        )

        assert [segment["start"] for segment in scene["segments"]] == [0]


class TestFinalizeRecording:
    """
    The safety net a demo process's exit relies on (see cramera.live.visualization) —
    the recording must already be written to disk by the time nothing is left to ask it
    to stop.
    """

    def test_an_idle_recording_has_nothing_to_finalize(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        bridge = attached_bridge()

        assert finalize_recording(bridge, Recording()) is None
        assert not (tmp_path / "scenes" / paths.RECORDING_SCENE_NAME).exists()

    def test_a_recording_with_frames_is_bundled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        bridge = attached_bridge()
        recording = Recording()
        recording.start()
        recording.append(bridge.state)

        scene_name = finalize_recording(bridge, recording)

        assert scene_name == paths.RECORDING_SCENE_NAME
        assert recording.scene_name == paths.RECORDING_SCENE_NAME
        assert (
            tmp_path / "scenes" / paths.RECORDING_SCENE_NAME / "scene.json"
        ).is_file()

    def test_a_recording_with_no_frames_is_left_unbundled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        bridge = attached_bridge()
        recording = Recording()
        recording.start()

        assert finalize_recording(bridge, recording) is None
        assert not (tmp_path / "scenes" / paths.RECORDING_SCENE_NAME).exists()

    def test_an_already_bundled_recording_is_not_rewritten(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        bridge = attached_bridge()
        recording = Recording()
        recording.start()
        recording.append(bridge.state)
        finalize_recording(bridge, recording)
        marker = tmp_path / "scenes" / paths.RECORDING_SCENE_NAME / "marker"
        marker.touch()

        scene_name = finalize_recording(bridge, recording)

        assert scene_name == paths.RECORDING_SCENE_NAME
        assert marker.exists()
