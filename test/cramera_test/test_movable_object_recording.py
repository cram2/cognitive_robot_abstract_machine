"""
Live and recorded objects retain their identity through native attachments.
"""

from __future__ import annotations

import json

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

from cramera.live.bridge import Bridge
from cramera.live.live_bundle import build_live_scene
from cramera.live.recording import Recording
from cramera.live.recording_bundle import write_recording_bundle
from cramera import paths

from .test_live_bundle import shaped


# %% native movable objects


@pytest.fixture()
def movable_crate():
    """
    A free primitive and a fixed support in a native world.
    """
    world = World()
    root = Body(name=PrefixedName("world"))
    crate = shaped("scene", "crate")
    support = shaped("scene", "support")
    with world.modify_world():
        world.add_connection(FixedConnection(parent=root, child=support))
        world.add_connection(
            Connection6DoF.create_with_dofs(parent=root, child=crate, world=world)
        )
    bridge = Bridge()
    bridge.attach(world)
    bridge.snapshot()
    return bridge, crate, support


def test_a_free_primitive_streams_its_native_pose(movable_crate):
    """
    A movable box is tracked without a mesh filename in its name.
    """
    bridge, crate, _ = movable_crate

    assert bridge.object_body(crate.name.name) is crate
    assert bridge.get_state()["objects"][crate.name.name] == [0, 0, 0, 0, 0, 0, 1]


def test_attachment_preserves_the_object_and_static_bundle(
    movable_crate, tmp_path, monkeypatch
):
    """
    An attached object stays in its overlay instead of being duplicated in URDF.
    """
    bridge, crate, support = movable_crate
    monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
    signature = bridge.bundle_signature()
    with bridge.world.modify_world():
        bridge.world.remove_connection(crate.parent_connection)
        bridge.world.add_connection(
            FixedConnection(
                parent=support,
                child=crate,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    1, 0, 0
                ),
            )
        )
    bridge.observe_model_change()
    bridge.snapshot()

    assert bridge.object_body(crate.name.name) is crate
    assert bridge.bundle_signature() == signature
    assert bridge.get_state()["objects"][crate.name.name][:3] == [1, 0, 0]
    scene_name = build_live_scene(bridge)
    environment = paths.local_scenes_directory() / scene_name / "environment.urdf"
    assert str(crate.name) not in environment.read_text()


def test_a_moving_primitive_is_kept_in_a_replay(movable_crate, tmp_path):
    """
    The recording contains both the primitive geometry and its motion track.
    """
    bridge, crate, _ = movable_crate
    recording = Recording()
    recording.start()
    recording.append(bridge.state)
    crate.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, 2, 0, reference_frame=bridge.world.root
    )
    bridge.snapshot()
    recording.append(bridge.state)
    output = tmp_path / "recording"

    scene = write_recording_bundle(bridge, recording.stop(), 30, output, "crate_motion")

    assert [entry["key"] for entry in scene["objects"]] == [crate.name.name]
    trajectory = json.loads((output / "trajectory.json").read_text())
    assert trajectory["objects"][1][crate.name.name][:3] == [1, 2, 0]


def test_fixed_children_follow_a_moving_primitive(movable_crate, tmp_path, monkeypatch):
    """
    A crate handle follows the crate in live state and saved geometry.
    """
    bridge, crate, _ = movable_crate
    handle = shaped("scene", "handle")
    with bridge.world.modify_world():
        bridge.world.add_connection(
            FixedConnection(
                parent=crate,
                child=handle,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0, 0, 1
                ),
            )
        )
    bridge.observe_model_change()
    crate.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, 2, 0, reference_frame=bridge.world.root
    )
    bridge.snapshot()
    monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))

    assert bridge.get_state()["objects"][handle.name.name][:3] == [1, 2, 1]
    scene_name = build_live_scene(bridge)
    bundle = paths.local_scenes_directory() / scene_name
    assert str(handle.name) not in (bundle / "environment.urdf").read_text()
    recording = Recording()
    recording.start()
    recording.append(bridge.state)
    scene = write_recording_bundle(
        bridge, recording.stop(), 30, tmp_path / "recording", "assembly"
    )
    assert {entry["key"] for entry in scene["objects"]} == {
        crate.name.name,
        handle.name.name,
    }
