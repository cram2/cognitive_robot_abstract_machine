import numpy as np
import rerun
import trimesh
from PIL import Image

from semantic_digital_twin.adapters.rerun import (
    RerunAdapter,
    RerunMode,
    RerunModelCallback,
    body_entity_path,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.testing import world_setup_simple
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.world_entity import Body


def test_records_every_body(world_setup_simple, tmp_path) -> None:
    """
    The adapter records every body of the world to an ``.rrd``.

    ``RerunMode.SAVE`` writes a static snapshot (geometry plus current transforms) to
    disk natively; it is read back through Rerun's in- process server / DataFusion
    reader, asserting every body appears as a logged entity under its entity path.
    """
    world: World = world_setup_simple[0]
    recording_file_path = tmp_path / "world.rrd"

    adapter = RerunAdapter(
        _world=world, mode=RerunMode.SAVE, target=str(recording_file_path)
    )
    adapter.stop()

    recorded = RerunAdapter.read_recording_entities(str(recording_file_path))
    for body in world.bodies:
        entity = f"/{body_entity_path('world', body)}"
        assert any(
            path == entity or path.startswith(f"{entity}/") for path in recorded
        ), f"body '{body.name}' was not recorded"


def test_adapter_registers_handles_state_and_stops(world_setup_simple) -> None:
    """
    The adapter attaches callbacks, handles a state change, and detaches on stop.
    """
    world = world_setup_simple[0]
    state_callbacks_before = len(world.state.state_change_callbacks)

    adapter = RerunAdapter(_world=world, mode=RerunMode.NONE)
    assert len(world.state.state_change_callbacks) > state_callbacks_before

    world.notify_state_change()  # exercises the state callback path

    adapter.stop()
    assert len(world.state.state_change_callbacks) == state_callbacks_before


# %% entity paths


def test_body_entity_path_groups_by_prefix() -> None:
    """
    A body's entity path nests its local name under its prefix, keeping equally named
    bodies from different merged worlds distinct.
    """
    prefixed = Body(name=PrefixedName("base_link", "pr2"))
    unprefixed = Body(name=PrefixedName("bowl.stl"))
    assert body_entity_path("world", prefixed) == "world/pr2/base_link"
    assert body_entity_path("world", unprefixed) == "world/bowl.stl"


# %% mesh archetypes


def _textured_box_shape(tmp_path) -> Mesh:
    """
    A small box mesh with UV coordinates and an image texture, exported and re-read the
    way world geometry is.
    """
    box = trimesh.creation.box((0.1, 0.1, 0.1))
    uv = np.tile(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        (len(box.faces), 1),
    )
    texture_file_path = tmp_path / "texture.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(texture_file_path)
    return Mesh.from_trimesh(
        mesh=box,
        uv=uv,
        texture_file_path=str(texture_file_path),
        directory=tmp_path,
    )


def test_textured_mesh_archetype_carries_texture(tmp_path) -> None:
    """
    A textured mesh is logged with its albedo texture and V-flipped UV coordinates
    instead of being baked down to per-vertex colors.
    """
    shape = _textured_box_shape(tmp_path)

    archetype = RerunModelCallback.mesh_archetype(shape)

    assert archetype.albedo_texture_buffer is not None
    assert archetype.vertex_colors is None
    expected_texture_coordinates = np.asarray(shape.mesh.visual.uv, dtype=np.float32)
    expected_texture_coordinates = expected_texture_coordinates.copy()
    expected_texture_coordinates[:, 1] = 1.0 - expected_texture_coordinates[:, 1]
    recorded_texture_coordinates = np.asarray(
        archetype.vertex_texcoords.as_arrow_array().values
    ).reshape(-1, 2)
    assert np.allclose(recorded_texture_coordinates, expected_texture_coordinates)


def test_colorless_mesh_uses_shape_color_albedo(tmp_path) -> None:
    """
    A mesh without any color information of its own is tinted with the shape's color via
    the albedo factor instead of trimesh's default gray vertex colors.
    """
    box = trimesh.creation.box((0.1, 0.1, 0.1))
    shape = Mesh.from_trimesh(mesh=box, file_type="stl", directory=tmp_path)

    archetype = RerunModelCallback.mesh_archetype(shape)

    assert archetype.vertex_colors is None
    red, green, blue, alpha = (
        round(channel * 255) for channel in shape.color.to_rgba()
    )
    expected_packed_rgba = (red << 24) | (green << 16) | (blue << 8) | alpha
    assert archetype.albedo_factor.as_arrow_array().to_pylist() == [
        expected_packed_rgba
    ]


# %% state logging stride


def test_state_log_stride_skips_intermediate_versions(world_setup_simple) -> None:
    """
    With a stride, only every N-th state version is logged, and ``log_current_state``
    forces a log regardless of the stride.
    """
    world: World = world_setup_simple[0]
    adapter = RerunAdapter(
        _world=world, mode=RerunMode.NONE, state_history=True, state_log_stride=3
    )

    for _ in range(6):
        world.notify_state_change()
        if world.state.version % 3 == 0:
            assert adapter.last_logged_version == world.state.version
        else:
            assert adapter.last_logged_version != world.state.version

    if world.state.version % 3 == 0:
        world.notify_state_change()
    assert adapter.last_logged_version != world.state.version
    adapter.log_current_state()
    assert adapter.last_logged_version == world.state.version

    adapter.stop()


# %% default blueprint


def test_default_blueprint_builds(world_setup_simple) -> None:
    """
    The default viewer layout is constructible without a running viewer.
    """
    world: World = world_setup_simple[0]
    adapter = RerunAdapter(_world=world, mode=RerunMode.NONE)

    blueprint = adapter.default_blueprint()

    assert isinstance(blueprint, rerun.blueprint.Blueprint)
    adapter.stop()


# %% forward kinematics


def test_batched_body_fks_match_per_body(world_setup_simple) -> None:
    """
    Each slice of the batched body forward kinematics matches the per-body computation.
    """
    world: World = world_setup_simple[0]
    adapter = RerunAdapter(_world=world, mode=RerunMode.NONE)

    batched_body_fks = adapter.model_cb.compute()
    for index, body in enumerate(world.bodies):
        world_transform_body = batched_body_fks[index * 4 : index * 4 + 4]
        assert np.allclose(
            world_transform_body,
            world.compute_forward_kinematics_np(world.root, body),
        )

    adapter.stop()
