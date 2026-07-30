import numpy as np

from semantic_digital_twin.adapters.rerun import RerunMode, RerunAdapter
from semantic_digital_twin.testing import world_setup_simple
from semantic_digital_twin.world import World


def test_records_every_body(world_setup_simple, tmp_path) -> None:
    """
    The adapter records every body of the world to an ``.rrd``.

    ``RerunMode.SAVE`` writes a static snapshot (geometry plus current transforms) to
    disk natively; it is read back through Rerun's in- process server / DataFusion
    reader, asserting every body appears as a logged entity under ``world/<body>``.
    """
    world: World = world_setup_simple[0]
    recording_file_path = tmp_path / "world.rrd"

    adapter = RerunAdapter(
        _world=world, mode=RerunMode.SAVE, target=str(recording_file_path)
    )
    adapter.stop()

    recorded = RerunAdapter.read_recording_entities(str(recording_file_path))
    for body in world.bodies:
        entity = f"/world/{body.name.name}"
        assert any(
            path == entity or path.startswith(f"{entity}/") for path in recorded
        ), f"body '{body.name.name}' was not recorded"


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
