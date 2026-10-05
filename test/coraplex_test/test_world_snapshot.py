from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

from .world_snapshot import WorldSnapshot


def test_restore_removes_a_body_added_after_capture(simple_pr2_world_setup):
    world, _, _ = simple_pr2_world_setup
    bodies_before = list(world.bodies)
    snapshot = WorldSnapshot.capture(world)
    body = Body(name=PrefixedName("snapshot_test_body"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_connection(FixedConnection(parent=world.root, child=body))

    snapshot.restore()

    assert world.bodies == bodies_before


def test_restore_resets_degree_of_freedom_positions(simple_pr2_world_setup):
    world, _, _ = simple_pr2_world_setup
    positions_before = world.state.to_uuid_position_dict()
    snapshot = WorldSnapshot.capture(world)
    degree_of_freedom_id = world.state.keys()[0]
    world.state[degree_of_freedom_id].position += 0.1
    world.notify_state_change()

    snapshot.restore()

    assert world.state.to_uuid_position_dict() == positions_before
