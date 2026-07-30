from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    SetDofAllowExternalStateUpdate,
    SetDofHasHardwareInterface,
)

# %% builders


def build_world_with_prismatic_connection() -> tuple[World, PrismaticConnection]:
    """
    Build a world containing a single prismatic connection with one DOF.
    """
    world = World()
    parent = Body(name=PrefixedName("parent"))
    child = Body(name=PrefixedName("child"))
    with world.modify_world():
        world.add_body(parent)
    with world.modify_world():
        connection = PrismaticConnection.create_with_dofs(
            world=world, parent=parent, child=child, axis=Vector3.Z()
        )
        world.add_connection(connection)
    return world, connection


# %% boolean DOF flag modifications


def test_set_dof_has_hardware_interface_applies_flag():
    world, connection = build_world_with_prismatic_connection()
    assert connection.raw_dof.has_hardware_interface is False

    modification = SetDofHasHardwareInterface(
        degree_of_freedom_ids=[connection.raw_dof.id], value=True
    )
    modification.apply(world)

    assert connection.raw_dof.has_hardware_interface is True
    assert connection.raw_dof.allows_external_state_update is False


def test_set_dof_allow_external_state_update_applies_flag():
    world, connection = build_world_with_prismatic_connection()
    assert connection.raw_dof.allows_external_state_update is False

    modification = SetDofAllowExternalStateUpdate(
        degree_of_freedom_ids=[connection.raw_dof.id], value=True
    )
    modification.apply(world)

    assert connection.raw_dof.allows_external_state_update is True
    assert connection.raw_dof.has_hardware_interface is False


def test_set_dof_flag_json_round_trip():
    _, connection = build_world_with_prismatic_connection()
    modification = SetDofAllowExternalStateUpdate(
        degree_of_freedom_ids=[connection.raw_dof.id], value=True
    )
    restored = from_json(to_json(modification))
    assert restored == modification
