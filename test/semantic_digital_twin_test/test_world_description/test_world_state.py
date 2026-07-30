from uuid import uuid4

import numpy as np

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_state import WorldStateTrajectory

# %% builders


def build_trajectory_with_known_positions() -> (
    tuple[WorldStateTrajectory, PrismaticConnection]
):
    """
    Build a trajectory recording positions 0.1, 0.2, 0.3 for a single prismatic DOF.
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

    world.state[connection.raw_dof.id].position = 0.1
    trajectory = WorldStateTrajectory.from_world_state(world.state, time=0.0)
    world.state[connection.raw_dof.id].position = 0.2
    trajectory.append(world.state, time=0.1)
    world.state[connection.raw_dof.id].position = 0.3
    trajectory.append(world.state, time=0.2)
    return trajectory, connection


# %% get_dof_positions


def test_get_dof_positions_returns_recorded_positions():
    trajectory, connection = build_trajectory_with_known_positions()
    np.testing.assert_array_equal(
        trajectory.get_dof_positions(connection.raw_dof.id),
        np.array([0.1, 0.2, 0.3]),
    )


def test_get_dof_positions_for_unknown_dof_is_none():
    trajectory, _ = build_trajectory_with_known_positions()
    assert trajectory.get_dof_positions(uuid4()) is None
