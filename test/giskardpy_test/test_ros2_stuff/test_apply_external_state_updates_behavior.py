"""Wiring test for the control-loop behavior that applies external state updates mid-motion."""

from types import SimpleNamespace

from py_trees.common import Status

from giskardpy.tree.behaviors.apply_external_state_updates import (
    ApplyExternalStateUpdates,
)
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from semantic_digital_twin.adapters.ros.messages import (
    MetaData,
    WorldStateUpdate,
    WorldUpdate,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.world_entity import Body


def _world_with_externally_updatable_dof() -> tuple[World, object]:
    """Build a world with a single translating DOF flagged as externally updatable."""
    world = World()
    with world.modify_world():
        parent = Body(name=PrefixedName("parent"))
        child = Body(name=PrefixedName("child"))
        world.add_body(parent)
        world.add_body(child)
        connection = PrismaticConnection.create_with_dofs(
            world=world, parent=parent, child=child, axis=Vector3.X()
        )
        world.add_connection(connection)
    externally_updatable_dof = world.get_degree_of_freedom_by_id(connection.dof.id)
    with world.modify_world():
        world.set_dofs_allow_external_state_update([externally_updatable_dof], True)
    return world, externally_updatable_dof


def test_behavior_applies_buffered_external_update_on_tick(rclpy_node):
    """Ticking the behavior drains a buffered external update for a flagged DOF into the world.

    This is the mid-motion path: the server synchronizer stays paused (so it only buffers), and the
    control loop drives the application by ticking this behavior each cycle.
    """
    world, externally_updatable_dof = _world_with_externally_updatable_dof()
    synchronizer = WorldSynchronizer(node=rclpy_node, _world=world)
    synchronizer.pause()

    synchronizer.missed_messages.append(
        WorldUpdate(
            meta_data=MetaData(node_name="perception", process_id=1),
            state_update=WorldStateUpdate(
                meta_data=MetaData(node_name="perception", process_id=1),
                ids=[externally_updatable_dof.id],
                states=[0.42],
            ),
        )
    )

    blackboard = GiskardBlackboard()
    had_giskard = "giskard" in blackboard.__dict__
    previous_giskard = blackboard.__dict__.get("giskard")
    blackboard.giskard = SimpleNamespace(world_synchronizer=synchronizer)
    try:
        status = ApplyExternalStateUpdates().update()
    finally:
        if had_giskard:
            blackboard.giskard = previous_giskard
        else:
            del blackboard.__dict__["giskard"]

    assert status == Status.SUCCESS
    assert world.state[externally_updatable_dof.id].position == 0.42

    synchronizer.close()
