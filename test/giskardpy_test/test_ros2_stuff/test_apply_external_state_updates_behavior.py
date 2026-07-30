"""
Tests for the control-loop behavior that applies external state updates mid-motion.
"""

from dataclasses import dataclass

import pytest
from py_trees.common import Status

from giskardpy.tree.behaviors.apply_external_state_updates import (
    ApplyExternalStateUpdates,
)
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from giskardpy.tree.branches.synchronization import Synchronization
from semantic_digital_twin.adapters.ros.messages import (
    MetaData,
    ModificationBlock,
    WorldStateUpdate,
    WorldUpdate,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    WorldModelModificationBlock,
)

# %% test scaffolding


@dataclass
class GiskardWithWorldSynchronizer:
    """
    Stand-in for the Giskard blackboard entry exposing only the world synchronizer.
    """

    world_synchronizer: WorldSynchronizer
    """The synchronizer the behavior drains external state updates from."""


@dataclass
class ExternallyUpdatableWorld:
    """
    A world with one externally-updatable DOF and one controller-owned DOF.
    """

    world: World
    """The world owning both degrees of freedom."""

    externally_updatable_dof: DegreeOfFreedom
    """
    The DOF flagged to accept external state updates mid-motion.
    """

    controller_owned_dof: DegreeOfFreedom
    """A DOF without the flag, whose state the behavior must never touch."""


def _world_with_externally_updatable_dof() -> ExternallyUpdatableWorld:
    """
    Build a world with one flagged translating DOF and one unflagged one.
    """
    world = World()
    with world.modify_world():
        parent = Body(name=PrefixedName("parent"))
        child = Body(name=PrefixedName("child"))
        second_child = Body(name=PrefixedName("second_child"))
        world.add_body(parent)
        world.add_body(child)
        world.add_body(second_child)
        flagged_connection = PrismaticConnection.create_with_dofs(
            world=world, parent=parent, child=child, axis=Vector3.X()
        )
        world.add_connection(flagged_connection)
        controller_owned_connection = PrismaticConnection.create_with_dofs(
            world=world, parent=parent, child=second_child, axis=Vector3.X()
        )
        world.add_connection(controller_owned_connection)
    externally_updatable_dof = world.get_degree_of_freedom_by_id(
        flagged_connection.dof.id
    )
    controller_owned_dof = world.get_degree_of_freedom_by_id(
        controller_owned_connection.dof.id
    )
    with world.modify_world():
        world.set_dofs_allow_external_state_update([externally_updatable_dof], True)
    return ExternallyUpdatableWorld(
        world=world,
        externally_updatable_dof=externally_updatable_dof,
        controller_owned_dof=controller_owned_dof,
    )


def _perception_meta_data() -> MetaData:
    """
    Meta data of the simulated external perception source.
    """
    return MetaData(node_name="perception", process_id=1)


def _state_update(dof_ids: list, states: list[float]) -> WorldStateUpdate:
    """
    Build a state update message for the given DOF ids.
    """
    return WorldStateUpdate(
        meta_data=_perception_meta_data(), ids=dof_ids, states=states
    )


@pytest.fixture
def externally_updatable_world() -> ExternallyUpdatableWorld:
    """
    World with a flagged and an unflagged DOF.
    """
    return _world_with_externally_updatable_dof()


@pytest.fixture
def paused_synchronizer(
    rclpy_node, externally_updatable_world: ExternallyUpdatableWorld
) -> WorldSynchronizer:
    """
    A paused world synchronizer that only buffers incoming updates, closed on teardown.
    """
    synchronizer = WorldSynchronizer(
        node=rclpy_node, _world=externally_updatable_world.world
    )
    synchronizer.pause()
    yield synchronizer
    synchronizer.close()


@pytest.fixture
def blackboard_with_synchronizer(
    monkeypatch: pytest.MonkeyPatch, paused_synchronizer: WorldSynchronizer
) -> WorldSynchronizer:
    """
    Expose the synchronizer on the shared blackboard state for the behavior under test.
    """
    monkeypatch.setitem(
        GiskardBlackboard().__dict__,
        "giskard",
        GiskardWithWorldSynchronizer(world_synchronizer=paused_synchronizer),
    )
    return paused_synchronizer


# %% behavior


def test_behavior_applies_buffered_external_update_on_tick(
    externally_updatable_world: ExternallyUpdatableWorld,
    blackboard_with_synchronizer: WorldSynchronizer,
) -> None:
    """
    Ticking the behavior drains a buffered external update for a flagged DOF into the
    world.

    This is the mid-motion path: the server synchronizer stays paused (so it only buffers), and the
    control loop drives the application by ticking this behavior each cycle.
    """
    world = externally_updatable_world.world
    flagged_dof = externally_updatable_world.externally_updatable_dof
    blackboard_with_synchronizer.missed_messages.append(
        WorldUpdate(
            meta_data=_perception_meta_data(),
            state_update=_state_update([flagged_dof.id], [0.42]),
        )
    )

    status = ApplyExternalStateUpdates().update()

    assert status == Status.SUCCESS
    assert world.state[flagged_dof.id].position == 0.42
    assert blackboard_with_synchronizer.missed_messages == []


def test_behavior_defers_non_flagged_state_and_model_updates(
    externally_updatable_world: ExternallyUpdatableWorld,
    blackboard_with_synchronizer: WorldSynchronizer,
) -> None:
    """
    Non-flagged DOF state and model modifications must stay buffered, not be applied.

    Applying either mid-motion would overwrite a controller-owned DOF or invalidate the
    compiled controller, so the behavior may only drain the flagged entries.
    """
    world = externally_updatable_world.world
    flagged_dof = externally_updatable_world.externally_updatable_dof
    controller_owned_dof = externally_updatable_world.controller_owned_dof
    buffered_message = WorldUpdate(
        meta_data=_perception_meta_data(),
        modification_block=ModificationBlock(
            meta_data=_perception_meta_data(),
            modifications=WorldModelModificationBlock(),
        ),
        state_update=_state_update(
            [flagged_dof.id, controller_owned_dof.id], [0.42, 0.7]
        ),
    )
    blackboard_with_synchronizer.missed_messages.append(buffered_message)

    status = ApplyExternalStateUpdates().update()

    assert status == Status.SUCCESS
    assert world.state[flagged_dof.id].position == 0.42
    assert world.state[controller_owned_dof.id].position == 0.0
    assert blackboard_with_synchronizer.missed_messages == [buffered_message]
    assert buffered_message.modification_block is not None
    assert buffered_message.state_update.ids == [controller_owned_dof.id]
    assert buffered_message.state_update.states == [0.7]


# %% wiring


def test_synchronization_branch_applies_external_updates_first() -> None:
    """
    Every synchronization branch starts with the external-state-update behavior.

    The control loop's projection and closed-loop branches are both built from
    :class:`Synchronization`, so the behavior runs before the controller reads the world
    in either mode.
    """
    synchronization = Synchronization()

    assert isinstance(synchronization.children[0], ApplyExternalStateUpdates)
