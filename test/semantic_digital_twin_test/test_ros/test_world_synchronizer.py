import hashlib
import json
import os
import threading
import time
from datetime import timedelta
import unittest
import uuid
from dataclasses import dataclass, field
from multiprocessing.synchronize import RLock
from time import sleep
from typing import Callable, Optional, Set, Tuple, List
from uuid import uuid4

import numpy as np
import pytest
import rclpy
import std_msgs.msg
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sqlalchemy import select
from sqlalchemy.orm import Session

from krrood.adapters.json_serializer import JSONAttributeDiff, to_json, from_json
from semantic_digital_twin.adapters.ros.messages import (
    MetaData,
    WorldStateUpdate,
    LoadModel,
    StreamPosition,
    WorldUpdate,
)
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelReloadSynchronizer,
    Synchronizer,
    WorldSynchronizer,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    MissingWorldModificationContextError,
    MismatchingPublishChangesAttribute,
    ApplyMissedMessagesWhileWorldIsBeingModifiedError,
    StateUpdateContainsUnknownDegreesOfFreedomError,
    BrokenWorldModificationHistoryError,
    WorldHasMultipleSynchronizersError,
    WorldHasNoSynchronizerError,
)
from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Door,
    Fridge,
    Drawer,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)
from semantic_digital_twin.world_description.world_modification import (
    AttributeUpdateModification,
    synchronized_attribute_modification,
)
from krrood.adapters.json_serializer import JSONAttributeDiff, to_json, from_json
from semantic_digital_twin.callbacks.callback import (
    StateChangeCallback,
    ModelChangeCallback,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.adapters.ros.messages import (
    MetaData,
    WorldStateUpdate,
    ModificationBlock,
    LoadModel,
    WorldUpdate,
)
from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO


def create_dummy_world(w: Optional[World] = None) -> World:
    def deterministic_uuid(seed: str) -> uuid.UUID:
        h = hashlib.sha1(seed.encode()).hexdigest()[:32]
        return uuid.UUID(h)

    id1 = deterministic_uuid("id1")
    id2 = deterministic_uuid("id2")
    if w is None:
        w = World()
    b1 = Body(name=PrefixedName("b1"), id=id1)
    b2 = Body(name=PrefixedName("b2"), id=id2)
    with w.modify_world():
        x_dof = DegreeOfFreedom(name=PrefixedName("x"), id=deterministic_uuid("x_dof"))
        w.add_degree_of_freedom(x_dof)
        y_dof = DegreeOfFreedom(name=PrefixedName("y"), id=deterministic_uuid("y_dof"))
        w.add_degree_of_freedom(y_dof)
        z_dof = DegreeOfFreedom(name=PrefixedName("z"), id=deterministic_uuid("z_dof"))
        w.add_degree_of_freedom(z_dof)
        qx_dof = DegreeOfFreedom(
            name=PrefixedName("qx"), id=deterministic_uuid("qx_dof")
        )
        w.add_degree_of_freedom(qx_dof)
        qy_dof = DegreeOfFreedom(
            name=PrefixedName("qy"), id=deterministic_uuid("qy_dof")
        )
        w.add_degree_of_freedom(qy_dof)
        qz_dof = DegreeOfFreedom(
            name=PrefixedName("qz"), id=deterministic_uuid("qz_dof")
        )
        w.add_degree_of_freedom(qz_dof)
        qw_dof = DegreeOfFreedom(
            name=PrefixedName("qw"), id=deterministic_uuid("qw_dof")
        )
        w.add_degree_of_freedom(qw_dof)
        w.state[qw_dof.id].position = 1.0

        w.add_connection(
            Connection6DoF(
                parent=b1,
                child=b2,
                x=x_dof,
                y=y_dof,
                z=z_dof,
                qx=qx_dof,
                qy=qy_dof,
                qz=qz_dof,
                qw=qw_dof,
            )
        )
    return w


def wait_for_sync_kse_and_return_ids(
    w1: World, w2: World, timeout: float = 5.0, interval: float = 0.05
) -> Tuple[Set[uuid.UUID], Set[uuid.UUID]]:
    """
    Waits until the sets of kinematic structure entity IDs in both worlds are identical,
    or until the timeout is reached.

    :param w1: The first world.
    :param w2: The second world.
    :param timeout: The maximum time to wait for synchronization, in seconds. Defaults
        to 5.0.
    :param interval: The time interval between checks, in seconds. Defaults to 0.05.
    :return: A tuple containing the sets of kinematic structure entity IDs in both
        worlds.
    """
    start = time.time()
    while time.time() - start < timeout:
        body_ids_1 = {body.id for body in w1.kinematic_structure_entities}
        body_ids_2 = {body.id for body in w2.kinematic_structure_entities}
        if body_ids_1 == body_ids_2:
            return body_ids_1, body_ids_2
        time.sleep(interval)

    body_ids_1 = {body.id for body in w1.kinematic_structure_entities}
    body_ids_2 = {body.id for body in w2.kinematic_structure_entities}
    return body_ids_1, body_ids_2


def wait_for_condition(condition, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """
    Waits until the condition callable returns True, or until the timeout is reached.

    :param condition: A callable returning a truthy value once the awaited state is
        reached.
    :param timeout: The maximum time to wait, in seconds. Defaults to 5.0.
    :param interval: The time interval between checks, in seconds. Defaults to 0.05.
    :return: The final result of the condition.
    """
    start = time.time()
    while time.time() - start < timeout:
        if condition():
            return True
        time.sleep(interval)
    return bool(condition())


def probe_lock_is_free(lock: RLock, timeout: float = 0.3) -> bool:
    """
    Determine, from a *separate* thread, whether ``lock`` can currently be acquired.

    The probe runs in another thread on purpose: ``World._world_lock`` is a reentrant
    lock, so the thread that already owns it would always re-acquire it successfully.
    A foreign thread, by contrast, only succeeds when the lock is genuinely free.

    :param lock: The lock to probe (e.g. ``world._world_lock``).
    :param timeout: How long the probe thread waits to acquire the lock.
    :return: ``True`` if the lock was free (acquired by the probe), ``False`` if it was held.
    """
    acquired = {"value": False}

    def _try_acquire():
        if lock.acquire(timeout=timeout):
            acquired["value"] = True
            lock.release()

    probe_thread = threading.Thread(target=_try_acquire, daemon=True)
    probe_thread.start()
    probe_thread.join(timeout + 1.0)
    return acquired["value"]


def test_state_synchronization(rclpy_node):
    w1 = create_dummy_world()
    w2 = create_dummy_world()

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    # Allow time for publishers/subscribers to connect on unique topics
    time.sleep(0.2)

    w1.state._data[0, 0] = 1.0
    w1.notify_state_change()
    time.sleep(0.2)
    assert w1.state._data[0, 0] == 1.0
    assert w1.state._data[0, 0] == w2.state._data[0, 0]

    synchronizer_1.close()
    synchronizer_2.close()


def test_state_synchronization_world_model_change_after_init(rclpy_node):
    w1 = World()
    w2 = World()

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    create_dummy_world(w1)
    create_dummy_world(w2)
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    # Allow time for publishers/subscribers to connect on unique topics
    time.sleep(0.2)

    w1.state._data[0, 0] = 1.0
    w1.notify_state_change()
    time.sleep(0.2)
    assert w1.state._data[0, 0] == 1.0
    assert w1.state._data[0, 0] == w2.state._data[0, 0]

    synchronizer_1.close()
    synchronizer_2.close()


def test_model_reload(rclpy_node, in_memory_session_maker):
    session1 = in_memory_session_maker()
    session2 = in_memory_session_maker()

    w1 = create_dummy_world()
    w2 = World()

    synchronizer_1 = ModelReloadSynchronizer(
        node=rclpy_node,
        _world=w1,
        session=session1,
    )
    synchronizer_2 = ModelReloadSynchronizer(
        node=rclpy_node,
        _world=w2,
        session=session2,
    )

    synchronizer_1.publish_reload_model()
    time.sleep(1.0)
    assert len(w2.kinematic_structure_entities) == 2

    query = session1.scalars(select(WorldMappingDAO)).all()
    assert len(query) == 1
    assert w2.get_kinematic_structure_entity_by_name("b2")

    synchronizer_1.close()
    synchronizer_2.close()


def test_model_synchronization_body_only(rclpy_node):

    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    with w1.modify_world():
        new_body = Body(name=PrefixedName("b3"))
        b3_id = new_body.id
        w1.add_kinematic_structure_entity(new_body)

    time.sleep(0.2)
    assert len(w1.kinematic_structure_entities) == 1
    assert len(w2.kinematic_structure_entities) == 1

    assert w2.get_kinematic_structure_entity_by_id(b3_id)

    synchronizer_1.close()
    synchronizer_2.close()


def test_model_synchronization_creation_only(rclpy_node):

    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    with w1.modify_world():
        b2 = Body(name=PrefixedName("b2"))
        w1.add_kinematic_structure_entity(b2)

        new_body = Body(name=PrefixedName("b3"))
        w1.add_kinematic_structure_entity(new_body)

        c = Connection6DoF.create_with_dofs(parent=b2, child=new_body, world=w1)
        w1.add_connection(c)
    wait_for_sync_kse_and_return_ids(w1, w2)
    wait_for_condition(lambda: len(w2.connections) == 1)
    assert len(w1.kinematic_structure_entities) == 2
    assert len(w2.kinematic_structure_entities) == 2
    assert len(w1.connections) == 1
    assert len(w2.connections) == 1

    synchronizer_1.close()
    synchronizer_2.close()


def test_model_synchronization_merge_full_world_stress_test(rclpy_node):

    def wait_for_sync(timeout=5.0, interval=0.05):
        start = time.time()
        while time.time() - start < timeout:
            body_hash_1 = {hash(body) for body in w1.kinematic_structure_entities}
            body_hash_2 = {hash(body) for body in w2.kinematic_structure_entities}

            connection_hash_1 = {hash(conn) for conn in w1.connections}
            connection_hash_2 = {hash(conn) for conn in w2.connections}

            dof_hash_1 = {hash(dof) for dof in w1.degrees_of_freedom}
            dof_hash_2 = {hash(dof) for dof in w2.degrees_of_freedom}

            semantic_annotation_hash_1 = {hash(sa) for sa in w1.semantic_annotations}
            semantic_annotation_hash_2 = {hash(sa) for sa in w2.semantic_annotations}

            if (
                body_hash_1 == body_hash_2
                and connection_hash_1 == connection_hash_2
                and dof_hash_1 == dof_hash_2
                and semantic_annotation_hash_1 == semantic_annotation_hash_2
            ):
                return
            time.sleep(interval)

        raise RuntimeError(
            f"World synchronization timed out after {i+1} attempts. bodylen: {len(body_hash_1)} vs {len(body_hash_2)}"
        )

    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=w2,
    )
    for i in range(10):

        pr2_world = URDFParser.from_file(PR2.get_ros_file_path()).parse()

        w1.merge_world(pr2_world)
        sleep(1)

    wait_for_sync()

    assert {body.id for body in w1.kinematic_structure_entities} == {
        body.id for body in w2.kinematic_structure_entities
    }
    assert len(w1.kinematic_structure_entities) == len(w2.kinematic_structure_entities)

    w1_connection_hashes = [hash(c) for c in w1.connections]
    w2_connection_hashes = [hash(c) for c in w2.connections]
    assert (
        w1_connection_hashes == w2_connection_hashes
    ), f"w1: {[c.name for c in w1.connections]}, w2: {[c.name for c in w2.connections]}, If this feels flaky, contact @LucaKro"
    assert [d.id for d in w1.degrees_of_freedom] == [
        d.id for d in w2.degrees_of_freedom
    ], f"w1: {[d.name for d in w1.degrees_of_freedom]}, w2: {[d.name for d in w2.degrees_of_freedom]}, If this feels flaky, contact @LucaKro"

    synchronizer_1.close()
    synchronizer_2.close()


def test_callback_pausing(rclpy_node):

    w1 = World(name="w1")
    w2 = World(name="w2")

    ws1 = WorldSynchronizer(node=rclpy_node, _world=w1)
    ws2 = WorldSynchronizer(node=rclpy_node, _world=w2)

    ws2.pause()
    assert ws2._is_paused

    with w1.modify_world():
        b2 = Body(name=PrefixedName("b2"))
        w1.add_kinematic_structure_entity(b2)

        new_body = Body(name=PrefixedName("b3"))
        w1.add_kinematic_structure_entity(new_body)

        c = Connection6DoF.create_with_dofs(parent=b2, child=new_body, world=w1)
        w1.add_connection(c)

    time.sleep(0.2)
    assert len(ws2.missed_messages) == 2
    assert len(w1.kinematic_structure_entities) == 2
    assert len(w2.kinematic_structure_entities) == 0
    assert len(w1.connections) == 1
    assert len(w2.connections) == 0

    ws2.resume()
    ws2.apply_missed_messages()

    time.sleep(0.2)
    assert len(w1.kinematic_structure_entities) == 2
    assert len(w2.kinematic_structure_entities) == 2
    assert len(w1.connections) == 1
    assert len(w2.connections) == 1


def test_synchronizer_keeps_receiving_while_its_world_is_modified(rclpy_node):
    """
    A pause makes the synchronizer buffer inbound updates into ``missed_messages``
    instead of applying them, and nothing drains that buffer on its own.

    Modifying the
    world must therefore not pause it: outgoing publications are already deferred by
    :meth:`WorldSynchronizer._publish_or_defer`.
    """
    world = World(name="modified_world")
    synchronizer = WorldSynchronizer(node=rclpy_node, _world=world)

    with world.modify_world():
        assert not synchronizer._is_paused
        world.add_kinematic_structure_entity(Body(name=PrefixedName("b1")))

    assert not synchronizer._is_paused

    synchronizer.close()


def test_modify_world_preserves_a_deliberate_pause(rclpy_node):
    """
    A caller that paused a synchronizer on purpose keeps it paused across a
    ``modify_world`` block, so buffered updates are not silently applied behind its
    back.
    """
    world = World(name="paused_world")
    synchronizer = WorldSynchronizer(node=rclpy_node, _world=world)
    synchronizer.pause()

    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("b1")))

    assert synchronizer._is_paused

    synchronizer.close()


def test_ChangeDifHasHardwareInterface(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    try:

        with w1.modify_world():
            body1 = Body(name=PrefixedName("b1"))
            body2 = Body(name=PrefixedName("b2"))
            w1.add_kinematic_structure_entity(body1)
            w1.add_kinematic_structure_entity(body2)
            dof = DegreeOfFreedom(name=PrefixedName("dof"))
            w1.add_degree_of_freedom(dof)
            connection = PrismaticConnection(
                raw_dof=dof, parent=body1, child=body2, axis=Vector3(1, 1, 1)
            )
            w1.add_connection(connection)

        w1_ids, w2_ids = wait_for_sync_kse_and_return_ids(w1, w2)

        assert len(w1.kinematic_structure_entities) == 2
        assert len(w1.connections) == 1

        time.sleep(0.2)
        assert len(w1.kinematic_structure_entities) == 2
        assert len(w2.kinematic_structure_entities) == 2
        assert len(w2.connections) == 1
        assert not w2.connections[0].dof.has_hardware_interface
        assert not w2.connections[0].dof.has_hardware_interface

        assert w2.get_kinematic_structure_entity_by_name("b2")

        with w1.modify_world():
            w1.set_dofs_has_hardware_interface(w1.degrees_of_freedom, True)

        time.sleep(0.2)
        assert w1.connections[0].dof.has_hardware_interface
        assert w2.connections[0].dof.has_hardware_interface

    finally:
        synchronizer_1.close()
        synchronizer_2.close()


def test_semantic_annotation_modifications(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    b1 = Body(name=PrefixedName("b1"))
    v1 = Handle(root=b1)
    v2 = Door(root=b1, handle=v1)

    with w1.modify_world():
        w1.add_body(b1)
        w1.add_semantic_annotation(v1)
        w1.add_semantic_annotation(v2)

    time.sleep(0.5)
    assert [hash(sa) for sa in w1.semantic_annotations] == [
        hash(sa) for sa in w2.semantic_annotations
    ]


def test_semantic_annotation_modifications_merge_world(rclpy_node):
    w0 = World(name="w0")
    root = Body(name=PrefixedName("root"))
    with w0.modify_world():
        w0.add_body(root)

    with w0.modify_world():
        door = Door.create_with_new_body_in_world(
            name="door",
            world=w0,
        )
        handle = Handle.create_with_new_body_in_world(
            name="handle",
            world=w0,
        )
        door.add(handle)

    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    with w1.modify_world():
        w1.merge_world(w0)

    time.sleep(1)
    assert [hash(sa) for sa in w1.semantic_annotations] == [
        hash(sa) for sa in w2.semantic_annotations
    ]


def test_semantic_annotation_change_parameter_during_same_modification_block(
    rclpy_node,
):
    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=w2,
    )
    root = Body(name=PrefixedName("root"))
    b1 = Body(name=PrefixedName("b1"))
    drawer = Drawer(root=b1)

    b2 = Body(name=PrefixedName("b2"))
    handle = Handle(root=b2)

    with w1.modify_world():
        w1.add_body(root)
        w1.add_body(b1)
        w1.add_body(b2)
        root_C_b1 = Connection6DoF.create_with_dofs(parent=root, child=b1, world=w1)
        w1.add_connection(root_C_b1)
        root_C_b2 = Connection6DoF.create_with_dofs(parent=root, child=b2, world=w1)
        w1.add_connection(root_C_b2)
    with w1.modify_world():
        w1.add_semantic_annotation(drawer)
        w1.add_semantic_annotation(handle)
        drawer.add(handle)

    time.sleep(1)
    assert [hash(sa) for sa in w1.semantic_annotations] == [
        hash(sa) for sa in w2.semantic_annotations
    ], f"w1: {[sa.name for sa in w1.semantic_annotations]}, w2: {[sa.name for sa in w2.semantic_annotations]}"


def test_synchronize_6dof(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")

    ws1 = WorldSynchronizer(node=rclpy_node, _world=w1)
    ws2 = WorldSynchronizer(node=rclpy_node, _world=w2)

    b1 = Body(name=PrefixedName("b1"))
    b2 = Body(name=PrefixedName("b2"))

    with w1.modify_world():
        w1.add_body(b1)
        w1.add_body(b2)
        c1 = Connection6DoF.create_with_dofs(parent=b1, child=b2, world=w1)
        w1.add_connection(c1)

    time.sleep(1)
    c2 = w2.get_connection_by_name(c1.name)
    assert isinstance(c2, Connection6DoF)
    assert w1.state[c1.qw.id].position == w2.state[c2.qw.id].position
    np.testing.assert_array_almost_equal(w1.state._data, w2.state._data)

    ws1.close()
    ws2.close()


def test_compute_state_changes_no_changes(rclpy_node):
    w = create_dummy_world()
    s = WorldSynchronizer(node=rclpy_node, _world=w)
    # Immediately compare without changing state
    changes = s.compute_state_changes()
    assert changes == {}
    s.close()


def test_compute_state_changes_single_change(rclpy_node):
    w = create_dummy_world()
    s = WorldSynchronizer(node=rclpy_node, _world=w)
    # change first position
    w.state._data[0, 0] += 1e-3
    changes = s.compute_state_changes()
    names = w.state.keys()
    assert list(changes.keys()) == [names[0]]
    assert np.isclose(changes[names[0]], w.state.positions[0])
    s.close()


def test_compute_state_changes_shape_change_full_snapshot(rclpy_node):
    w = create_dummy_world()
    s = WorldSynchronizer(node=rclpy_node, _world=w)
    # append a new DOF by writing a new name into state
    new_uuid = uuid4()
    w.state._add_dof(new_uuid)
    w.state[new_uuid] = np.zeros(4)
    changes = s.compute_state_changes()
    # full snapshot expected
    assert len(changes) == len(w.state)
    s.close()


def test_compute_state_changes_nan_handling(rclpy_node):
    w = create_dummy_world()
    s = WorldSynchronizer(node=rclpy_node, _world=w)
    # set both previous and current to NaN for entry 0
    w.state._data[0, 0] = np.nan
    s.previous_world_state_data[0] = np.nan
    assert s.compute_state_changes() == {}
    s.close()


def test_attribute_updates(rclpy_node):
    world1 = World(name="w1")
    world2 = World(name="w2")
    world1._id = uuid.UUID(int=1)
    world2._id = uuid.UUID(int=2)

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=world1,
    )
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=world2,
    )

    root = Body(name=PrefixedName("root"))
    with world1.modify_world():
        world1.add_body(root)
    time.sleep(1)
    with world1.modify_world():
        fridge = Fridge.create_with_new_body_in_world(
            name="case",
            world=world1,
            scale=Scale(1, 1, 2.0),
        )
        door = Door.create_with_new_body_in_world(
            name="left_door",
            world=world1,
        )
    time.sleep(1)
    assert [hash(sa) for sa in world1.semantic_annotations] == [
        hash(sa) for sa in world2.semantic_annotations
    ], f"{[sa.name for sa in world1.semantic_annotations]} vs {[sa.name for sa in world2.semantic_annotations]}"

    with world1.modify_world():
        fridge.add(door)

    time.sleep(1)
    assert [hash(sa) for sa in world1.semantic_annotations] == [
        hash(sa) for sa in world2.semantic_annotations
    ], f"{[sa.name for sa in world1.semantic_annotations]} vs {[sa.name for sa in world2.semantic_annotations]}"


@dataclass(eq=False)
class TestAnnotation(SemanticAnnotation):
    value: str = "default"
    entity: Optional[Body] = None
    entities: List[Body] = field(default_factory=list, hash=False)

    @synchronized_attribute_modification
    def update_value(self, new_value: str):
        self.value = new_value

    @synchronized_attribute_modification
    def update_entity(self, new_entity: Body):
        self.entity = new_entity

    @synchronized_attribute_modification
    def add_to_list(self, new_entity: Body):
        self.entities.append(new_entity)

    @synchronized_attribute_modification
    def remove_from_list(self, old_entity: Body):
        self.entities.remove(old_entity)


def test_synchronized_attribute_modification(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")
    sync1 = WorldSynchronizer(node=rclpy_node, _world=w1)
    sync2 = WorldSynchronizer(node=rclpy_node, _world=w2)

    # Allow time for publishers/subscribers to connect
    time.sleep(0.5)

    # 1. Add TestAnnotation and some bodies to w1
    b1 = Body(name=PrefixedName("b1"))
    b2 = Body(name=PrefixedName("b2"))
    anno = TestAnnotation(name=PrefixedName("anno"))

    with w1.modify_world():
        w1.add_body(b1)
        w1.add_body(b2)
        w1.add_connection(FixedConnection(parent=b1, child=b2))
        w1.add_semantic_annotation(anno)

    time.sleep(0.5)

    # Verify initial sync
    assert len(w2.kinematic_structure_entities) == 2
    assert len(w2.semantic_annotations) == 1

    anno2 = w2.semantic_annotations[0]
    assert isinstance(anno2, TestAnnotation)
    assert anno2.value == "default"
    assert anno2.entity is None
    assert len(anno2.entities) == 0

    # 2. Test single attribute modification (primitive)
    with w1.modify_world():
        anno.update_value("new_value")

    time.sleep(0.5)
    assert anno2.value == "new_value"

    # 3. Test single attribute modification (entity)
    with w1.modify_world():
        anno.update_entity(b1)

    time.sleep(0.5)
    assert anno2.entity is not None
    assert anno2.entity.id == b1.id

    # 4. Test list attribute modification (addition)
    with w1.modify_world():
        anno.add_to_list(b1)
        anno.add_to_list(b2)

    time.sleep(0.5)
    assert len(anno2.entities) == 2
    assert w2.get_kinematic_structure_entity_by_id(b1.id) in anno2.entities
    assert w2.get_kinematic_structure_entity_by_id(b2.id) in anno2.entities

    # 5. Test list attribute modification (removal)
    with w1.modify_world():
        anno.remove_from_list(b1)

    time.sleep(0.5)
    assert len(anno2.entities) == 1
    assert w2.get_kinematic_structure_entity_by_id(b1.id) not in anno2.entities
    assert w2.get_kinematic_structure_entity_by_id(b2.id) in anno2.entities

    # 6. Test attribute modification with invalid context
    with pytest.raises(MissingWorldModificationContextError):
        anno.update_value("new_value")

    sync1.close()
    sync2.close()


def test_attribute_update_modification_apply_direct():
    w = World(name="w")
    b1 = Body(name=PrefixedName("b1"))
    anno = TestAnnotation(name=PrefixedName("anno"))
    with w.modify_world():
        w.add_body(b1)
        w.add_semantic_annotation(anno)

    # Test single value update
    mod = AttributeUpdateModification(
        entity_id=anno.id,
        updated_kwargs_json_list=[
            JSONAttributeDiff(
                attribute_name="value", added_values=[to_json("direct_value")]
            )
        ],
    )
    mod.apply(w)
    assert anno.value == "direct_value"

    # Test entity reference update
    mod = AttributeUpdateModification(
        entity_id=anno.id,
        updated_kwargs_json_list=[
            JSONAttributeDiff(attribute_name="entity", added_values=[to_json(b1.id)])
        ],
    )
    mod.apply(w)
    assert anno.entity == b1

    # Test list update (add)
    mod = AttributeUpdateModification(
        entity_id=anno.id,
        updated_kwargs_json_list=[
            JSONAttributeDiff(attribute_name="entities", added_values=[to_json(b1.id)])
        ],
    )
    mod.apply(w)
    assert b1 in anno.entities

    # Test list update (remove)
    mod = AttributeUpdateModification(
        entity_id=anno.id,
        updated_kwargs_json_list=[
            JSONAttributeDiff(
                attribute_name="entities", removed_values=[to_json(b1.id)]
            )
        ],
    )
    mod.apply(w)
    assert b1 not in anno.entities


def test_skipping_incorrect_message(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    with w1.modify_world():
        new_body = Body(name=PrefixedName("b3"))
        w1.add_kinematic_structure_entity(new_body)

    time.sleep(0.2)

    assert len(w1.kinematic_structure_entities) == len(w2.kinematic_structure_entities)

    synchronizer_1.apply_missed_messages()
    with w1.modify_world():
        handle = Handle.create_with_new_body_in_world("handle", w1)

    time.sleep(1)
    assert len(w1.kinematic_structure_entities) == len(w2.kinematic_structure_entities)

    synchronizer_1.close()
    synchronizer_2.close()


@pytest.mark.parametrize("before_w2", [1, 3, 4])
@pytest.mark.parametrize("in_w2", [2, 4, 6])
@pytest.mark.parametrize("after_w2", [1, 2, 3])
def test_world_simultaneous_synchronization_stress_test(
    rclpy_node, before_w2, in_w2, after_w2
):
    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    with w1.modify_world():
        new_body = Body(name=PrefixedName("b3"))
        w1.add_kinematic_structure_entity(new_body)

    w1_ids, w2_ids = wait_for_sync_kse_and_return_ids(w1, w2)

    with w1.modify_world():
        # Create handles before nested context
        for _ in range(before_w2):
            Handle.create_with_new_body_in_world("handle", w1)

        # Nested w2 context
        with w2.modify_world():
            for _ in range(in_w2):
                Handle.create_with_new_body_in_world("handle2", w2)

        # Create handles after nested context
        for _ in range(after_w2):
            Handle.create_with_new_body_in_world("handle", w1)

    w1_ids, w2_ids = wait_for_sync_kse_and_return_ids(w1, w2)
    assert len(w1.kinematic_structure_entities) == len(w2.kinematic_structure_entities)
    assert w1_ids == w2_ids

    synchronizer_1.close()
    synchronizer_2.close()


def test_nested_modify_world_publish_changes_true_false(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    with w1.modify_world():
        new_body = Body(name=PrefixedName("b3"))
        w1.add_kinematic_structure_entity(new_body)

    time.sleep(0.5)

    assert len(w1.kinematic_structure_entities) == len(w2.kinematic_structure_entities)

    with pytest.raises(BrokenWorldModificationHistoryError):
        with w1.modify_world():
            handle = Handle.create_with_new_body_in_world("handle", w1)

            with w1.modify_world(publish_changes=False):
                handle = Handle.create_with_new_body_in_world("handle", w1)

    with pytest.raises(MismatchingPublishChangesAttribute):
        with w1.modify_world(publish_changes=False):
            handle = Handle.create_with_new_body_in_world("handle", w1)

            with w1.modify_world(publish_changes=True):
                handle = Handle.create_with_new_body_in_world("handle", w1)

    synchronizer_1.close()
    synchronizer_2.close()


def test_dont_publish_changes(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = WorldSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    with w1.modify_world(publish_changes=False):
        b1 = Body(name=PrefixedName("b1"))
        w1.add_body(b1)

    assert len(w1.kinematic_structure_entities) - 1 == len(
        w2.kinematic_structure_entities
    )

    synchronizer_1.close()
    synchronizer_2.close()


def test_world_state_update_serialization_round_trip():
    """
    Verify that WorldStateUpdate survives a to_json/from_json round trip.
    """
    meta = MetaData(node_name="test_node", process_id=42)
    original = WorldStateUpdate(
        meta_data=meta,
        ids=[uuid.uuid4(), uuid.uuid4()],
        states=[1.5, 2.5],
        sequence_number=3,
    )

    serialized = to_json(original)
    restored = from_json(serialized)

    assert isinstance(restored, WorldStateUpdate)
    assert restored.meta_data.node_name == original.meta_data.node_name
    assert restored.meta_data.process_id == original.meta_data.process_id
    assert restored.ids == original.ids
    assert restored.states == original.states
    assert restored.sequence_number == 3


def test_load_model_serialization_round_trip():
    """
    Verify that LoadModel survives a to_json/from_json round trip.
    """
    meta = MetaData(node_name="loader", process_id=99)
    original = LoadModel(meta_data=meta, primary_key=7, sequence_number=5)

    serialized = to_json(original)
    restored = from_json(serialized)

    assert isinstance(restored, LoadModel)
    assert restored.primary_key == 7
    assert restored.meta_data.node_name == "loader"
    assert restored.sequence_number == 5


def test_simultaneous_state_and_model_updates(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")

    b1 = Body(name=PrefixedName("b1"))
    b2 = Body(name=PrefixedName("b2"))

    ws1 = WorldSynchronizer(node=rclpy_node, _world=w1)
    ws2 = WorldSynchronizer(node=rclpy_node, _world=w2)

    with w1.modify_world():
        w1.add_body(b1)
        w1.add_body(b2)
        connection = PrismaticConnection.create_with_dofs(
            world=w1, parent=b1, child=b2, axis=Vector3.X()
        )
        w1.add_connection(connection)

    sleep(1)

    synced_connection = w2.get_connections_by_type(PrismaticConnection)[0]

    with w2.modify_world():
        connection.position = 1

        sleep(1)

        assert synced_connection.position == 0

    sleep(1)
    assert synced_connection.position == 1


def test_two_parallel_modify_world_on_same_instance_are_serialized():
    """
    Two threads enter modify_world concurrently; operations must not interleave.
    """
    w = World(name="solo")

    # Seed a single root so the world remains a tree.
    with w.modify_world():
        root = Body(name=PrefixedName("root"))
        w.add_body(root)

    start_barrier = threading.Barrier(2)
    end_barrier = threading.Barrier(2)

    def worker(prefix: str, count: int):
        start_barrier.wait(timeout=2.0)
        with w.modify_world():
            for i in range(count):
                b = Body(name=PrefixedName(f"{prefix}_{i}"))
                w.add_body(b)
                # Keep the graph a tree: attach to root
                w.add_connection(FixedConnection(parent=root, child=b))
            time.sleep(0.05)  # increase contention while still holding the lock
        end_barrier.wait(timeout=2.0)

    t1 = threading.Thread(target=worker, args=("a", 5), daemon=True)
    t2 = threading.Thread(target=worker, args=("b", 5), daemon=True)

    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    # Normalize names: strip the optional prefix like "None/"
    def base(n: str) -> str:
        return n.split("/", 1)[-1]

    names = [base(b.name.name) for b in w.kinematic_structure_entities]
    assert sum(n.startswith("a_") for n in names) == 5
    assert sum(n.startswith("b_") for n in names) == 5


def test_state_changed_inside_a_model_change_arrives_with_the_model(rclpy_node):
    """
    A state change made inside a modification reaches the other world together with the
    model it belongs to.
    """
    receiver_node = rclpy.create_node("lock_order_receiver")
    receiver_executor = SingleThreadedExecutor()
    receiver_executor.add_node(receiver_node)
    rx_thread = threading.Thread(target=receiver_executor.spin, daemon=True)
    rx_thread.start()
    time.sleep(0.1)

    try:
        w1 = World(name="w1")
        w2 = World(name="w2")

        ws1 = WorldSynchronizer(node=rclpy_node, _world=w1)
        ws2 = WorldSynchronizer(node=receiver_node, _world=w2)

        time.sleep(0.2)

        with w1.modify_world():
            w1.add_body(Body(name=PrefixedName("b")))
            # change the state while still inside the modification to stress ordering
            if len(w1.state) > 0:
                w1.state._data[0, 0] = 0.5
                w1.notify_state_change()

        time.sleep(0.3)
        np.testing.assert_array_almost_equal(w1.state._data, w2.state._data)
        assert len(w2.kinematic_structure_entities) == 1

        ws1.close()
        ws2.close()
    finally:
        receiver_executor.shutdown()
        rx_thread.join(timeout=2.0)
        receiver_node.destroy_node()


def test_model_change_arrives_while_state_updates_are_published(rclpy_node):
    """
    A model change reaches the other world even while state updates are streaming.
    """
    receiver_node = rclpy.create_node("recv_node")
    from rclpy.executors import SingleThreadedExecutor

    exec2 = SingleThreadedExecutor()
    exec2.add_node(receiver_node)
    t = threading.Thread(target=exec2.spin, daemon=True)
    t.start()
    time.sleep(0.1)

    try:
        w1 = World(name="w1")
        w2 = World(name="w2")

        ws1 = WorldSynchronizer(node=rclpy_node, _world=w1)
        ws2 = WorldSynchronizer(node=receiver_node, _world=w2)

        # Seed a root
        with w1.modify_world():
            root = Body(name=PrefixedName("seed"))
            w1.add_body(root)

        time.sleep(0.3)

        # Publish state concurrently
        stop = threading.Event()

        def spam_state():
            i = 0
            while not stop.is_set() and i < 50:
                if len(w1.state) > 0:
                    w1.state._data[0, 0] = float(i % 3)
                    w1.notify_state_change()
                time.sleep(0.01)
                i += 1

        th = threading.Thread(target=spam_state, daemon=True)
        th.start()

        # Synchronous model update that preserves tree invariant
        with w1.modify_world():
            new_part = Body(name=PrefixedName("new_part"))
            w1.add_body(new_part)
            w1.add_connection(
                Connection6DoF.create_with_dofs(parent=root, child=new_part, world=w1)
            )

        th.join(timeout=5.0)
        stop.set()
        time.sleep(0.5)

        assert w2.get_kinematic_structure_entity_by_name("new_part") is not None

        ws1.close()
        ws2.close()
    finally:
        exec2.shutdown()
        t.join(timeout=2.0)
        receiver_node.destroy_node()


def test_read_operations_inside_modify_world_do_not_deadlock():
    """
    Read operations inside a write block must not deadlock.
    """
    w = World(name="w")
    with w.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        b2 = Body(name=PrefixedName("b2"))
        w.add_body(b1)
        w.add_body(b2)
        w.add_connection(FixedConnection(parent=b1, child=b2))
        # Calls that traverse graphs and caches while the lock is held
        assert w.root is not None
        assert w.validate()  # must not hang
        assert w.get_kinematic_structure_entity_by_name("b1") is b1


def test_state_diff_during_concurrent_dof_add_remove_is_consistent(rclpy_node):
    """
    When DOFs change concurrently, state diff must not observe torn shapes.
    """
    w = World(name="w")
    ss = WorldSynchronizer(node=rclpy_node, _world=w)

    with w.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        b2 = Body(name=PrefixedName("b2"))
        w.add_body(b1)
        w.add_body(b2)
        c = Connection6DoF.create_with_dofs(parent=b1, child=b2, world=w)
        w.add_connection(c)

    # Worker that changes the number of DOFs by adding/removing a temp 6DoF
    stop = threading.Event()

    def shape_flapper():
        i = 0
        while not stop.is_set() and i < 10:
            with w.modify_world():
                x = Body(name=PrefixedName(f"x{i}"))
                w.add_body(x)
                cc = Connection6DoF.create_with_dofs(parent=b1, child=x, world=w)
                w.add_connection(cc)
            with w.modify_world():
                w.remove_kinematic_structure_entity(x)
            i += 1

    t = threading.Thread(target=shape_flapper, daemon=True)
    t.start()

    # Meanwhile, compute diffs repeatedly; must not raise or produce NaNs spuriously
    for _ in range(50):
        changes = ss.compute_state_changes()
        # All reported names must be in the current world state
        for name in changes.keys():
            assert name in w.state.keys()
        time.sleep(0.01)

    stop.set()
    t.join(timeout=5.0)
    ss.close()


@pytest.mark.skip(
    reason="Exercises an ABBA lock-order inversion on two in-memory World._world_lock "
    "instances held simultaneously in a single process (one thread locks w1 then w2, the other "
    "w2 then w1). Currently we have the assumption that each World lives in its own OS process/address space and"
    "a process only ever holds its own world's lock, so this cross-world deadlock cannot occur."
    "If we decide that we want to support this, remove this skip mark"
)
def test_bidirectional_nested_modify_worlds_no_deadlock(rclpy_node):
    """
    Nested modify_world across two Worlds must not deadlock.
    """
    w1 = World(name="w1")
    w2 = World(name="w2")

    ms1 = WorldSynchronizer(node=rclpy_node, _world=w1)
    ms2 = WorldSynchronizer(node=rclpy_node, _world=w2)

    with w1.modify_world():
        w1.add_body(Body(name=PrefixedName("root1")))

    time.sleep(0.1)
    assert w2.root

    # Thread A: w1 -> w2 nested
    def a():
        for _ in range(5):
            with w1.modify_world():
                Handle.create_with_new_body_in_world("h1", w1)
                with w2.modify_world():
                    Handle.create_with_new_body_in_world("h2", w2)

    # Thread B: w2 -> w1 nested (reverse order)
    def b():
        for _ in range(5):
            with w2.modify_world():
                Handle.create_with_new_body_in_world("g2", w2)
                with w1.modify_world():
                    Handle.create_with_new_body_in_world("g1", w1)

    t1 = threading.Thread(target=a, daemon=True)
    t2 = threading.Thread(target=b, daemon=True)
    t1.start()
    t2.start()

    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    # 5x 2 Handles for both a and b, plus the root
    assert len(w1.kinematic_structure_entities) == 21
    assert len(w2.kinematic_structure_entities) == 21
    w1.validate()
    w2.validate()
    ms1.close()
    ms2.close()


def test_reentrant_modify_world_same_thread():
    """
    Nested modify_world on the same thread must be allowed and safe.
    """
    w = World(name="w")
    with w.modify_world():
        outer = Body(name=PrefixedName("outer"))
        w.add_body(outer)
        with w.modify_world():
            inner = Body(name=PrefixedName("inner"))
            w.add_body(inner)
            w.add_connection(FixedConnection(parent=outer, child=inner))
    assert {b.name.name.split("/", 1)[-1] for b in w.kinematic_structure_entities} == {
        "outer",
        "inner",
    }


def test_world_update_serialization_round_trip():
    """
    WorldUpdate round-trips through to_json / from_json correctly.
    """
    from krrood.adapters.json_serializer import to_json, from_json
    import json

    w = create_dummy_world()
    meta = MetaData(node_name="test_node", process_id=1, world_id=w._id)
    state_msg = WorldStateUpdate(
        meta_data=meta,
        ids=list(w.state.keys())[:2],
        states=[0.1, 0.2],
    )
    update = WorldUpdate(meta_data=meta, state_update=state_msg)

    serialized = json.dumps(to_json(update))
    restored = from_json(json.loads(serialized))

    assert restored.meta_data.node_name == update.meta_data.node_name
    assert restored.modification_block is None
    assert restored.state_update is not None
    assert restored.state_update.states == [0.1, 0.2]


def test_world_synchronizer_basic_state_sync(rclpy_node):
    """
    State changes on w1 are reflected on w2 via the single combined topic.
    """
    w1 = create_dummy_world()
    w2 = create_dummy_world()

    ws1 = WorldSynchronizer(node=rclpy_node, _world=w1)
    ws2 = WorldSynchronizer(node=rclpy_node, _world=w2)
    time.sleep(0.2)

    w1.state._data[0, 0] = 3.14
    w1.notify_state_change()
    time.sleep(0.3)

    assert w2.state._data[0, 0] == pytest.approx(3.14, abs=1e-9)

    ws1.close()
    ws2.close()


def test_world_synchronizer_basic_model_sync(rclpy_node):
    """
    Model changes on w1 (new bodies + connection) are applied on w2.
    """
    w1 = World(name="ws_model_w1")
    w2 = World(name="ws_model_w2")

    ws1 = WorldSynchronizer(node=rclpy_node, _world=w1)
    ws2 = WorldSynchronizer(node=rclpy_node, _world=w2)
    time.sleep(0.2)

    b1 = Body(name=PrefixedName("ws_b1"))
    b2 = Body(name=PrefixedName("ws_b2"))
    with w1.modify_world():
        w1.add_body(b1)
        w1.add_body(b2)
        conn = PrismaticConnection.create_with_dofs(
            world=w1, parent=b1, child=b2, axis=Vector3.X()
        )
        w1.add_connection(conn)

    ids1, ids2 = wait_for_sync_kse_and_return_ids(w1, w2, timeout=5.0)
    assert ids1 == ids2

    ws1.close()
    ws2.close()


def test_world_synchronizer_ordering_no_key_error_after_model_change(rclpy_node):
    """
    Single-topic ordering guarantee: state update is never applied before the model
    update that introduced the DOF UUIDs, so no KeyError or silent data loss occurs.
    """
    w1 = World(name="ws_order_w1")
    w2 = World(name="ws_order_w2")

    ws1 = WorldSynchronizer(node=rclpy_node, _world=w1)
    ws2 = WorldSynchronizer(node=rclpy_node, _world=w2)
    time.sleep(0.2)

    key_error_caught = threading.Event()
    original_apply_state = ws2._apply_state

    def catching_apply_state(msg):
        try:
            original_apply_state(msg)
        except KeyError:
            key_error_caught.set()

    ws2._apply_state = catching_apply_state

    b1 = Body(name=PrefixedName("ws_ord_b1"))
    b2 = Body(name=PrefixedName("ws_ord_b2"))
    with w1.modify_world():
        w1.add_body(b1)
        w1.add_body(b2)
        conn = PrismaticConnection.create_with_dofs(
            world=w1, parent=b1, child=b2, axis=Vector3.X()
        )
        w1.add_connection(conn)

    # Wait for both model and state to propagate
    wait_for_sync_kse_and_return_ids(w1, w2, timeout=5.0)
    time.sleep(0.3)

    assert not key_error_caught.is_set(), (
        "KeyError raised in _apply_state — state update arrived before model update "
        "despite single-topic FIFO ordering guarantee."
    )

    ws1.close()
    ws2.close()


def test_world_synchronizer_missed_messages_applied_in_order(rclpy_node):
    """
    Messages buffered while paused are applied (in order) after apply_missed_messages().
    """
    w1 = create_dummy_world()
    w2 = create_dummy_world()

    ws1 = WorldSynchronizer(node=rclpy_node, _world=w1)
    ws2 = WorldSynchronizer(node=rclpy_node, _world=w2)
    time.sleep(0.2)

    ws2.pause()

    w1.state._data[0, 0] = 9.9
    w1.notify_state_change()
    time.sleep(0.3)

    # While paused w2 must not have the new value
    assert w2.state._data[0, 0] != pytest.approx(9.9, abs=1e-6)
    assert len(ws2.missed_messages) > 0

    ws2.apply_missed_messages()
    time.sleep(0.1)

    assert w2.state._data[0, 0] == pytest.approx(9.9, abs=1e-9)

    ws1.close()
    ws2.close()


def test_stop_is_idempotent(rclpy_node):
    """
    Calling stop() twice must not raise ValueError.
    """
    world = World(name="idempotent_stop_world")
    world_synchronizer = WorldSynchronizer(node=rclpy_node, _world=world)

    world_synchronizer.stop()
    world_synchronizer.stop()

    world_synchronizer.close()


def test_stop_without_close_leaves_ros_resources_alive(rclpy_node):
    """
    Stop() deregisters callbacks but must not destroy the ROS subscriber or publisher.
    """
    world = World(name="stop_no_close_world")
    world_synchronizer = WorldSynchronizer(node=rclpy_node, _world=world)

    world_synchronizer.stop()

    assert (
        world_synchronizer.subscriber is not None
    ), "subscriber must remain alive after stop() — only close() destroys ROS resources"
    assert (
        world_synchronizer.publisher is not None
    ), "publisher must remain alive after stop() — only close() destroys ROS resources"

    world_synchronizer.close()


def test_stop_deregisters_from_model_change_callbacks(rclpy_node):
    """
    After stop(), the synchronizer must no longer be in model_change_callbacks.
    """
    world = World(name="stop_deregister_model_world")
    world_synchronizer = WorldSynchronizer(node=rclpy_node, _world=world)

    assert world_synchronizer in world.get_world_model_manager().model_change_callbacks

    world_synchronizer.stop()

    assert (
        world_synchronizer not in world.get_world_model_manager().model_change_callbacks
    )

    world_synchronizer.close()


def test_stop_deregisters_from_state_change_callbacks(rclpy_node):
    """
    After stop(), the synchronizer must no longer be in state_change_callbacks.
    """
    world = World(name="stop_deregister_state_world")
    world_synchronizer = WorldSynchronizer(node=rclpy_node, _world=world)

    assert world_synchronizer in world.state.state_change_callbacks

    world_synchronizer.stop()

    assert world_synchronizer not in world.state.state_change_callbacks

    world_synchronizer.close()


def test_apply_missed_messages_interleaved_model_and_state(rclpy_node):
    """
    Messages buffered while paused are applied in order even when model and state
    messages are interleaved — the model message must be applied before the state
    message that references its DOFs.
    """
    world_1 = World(name="interleaved_w1")
    world_2 = World(name="interleaved_w2")

    world_synchronizer_1 = WorldSynchronizer(node=rclpy_node, _world=world_1)
    world_synchronizer_2 = WorldSynchronizer(node=rclpy_node, _world=world_2)

    time.sleep(0.2)

    world_synchronizer_2.pause()

    body_1 = Body(name=PrefixedName("interleaved_b1"))
    body_2 = Body(name=PrefixedName("interleaved_b2"))

    with world_1.modify_world():
        world_1.add_body(body_1)
        world_1.add_body(body_2)
        prismatic_connection = PrismaticConnection.create_with_dofs(
            world=world_1, parent=body_1, child=body_2, axis=Vector3.X()
        )
        world_1.add_connection(prismatic_connection)

    time.sleep(0.2)

    world_1.state[prismatic_connection.dof.id].position = 3.5
    world_1.notify_state_change()

    time.sleep(0.2)

    assert (
        len(world_synchronizer_2.missed_messages) == 3
    ), "expected at least a model message and a state message to be buffered"
    assert len(world_2.kinematic_structure_entities) == 0

    world_synchronizer_2.resume()
    world_synchronizer_2.apply_missed_messages()

    assert len(world_2.kinematic_structure_entities) == 2
    synchronized_prismatic_connections = world_2.get_connections_by_type(
        PrismaticConnection
    )
    assert len(synchronized_prismatic_connections) == 1
    assert synchronized_prismatic_connections[0].position == pytest.approx(
        3.5, abs=1e-9
    )

    world_synchronizer_1.close()
    world_synchronizer_2.close()


def test_apply_state_with_unknown_identifier_raises(rclpy_node):
    """
    _apply_state must raise StateUpdateContainsUnknownDegreesOfFreedomError when any DOF
    identifier in the WorldStateUpdate is absent from the world state index, whether
    that is one unknown identifier or all of them.
    """
    world = create_dummy_world()
    world_synchronizer = WorldSynchronizer(node=rclpy_node, _world=world)

    known_identifier = world.state.keys()[0]
    unknown_identifier = uuid4()

    partially_unknown_state_update = WorldStateUpdate(
        meta_data=world_synchronizer.meta_data,
        ids=[known_identifier, unknown_identifier],
        states=[0.0, 9.9],
    )

    with pytest.raises(StateUpdateContainsUnknownDegreesOfFreedomError):
        world_synchronizer._apply_state(partially_unknown_state_update)

    all_unknown_state_update = WorldStateUpdate(
        meta_data=world_synchronizer.meta_data,
        ids=[uuid4(), uuid4()],
        states=[1.0, 2.0],
    )

    with pytest.raises(StateUpdateContainsUnknownDegreesOfFreedomError):
        world_synchronizer._apply_state(all_unknown_state_update)

    world_synchronizer.close()


def test_apply_missed_messages_inside_modify_world_raises(rclpy_node):
    """
    Calling apply_missed_messages() while a modify_world context is active must raise
    ApplyMissedMessagesWhileWorldIsBeingModifiedError before attempting to apply any
    message (which would otherwise cause a MismatchingPublishChangesAttribute crash).
    """
    world_1 = World(name="missed_in_modify_w1")
    world_2 = World(name="missed_in_modify_w2")

    world_synchronizer_1 = WorldSynchronizer(node=rclpy_node, _world=world_1)
    world_synchronizer_2 = WorldSynchronizer(node=rclpy_node, _world=world_2)

    world_synchronizer_2.pause()

    time.sleep(0.2)

    with world_1.modify_world():
        new_body = Body(name=PrefixedName("body_for_missed_in_modify"))
        world_1.add_kinematic_structure_entity(new_body)

    time.sleep(0.2)

    assert len(world_synchronizer_2.missed_messages) >= 1

    world_synchronizer_2.resume()

    with pytest.raises(ApplyMissedMessagesWhileWorldIsBeingModifiedError):
        with world_2.modify_world():
            world_synchronizer_2.apply_missed_messages()

    world_synchronizer_1.close()
    world_synchronizer_2.close()


def test_apply_state_does_not_deadlock_when_callback_acquires_world_lock(rclpy_node):
    """
    _apply_state must call notify_state_change after releasing _world_lock so that a
    StateChangeCallback whose on_state_change acquires _world_lock from a separate
    thread does not deadlock.

    A 3-second thread-join timeout is used as the deadlock sentinel — the test fails if
    the publish does not complete within that window.
    """
    receiver_node = rclpy.create_node("deadlock_test_receiver")
    receiver_executor = SingleThreadedExecutor()
    receiver_executor.add_node(receiver_node)
    receiver_thread = threading.Thread(
        target=receiver_executor.spin, daemon=True, name="deadlock-receiver"
    )
    receiver_thread.start()
    time.sleep(0.1)

    try:
        world_1 = create_dummy_world()
        world_2 = create_dummy_world()

        world_synchronizer_1 = WorldSynchronizer(node=rclpy_node, _world=world_1)
        world_synchronizer_2 = WorldSynchronizer(node=receiver_node, _world=world_2)

        @dataclass(eq=False)
        class LockAcquiringStateCallback(StateChangeCallback):
            """
            A state callback that acquires _world_lock from inside on_state_change.
            """

            def on_state_change(self, **kwargs):
                with self._world._world_lock:
                    pass

        locking_callback = LockAcquiringStateCallback(_world=world_2)

        time.sleep(0.2)

        completed = threading.Event()

        def trigger_state_change():
            world_1.state._data[0, 0] = 5.55
            world_1.notify_state_change()
            completed.set()

        trigger_thread = threading.Thread(target=trigger_state_change, daemon=True)
        trigger_thread.start()
        trigger_thread.join(timeout=3.0)

        assert completed.is_set(), (
            "Deadlock detected: notify_state_change did not complete within 3 seconds. "
            "Likely cause: notify_state_change is called while _world_lock is held in _apply_state."
        )

        locking_callback.stop()
        world_synchronizer_1.close()
        world_synchronizer_2.close()
    finally:
        receiver_executor.shutdown()
        receiver_thread.join(timeout=2.0)
        receiver_node.destroy_node()


def test_model_publish_does_not_hold_world_lock(rclpy_node):
    """
    A model update must be published *after* ``_world_lock`` is released, so that the
    modification it describes is complete by the time it leaves this process.
    """
    w = World()
    ms = WorldSynchronizer(node=rclpy_node, _world=w)

    lock_free_during_publish = []
    original_publish = ms.publish

    def probing_publish(msg):
        lock_free_during_publish.append(probe_lock_is_free(w._world_lock, 0.3))
        return original_publish(msg)

    ms.publish = probing_publish

    with w.modify_world():
        w.add_body(Body(name=PrefixedName("publish_lock_body")))

    ms.close()

    assert lock_free_during_publish, "synchronizer.publish was never called"
    assert all(lock_free_during_publish), (
        "Model update was published while _world_lock was held. Publishing must happen "
        "outside the modify_world critical section so inbound applies / acks cannot deadlock."
    )


def test_state_publish_does_not_hold_world_lock(rclpy_node):
    """
    A state update triggered from within ``modify_world`` must be published after the
    lock is released.
    """
    w = create_dummy_world()
    ms = WorldSynchronizer(node=rclpy_node, _world=w)

    state_publish_lock_free = []
    original_publish = ms.publish

    def probing_publish(msg):
        lock_free = probe_lock_is_free(w._world_lock, 0.3)
        if msg.state_update is not None:
            state_publish_lock_free.append(lock_free)
        return original_publish(msg)

    ms.publish = probing_publish

    with w.modify_world():
        w.state._data[0, 0] = 1.234

    ms.close()

    assert state_publish_lock_free, "no state update was published"
    assert all(
        state_publish_lock_free
    ), "State update was published while _world_lock was held."


def test_state_change_during_a_modification_is_published_after_its_model_change(
    rclpy_node,
):
    """
    A thread that waits for the world lock while a modification is running describes the
    world that modification produced, so whatever it announces afterwards must leave
    this process behind the model change of that modification.
    """
    world = create_dummy_world()
    synchronizer = WorldSynchronizer(
        node=rclpy_node, _world=world, topic_name=f"/publication_order_{uuid4().hex}"
    )

    published_updates: List[WorldUpdate] = []
    original_publish = synchronizer.publish

    def slowly_publishing_model_updates(update: WorldUpdate):
        if update.modification_block is not None:
            # Widen the window in which the announcing thread could overtake this update.
            time.sleep(0.2)
        published_updates.append(update)
        return original_publish(update)

    synchronizer.publish = slowly_publishing_model_updates

    def announce_state_change():
        # Waiting for the world lock is what puts this thread behind the modification.
        with world._world_lock:
            world.state._data[0, 0] = 1.5
        world.notify_state_change()

    try:
        with world.modify_world():
            child_body = Body(name=PrefixedName("publication_order_child"))
            world.add_body(child_body)
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=world.root, child=child_body, world=world
                )
            )
            announcing_thread = threading.Thread(target=announce_state_change)
            announcing_thread.start()
            time.sleep(0.1)

        announcing_thread.join(timeout=5.0)
        assert not announcing_thread.is_alive()

        model_update_positions = [
            update.sequence_number
            for update in published_updates
            if update.modification_block is not None
        ]
        state_update_positions = [
            update.sequence_number
            for update in published_updates
            if update.state_update is not None
        ]
        assert len(model_update_positions) == 1
        assert state_update_positions
        assert max(model_update_positions) < min(state_update_positions)
    finally:
        synchronizer.close()


def test_inbound_message_deserialization_holds_world_lock(rclpy_node):
    """
    ``subscription_callback`` reads the world (building the id/kwargs tracker and
    running ``from_json``) *before* acquiring ``_world_lock``, racing with concurrent
    modifications.

    Deserialization that reads world structure must hold the lock.
    """
    receiver_node = rclpy.create_node("deserialize_lock_receiver")
    receiver_executor = SingleThreadedExecutor()
    receiver_executor.add_node(receiver_node)
    receiver_thread = threading.Thread(
        target=receiver_executor.spin, daemon=True, name="deserialize-lock-receiver"
    )
    receiver_thread.start()
    time.sleep(0.1)

    original_from_world = WorldEntityWithIDKwargsTracker.from_world
    try:
        w1 = create_dummy_world()
        w2 = create_dummy_world()

        ms1 = WorldSynchronizer(node=rclpy_node, _world=w1)
        ms2 = WorldSynchronizer(node=receiver_node, _world=w2)
        time.sleep(0.3)

        lock_held_during_deserialization = {}

        def patched_from_world(world, *args, **kwargs):
            if world is w2 and "held" not in lock_held_during_deserialization:
                lock_held_during_deserialization["held"] = not probe_lock_is_free(
                    w2._world_lock, 0.2
                )
            return original_from_world(world, *args, **kwargs)

        WorldEntityWithIDKwargsTracker.from_world = staticmethod(patched_from_world)

        with w1.modify_world():
            race_body = Body(name=PrefixedName("deserialize_race_body"))
            w1.add_body(race_body)
            w1.add_connection(FixedConnection(parent=w1.root, child=race_body))

        assert wait_for_condition(
            lambda: "held" in lock_held_during_deserialization, timeout=3.0
        ), "Receiver never deserialized the inbound message"

        assert lock_held_during_deserialization["held"], (
            "Inbound message was deserialized (reading world structure) without holding "
            "_world_lock, racing with concurrent world modifications."
        )

        ms1.close()
        ms2.close()
    finally:
        WorldEntityWithIDKwargsTracker.from_world = original_from_world
        receiver_executor.shutdown()
        receiver_thread.join(timeout=2.0)
        receiver_node.destroy_node()


def test_callback_removal_during_notify_does_not_skip_callbacks():
    """
    Model-change callbacks are iterated over the *live* list, so a callback removing
    itself during notification shifts the indices and silently skips the following
    callback.
    """
    world = World()

    @dataclass(eq=False)
    class _RecordingModelCallback(ModelChangeCallback):
        fired: int = 0
        remove_self_on_fire: bool = False

        def on_model_change(self, **kwargs):
            self.fired += 1
            if self.remove_self_on_fire:
                self._world.get_world_model_manager().model_change_callbacks.remove(
                    self
                )

    # Registration order is [first, second]; the first removes itself when fired.
    first = _RecordingModelCallback(_world=world, remove_self_on_fire=True)
    second = _RecordingModelCallback(_world=world)

    with world.modify_world():
        world.add_body(Body(name=PrefixedName("callback_iter_body")))

    assert first.fired == 1, "the self-removing callback should still have fired once"
    assert second.fired == 1, (
        "the second callback was skipped because the callback list was mutated during "
        "iteration; notification must iterate over a copy."
    )


def test_apply_missed_messages_is_atomic_against_concurrent_modify(rclpy_node):
    """
    ``apply_missed_messages`` must apply the buffered batch atomically with respect to other
    threads: it should hold ``_world_lock`` across the whole batch so a concurrent ``modify_world``
    serializes behind it instead of interleaving between messages.

    Probed by checking, from another thread, whether the world lock is held at the moment
    ``apply_message`` is entered. Red today (the lock is free between messages), green once
    ``apply_missed_messages`` holds the lock across the batch.
    """
    w1 = create_dummy_world()
    w2 = create_dummy_world()

    ms1 = WorldSynchronizer(node=rclpy_node, _world=w1)
    ms2 = WorldSynchronizer(node=rclpy_node, _world=w2)
    time.sleep(0.3)

    ms2.pause()

    with w1.modify_world():
        toctou_body = Body(name=PrefixedName("toctou_body"))
        w1.add_body(toctou_body)
        w1.add_connection(FixedConnection(parent=w1.root, child=toctou_body))

    assert wait_for_condition(
        lambda: len(ms2.missed_messages) >= 1, timeout=3.0
    ), "paused receiver never buffered the model change"
    ms2.resume()

    lock_held_during_apply = {}
    original_apply_message = ms2.apply_message

    def probing_apply_message(message):
        # Probe at entry (before apply_message takes its own reentrant lock): the lock is only held
        # here if apply_missed_messages holds it across the batch.
        if "held" not in lock_held_during_apply:
            lock_held_during_apply["held"] = not probe_lock_is_free(w2._world_lock, 0.3)
        return original_apply_message(message)

    ms2.apply_message = probing_apply_message

    try:
        ms2.apply_missed_messages()
        assert lock_held_during_apply.get("held") is True, (
            "apply_missed_messages did not hold _world_lock across the batch; a concurrent "
            "modify_world could interleave between buffered messages."
        )
    finally:
        ms2.apply_message = original_apply_message
        ms1.close()
        ms2.close()


def test_combined_update_model_and_state_applied_atomically(rclpy_node):
    """
    ``apply_message`` applies the model block and the state update under *separate*
    ``_world_lock`` acquisitions, leaving a window where the new structure is visible
    without its state.

    The combined update must be applied atomically.
    """
    source_world = create_dummy_world()
    receiver_world = create_dummy_world()

    ms = WorldSynchronizer(node=rclpy_node, _world=receiver_world)

    # Build a real model-modification block (adds one DOF) on the source world.
    with source_world.modify_world():
        new_body = Body(name=PrefixedName("atomic_body"))
        source_world.add_body(new_body)
        connection = PrismaticConnection.create_with_dofs(
            world=source_world,
            parent=source_world.root,
            child=new_body,
            axis=Vector3.X(),
        )
        source_world.add_connection(connection)

    model_block = source_world.get_world_model_manager().model_modification_blocks[-1]
    new_dof_id = connection.dof.id

    source_meta = MetaData(node_name="atomic_src", process_id=0, world_id=uuid4())
    combined_update = WorldUpdate(
        meta_data=source_meta,
        modification_block=ModificationBlock(
            meta_data=source_meta, modifications=model_block
        ),
        state_update=WorldStateUpdate(
            meta_data=source_meta, ids=[new_dof_id], states=[0.73]
        ),
    )

    # Round-trip through JSON against the receiver's tracker (as subscription_callback does)
    # so the applied modifications operate on fresh objects owned by the receiver world.
    tracker = WorldEntityWithIDKwargsTracker.from_world(receiver_world)
    combined_update = from_json(
        json.loads(json.dumps(to_json(combined_update))), **tracker.create_kwargs()
    )

    lock_held_between_model_and_state = {}
    original_apply_model = ms._apply_model

    def probing_apply_model(modification_block_message):
        original_apply_model(modification_block_message)
        # We are now in the gap between model application and state application.
        lock_held_between_model_and_state["held"] = not probe_lock_is_free(
            receiver_world._world_lock, 0.3
        )

    ms._apply_model = probing_apply_model

    ms.apply_message(combined_update)
    ms.close()

    assert lock_held_between_model_and_state.get("held") is True, (
        "_world_lock was released between applying the model block and the state update; "
        "a combined WorldUpdate must be applied atomically under a single lock."
    )


# %% deferring incoming updates without silencing outgoing ones


@dataclass(eq=False)
class DelayedApplyWorldSynchronizer(WorldSynchronizer):
    """
    A synchronizer that applies a single message slowly, so another thread can append to
    the buffer while a drain is still running.
    """

    apply_delay: float = 0.3
    """
    Seconds spent in every ``apply_message`` before the message is really applied.
    """

    def apply_message(self, message: WorldUpdate):
        time.sleep(self.apply_delay)
        super().apply_message(message)


def create_connected_worlds(
    rclpy_node: Node, name: str
) -> Tuple[World, World, WorldSynchronizer, WorldSynchronizer]:
    """
    Build a publishing world and a receiving world that share one prismatic connection.

    The receiver applies inline while it is being set up; the caller switches it to
    deferring afterwards.
    """
    publisher_world = World(name=f"{name}_publisher")
    receiver_world = World(name=f"{name}_receiver")
    publisher_synchronizer = WorldSynchronizer(node=rclpy_node, _world=publisher_world)
    receiver_synchronizer = WorldSynchronizer(node=rclpy_node, _world=receiver_world)
    time.sleep(0.2)

    with publisher_world.modify_world():
        parent_body = Body(name=PrefixedName(f"{name}_parent"))
        child_body = Body(name=PrefixedName(f"{name}_child"))
        publisher_world.add_body(parent_body)
        publisher_world.add_body(child_body)
        publisher_world.add_connection(
            PrismaticConnection.create_with_dofs(
                world=publisher_world,
                parent=parent_body,
                child=child_body,
                axis=Vector3.X(),
            )
        )
    assert wait_for_condition(
        lambda: len(receiver_world.kinematic_structure_entities) == 2
    ), "the receiver never picked up the initial model"
    return (
        publisher_world,
        receiver_world,
        publisher_synchronizer,
        receiver_synchronizer,
    )


def publish_position(world: World, position: float) -> None:
    """
    Move the prismatic connection of the given world and announce the change.
    """
    connection = world.get_connections_by_type(PrismaticConnection)[0]
    world.state[connection.dof.id].position = position
    world.notify_state_change()


def test_deferring_incoming_updates_keeps_outgoing_publishing_alive(rclpy_node):
    """
    Deferring is one-directional: a synchronizer that queues incoming updates must still
    publish its own model and state changes, so an owner of the world does not have to
    reach around the callbacks to publish.
    """
    world_1 = World(name="one_directional_deferring_1")
    world_2 = World(name="one_directional_deferring_2")
    synchronizer_1 = WorldSynchronizer(
        node=rclpy_node, _world=world_1, defer_incoming_updates=True
    )
    synchronizer_2 = WorldSynchronizer(node=rclpy_node, _world=world_2)
    time.sleep(0.2)

    try:
        with world_1.modify_world():
            parent_body = Body(name=PrefixedName("one_directional_parent"))
            child_body = Body(name=PrefixedName("one_directional_child"))
            world_1.add_body(parent_body)
            world_1.add_body(child_body)
            world_1.add_connection(
                PrismaticConnection.create_with_dofs(
                    world=world_1,
                    parent=parent_body,
                    child=child_body,
                    axis=Vector3.X(),
                )
            )
        assert wait_for_condition(
            lambda: len(world_2.kinematic_structure_entities) == 2
        ), "the model modification of the deferring synchronizer was not published"

        publish_position(world_1, 1.25)
        assert wait_for_condition(
            lambda: world_2.get_connections_by_type(PrismaticConnection)[0].position
            == pytest.approx(1.25, abs=1e-9)
        ), "the state change of the deferring synchronizer was not published"
    finally:
        synchronizer_1.close()
        synchronizer_2.close()


def test_state_updates_are_applied_up_to_the_next_model_modification(rclpy_node):
    """
    Draining stops at the first buffered model modification, so state that was published
    before it can be consumed while the modification itself keeps waiting in order.
    """
    (
        publisher_world,
        receiver_world,
        publisher_synchronizer,
        receiver_synchronizer,
    ) = create_connected_worlds(rclpy_node, "partial_drain")
    receiver_synchronizer.defer_incoming_updates = True

    try:
        publish_position(publisher_world, 1.5)
        # A fixed connection carries no degree of freedom, so this modification does not
        # publish a state message of its own and the buffer stays predictable.
        with publisher_world.modify_world():
            late_body = Body(name=PrefixedName("partial_drain_late_body"))
            publisher_world.add_body(late_body)
            publisher_world.add_connection(
                FixedConnection(parent=publisher_world.root, child=late_body)
            )
        publish_position(publisher_world, 2.5)
        assert wait_for_condition(
            lambda: len(receiver_synchronizer.missed_messages) == 3
        ), "expected a state, a model and a second state message to be buffered"

        receiver_synchronizer.apply_missed_state_updates()

        assert receiver_world.get_connections_by_type(PrismaticConnection)[
            0
        ].position == pytest.approx(1.5, abs=1e-9)
        assert len(receiver_world.kinematic_structure_entities) == 2
        assert len(receiver_synchronizer.missed_messages) == 2
        assert receiver_synchronizer.has_buffered_model_modification

        receiver_synchronizer.apply_missed_messages()

        assert len(receiver_world.kinematic_structure_entities) == 3
        assert receiver_world.get_connections_by_type(PrismaticConnection)[
            0
        ].position == pytest.approx(2.5, abs=1e-9)
        assert not receiver_synchronizer.has_buffered_model_modification
    finally:
        publisher_synchronizer.close()
        receiver_synchronizer.close()


def test_draining_state_updates_leaves_nothing_behind_without_model_modifications(
    rclpy_node,
):
    """
    Without a buffered model modification the drain consumes the whole buffer, so a
    caller draining every cycle never accumulates a backlog.
    """
    (
        publisher_world,
        receiver_world,
        publisher_synchronizer,
        receiver_synchronizer,
    ) = create_connected_worlds(rclpy_node, "full_state_drain")
    receiver_synchronizer.defer_incoming_updates = True

    try:
        publish_position(publisher_world, 0.5)
        publish_position(publisher_world, 1.5)
        assert wait_for_condition(
            lambda: len(receiver_synchronizer.missed_messages) == 2
        )

        receiver_synchronizer.apply_missed_state_updates()

        assert receiver_synchronizer.missed_messages == []
        assert receiver_world.get_connections_by_type(PrismaticConnection)[
            0
        ].position == pytest.approx(1.5, abs=1e-9)
    finally:
        publisher_synchronizer.close()
        receiver_synchronizer.close()


def test_deferred_model_reload_is_only_applied_on_demand(
    rclpy_node, in_memory_session_maker
):
    """
    A reload replaces the whole world, so a process controlling that world has to be
    able to postpone it instead of having it applied on the receiving thread.
    """
    publisher_world = create_dummy_world()
    receiver_world = World()
    publisher_synchronizer = ModelReloadSynchronizer(
        node=rclpy_node, _world=publisher_world, session=in_memory_session_maker()
    )
    receiver_synchronizer = ModelReloadSynchronizer(
        node=rclpy_node,
        _world=receiver_world,
        session=in_memory_session_maker(),
        defer_incoming_reloads=True,
    )

    try:
        publisher_synchronizer.publish_reload_model()
        assert wait_for_condition(lambda: receiver_synchronizer.has_pending_reload)
        assert len(receiver_world.kinematic_structure_entities) == 0

        receiver_synchronizer.apply_pending_reload()

        assert len(receiver_world.kinematic_structure_entities) == 2
        assert not receiver_synchronizer.has_pending_reload
    finally:
        publisher_synchronizer.close()
        receiver_synchronizer.close()


def test_message_arriving_during_a_drain_stays_buffered(rclpy_node):
    """
    The buffer is appended to on the subscription thread while its owner drains it, so
    the drain must remove exactly the messages it applied and keep the rest.
    """
    world = World(name="drain_race")
    synchronizer = DelayedApplyWorldSynchronizer(
        node=rclpy_node,
        _world=world,
        defer_incoming_updates=True,
        topic_name=f"/drain_race_{uuid4().hex}",
    )
    try:
        empty_state_update = WorldStateUpdate(
            meta_data=synchronizer.meta_data, ids=[], states=[]
        )
        synchronizer.missed_messages.append(
            WorldUpdate(
                meta_data=synchronizer.meta_data, state_update=empty_state_update
            )
        )
        drain = threading.Thread(
            target=synchronizer.apply_missed_messages, name="drain-race"
        )
        drain.start()
        time.sleep(0.1)
        late_message = WorldUpdate(
            meta_data=synchronizer.meta_data, state_update=empty_state_update
        )
        synchronizer._subscription_callback(late_message)
        drain.join(timeout=5.0)

        assert not drain.is_alive()
        assert synchronizer.missed_messages == [late_message]
    finally:
        synchronizer.close()


# %% positions in the stream of a publisher


def test_a_message_reports_where_it_sits_in_the_stream_of_its_publisher():
    """
    A message carries everything a reader needs to tell how far it caught up with the
    publisher that sent it.
    """
    meta_data = MetaData(node_name="publisher", process_id=1)

    message = WorldUpdate(meta_data=meta_data, sequence_number=3)

    assert message.position == StreamPosition(origin=meta_data, sequence_number=3)


def test_every_publication_advances_the_stream_position(rclpy_node):
    """
    Positions count the messages one synchronizer sent, so a reader of the stream can
    tell how much of it it has seen.
    """
    world = World(name="stream_position")
    synchronizer = WorldSynchronizer(
        node=rclpy_node, _world=world, topic_name=f"/stream_position_{uuid4().hex}"
    )
    try:
        assert synchronizer.published_sequence_number == 0

        for expected_position in (1, 2, 3):
            synchronizer.publish(
                WorldUpdate(
                    meta_data=synchronizer.meta_data,
                    state_update=WorldStateUpdate(
                        meta_data=synchronizer.meta_data, ids=[], states=[]
                    ),
                )
            )
            assert synchronizer.published_sequence_number == expected_position
            assert (
                synchronizer.latest_published_position.sequence_number
                == expected_position
            )
            assert synchronizer.latest_published_position.origin == (
                synchronizer.meta_data
            )
    finally:
        synchronizer.close()


def test_applying_an_update_records_the_position_of_its_publisher(rclpy_node):
    """
    Catching up is tracked per publisher, because positions of different publishers say
    nothing about each other.
    """
    (
        publisher_world,
        receiver_world,
        publisher_synchronizer,
        receiver_synchronizer,
    ) = create_connected_worlds(rclpy_node, "applied_position")
    try:
        publish_position(publisher_world, 0.75)
        position = publisher_synchronizer.latest_published_position

        assert wait_for_condition(
            lambda: receiver_synchronizer.has_applied(position)
        ), "the receiver never caught up with the published position"
        assert receiver_synchronizer.has_applied(
            StreamPosition(
                origin=position.origin, sequence_number=position.sequence_number - 1
            )
        )
        assert not receiver_synchronizer.has_applied(
            StreamPosition(
                origin=position.origin, sequence_number=position.sequence_number + 1
            )
        )
    finally:
        publisher_synchronizer.close()
        receiver_synchronizer.close()


def test_a_publisher_that_was_never_heard_from_is_at_the_start_of_its_stream(
    rclpy_node,
):
    """
    Nothing of an unknown publisher was applied, so its first message is still awaited.
    """
    world = World(name="unknown_publisher")
    synchronizer = WorldSynchronizer(
        node=rclpy_node, _world=world, topic_name=f"/unknown_publisher_{uuid4().hex}"
    )
    stranger = MetaData(node_name="stranger", process_id=1)
    try:
        assert synchronizer.has_applied(
            StreamPosition(origin=stranger, sequence_number=0)
        )
        assert not synchronizer.has_applied(
            StreamPosition(origin=stranger, sequence_number=1)
        )
    finally:
        synchronizer.close()


def test_a_buffered_update_does_not_count_as_caught_up_with(rclpy_node):
    """
    A deferred update has not reached the world yet, so whoever waits for it must keep
    waiting until it is applied.
    """
    (
        publisher_world,
        receiver_world,
        publisher_synchronizer,
        receiver_synchronizer,
    ) = create_connected_worlds(rclpy_node, "buffered_position")
    receiver_synchronizer.defer_incoming_updates = True
    try:
        publish_position(publisher_world, 0.5)
        position = publisher_synchronizer.latest_published_position
        assert wait_for_condition(
            lambda: len(receiver_synchronizer.missed_messages) == 1
        ), "the update was never received"

        assert not receiver_synchronizer.has_applied(position)

        receiver_synchronizer.apply_missed_messages()

        assert receiver_synchronizer.has_applied(position)
    finally:
        publisher_synchronizer.close()
        receiver_synchronizer.close()


def test_the_synchronizer_of_a_world_is_found_through_the_world(rclpy_node):
    """
    A client that was handed a world, but not the synchronizer publishing its changes,
    still has to name the stream its positions belong to.
    """
    world = World(name="synchronizer_lookup")
    synchronizer = WorldSynchronizer(
        node=rclpy_node, _world=world, topic_name=f"/synchronizer_lookup_{uuid4().hex}"
    )
    try:
        assert WorldSynchronizer.of_world(world) is synchronizer
    finally:
        synchronizer.close()


def test_a_world_that_publishes_nowhere_has_no_synchronizer():
    """
    Referring to the stream of a world that has none is a mistake worth naming.
    """
    world = World(name="no_synchronizer")

    with pytest.raises(WorldHasNoSynchronizerError):
        WorldSynchronizer.of_world(world)


def test_several_synchronizers_leave_the_stream_of_a_world_undecided(rclpy_node):
    """
    Positions name one stream, so a world publishing through several synchronizers
    cannot answer which one is meant.
    """
    world = World(name="ambiguous_synchronizer")
    first = WorldSynchronizer(
        node=rclpy_node, _world=world, topic_name=f"/ambiguous_a_{uuid4().hex}"
    )
    second = WorldSynchronizer(
        node=rclpy_node, _world=world, topic_name=f"/ambiguous_b_{uuid4().hex}"
    )
    try:
        with pytest.raises(WorldHasMultipleSynchronizersError):
            WorldSynchronizer.of_world(world)
    finally:
        first.close()
        second.close()


if __name__ == "__main__":
    unittest.main()
