import hashlib
import os
import threading
import time
import unittest
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Set, Tuple
from uuid import uuid4

import numpy as np
import pytest
import sqlalchemy
from pkg_resources import resource_filename
from sqlalchemy import select
from sqlalchemy.orm import Session

from krrood.ormatic.utils import drop_database, create_engine
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    StateSynchronizer,
    ModelReloadSynchronizer,
    ModelSynchronizer,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    MissingWorldModificationContextError,
    MismatchingPublishChangesAttribute,
)
from semantic_digital_twin.orm.ormatic_interface import Base, WorldMappingDAO
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Door,
    Fridge,
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
from semantic_digital_twin.adapters.ros.messages import (
    MetaData,
    WorldStateUpdate,
    LoadModel,
    Acknowledgment,
)


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
                x_id=x_dof.id,
                y_id=y_dof.id,
                z_id=z_dof.id,
                qx_id=qx_dof.id,
                qy_id=qy_dof.id,
                qz_id=qz_dof.id,
                qw_id=qw_dof.id,
            )
        )
    return w


def wait_for_sync_kse_and_return_ids(
    w1: World, w2: World, timeout: float = 5.0, interval: float = 0.05
) -> Tuple[Set[uuid.UUID], Set[uuid.UUID]]:
    """
    Waits until the sets of kinematic structure entity IDs in both worlds are identical, or until the timeout is reached.

    :param w1: The first world.
    :param w2: The second world.
    :param timeout: The maximum time to wait for synchronization, in seconds. Defaults to 5.0.
    :param interval: The time interval between checks, in seconds. Defaults to 0.05.

    :return: A tuple containing the sets of kinematic structure entity IDs in both worlds.
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


def test_state_synchronization(rclpy_node):
    w1 = create_dummy_world()
    w2 = create_dummy_world()

    synchronizer_1 = StateSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = StateSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    # Allow time for publishers/subscribers to connect on unique topics
    time.sleep(0.2)

    w1.state.data[0, 0] = 1.0
    w1.notify_state_change()
    time.sleep(0.2)
    assert w1.state.data[0, 0] == 1.0
    assert w1.state.data[0, 0] == w2.state.data[0, 0]

    synchronizer_1.close()
    synchronizer_2.close()


def test_state_synchronization_world_model_change_after_init(rclpy_node):
    w1 = World()
    w2 = World()

    synchronizer_1 = StateSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    create_dummy_world(w1)
    create_dummy_world(w2)
    synchronizer_2 = StateSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    # Allow time for publishers/subscribers to connect on unique topics
    time.sleep(0.2)

    w1.state.data[0, 0] = 1.0
    w1.notify_state_change()
    time.sleep(0.2)
    assert w1.state.data[0, 0] == 1.0
    assert w1.state.data[0, 0] == w2.state.data[0, 0]

    synchronizer_1.close()
    synchronizer_2.close()


def test_model_reload(rclpy_node):
    engine = create_engine(
        "sqlite+pysqlite:///file::memory:?cache=shared",
        connect_args={"check_same_thread": False, "uri": True},
    )
    drop_database(engine)
    Base.metadata.create_all(engine)
    session_maker = sqlalchemy.orm.sessionmaker(bind=engine)
    session1 = session_maker()
    session2 = session_maker()

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

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
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

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
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
    time.sleep(0.1)
    assert len(w1.kinematic_structure_entities) == 2
    assert len(w2.kinematic_structure_entities) == 2
    assert len(w1.connections) == 1
    assert len(w2.connections) == 1

    synchronizer_1.close()
    synchronizer_2.close()


def test_model_synchronization_merge_full_world(rclpy_node):

    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    pr2_world = URDFParser.from_file(
        os.path.join(
            resource_filename("semantic_digital_twin", "../../"),
            "resources",
            "urdf",
            "pr2_kinematic_tree.urdf",
        )
    ).parse()

    def wait_for_sync(timeout=3.0, interval=0.05):
        start = time.time()
        while time.time() - start < timeout:
            body_ids_1 = [body.id for body in w1.kinematic_structure_entities]
            body_ids_2 = [body.id for body in w2.kinematic_structure_entities]
            if body_ids_1 == body_ids_2:
                return body_ids_1, body_ids_2
            time.sleep(interval)

        body_ids_1 = [body.id for body in w1.kinematic_structure_entities]
        body_ids_2 = [body.id for body in w2.kinematic_structure_entities]
        return body_ids_1, body_ids_2

    with w1.modify_world():
        new_body = Body(name=PrefixedName("b3"))
        w1.add_kinematic_structure_entity(new_body)

    fixed_connection = FixedConnection(child=new_body, parent=pr2_world.root)
    w1.merge_world(pr2_world, fixed_connection)

    body_ids_1, body_ids_2 = wait_for_sync()

    assert body_ids_1 == body_ids_2
    assert len(w1.kinematic_structure_entities) == len(w2.kinematic_structure_entities)

    w1_connection_hashes = [hash(c) for c in w1.connections]
    w2_connection_hashes = [hash(c) for c in w2.connections]
    assert w1_connection_hashes == w2_connection_hashes
    assert len(w1.connections) == len(w2.connections)
    assert len(w2.degrees_of_freedom) == len(w1.degrees_of_freedom)

    synchronizer_1.close()
    synchronizer_2.close()


def test_callback_pausing(rclpy_node):

    w1 = World(name="w1")
    w2 = World(name="w2")

    model_synchronizer_1 = ModelSynchronizer(node=rclpy_node, _world=w1)
    model_synchronizer_2 = ModelSynchronizer(node=rclpy_node, _world=w2)
    state_synchronizer_1 = StateSynchronizer(node=rclpy_node, _world=w1)
    state_synchronizer_2 = StateSynchronizer(node=rclpy_node, _world=w2)

    model_synchronizer_2.pause()
    state_synchronizer_2.pause()
    assert model_synchronizer_2._is_paused
    assert state_synchronizer_2._is_paused

    with w1.modify_world():
        b2 = Body(name=PrefixedName("b2"))
        w1.add_kinematic_structure_entity(b2)

        new_body = Body(name=PrefixedName("b3"))
        w1.add_kinematic_structure_entity(new_body)

        c = Connection6DoF.create_with_dofs(parent=b2, child=new_body, world=w1)
        w1.add_connection(c)

    time.sleep(0.2)
    assert len(model_synchronizer_2.missed_messages) == 1
    assert len(w1.kinematic_structure_entities) == 2
    assert len(w2.kinematic_structure_entities) == 0
    assert len(w1.connections) == 1
    assert len(w2.connections) == 0

    model_synchronizer_2.resume()
    state_synchronizer_2.resume()
    model_synchronizer_2.apply_missed_messages()
    state_synchronizer_2.apply_missed_messages()

    time.sleep(0.2)
    assert len(w1.kinematic_structure_entities) == 2
    assert len(w2.kinematic_structure_entities) == 2
    assert len(w1.connections) == 1
    assert len(w2.connections) == 1


def test_ChangeDifHasHardwareInterface(rclpy_node):

    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    with w1.modify_world():
        body1 = Body(name=PrefixedName("b1"))
        body2 = Body(name=PrefixedName("b2"))
        w1.add_kinematic_structure_entity(body1)
        w1.add_kinematic_structure_entity(body2)
        dof = DegreeOfFreedom(name=PrefixedName("dof"))
        w1.add_degree_of_freedom(dof)
        connection = PrismaticConnection(
            dof_id=dof.id, parent=body1, child=body2, axis=Vector3(1, 1, 1)
        )
        w1.add_connection(connection)
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

    synchronizer_1.close()
    synchronizer_2.close()


def test_semantic_annotation_modifications(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
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

    time.sleep(0.2)
    assert [hash(sa) for sa in w1.semantic_annotations] == [
        hash(sa) for sa in w2.semantic_annotations
    ]


def test_synchronize_6dof(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
        node=rclpy_node,
        _world=w2,
    )
    state_synch = StateSynchronizer(_world=w1, node=rclpy_node)
    state_synch2 = StateSynchronizer(_world=w2, node=rclpy_node)

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
    assert w1.state[c1.qw_id].position == w2.state[c2.qw_id].position
    np.testing.assert_array_almost_equal(w1.state.data, w2.state.data)


def test_synchronous_state_synchronization(rclpy_node):
    """When synchronous=True the notify_state_change call blocks until
    all subscribers have acknowledged receipt, so the remote world is
    already up-to-date when the call returns."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    receiver_node = rclpy.create_node("test_sync_state_receiver")
    receiver_executor = SingleThreadedExecutor()
    receiver_executor.add_node(receiver_node)
    receiver_thread = threading.Thread(
        target=receiver_executor.spin, daemon=True, name="sync-state-receiver"
    )
    receiver_thread.start()
    time.sleep(0.1)

    try:
        w1 = create_dummy_world()
        w2 = create_dummy_world()

        synchronizer_1 = StateSynchronizer(
            node=rclpy_node,
            _world=w1,
            synchronous=True,
        )
        synchronizer_2 = StateSynchronizer(
            node=receiver_node,
            _world=w2,
        )

        # Allow time for publishers/subscribers to discover each other
        time.sleep(0.2)

        w1.state.data[0, 0] = 1.0
        w1.notify_state_change()

        # With synchronous publishing the state must already be propagated
        # by the time notify_state_change returns.
        assert w1.state.data[0, 0] == w2.state.data[0, 0]

        synchronizer_1.close()
        synchronizer_2.close()
    finally:
        receiver_executor.shutdown()
        receiver_thread.join(timeout=2.0)
        receiver_node.destroy_node()


def test_synchronous_model_synchronization(rclpy_node):
    """When synchronous=True the modify_world call blocks until all subscribers
    acknowledge receipt, so the remote world is already up-to-date when the call
    returns."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    receiver_node = rclpy.create_node("test_sync_model_receiver")
    receiver_executor = SingleThreadedExecutor()
    receiver_executor.add_node(receiver_node)
    receiver_thread = threading.Thread(
        target=receiver_executor.spin, daemon=True, name="sync-model-receiver"
    )
    receiver_thread.start()
    time.sleep(0.1)

    try:
        w1 = World(name="w1")
        w2 = World(name="w2")

        synchronizer_1 = ModelSynchronizer(
            node=rclpy_node,
            _world=w1,
            synchronous=True,
        )
        synchronizer_2 = ModelSynchronizer(
            node=receiver_node,
            _world=w2,
        )

        # Allow time for publishers/subscribers to discover each other
        time.sleep(0.5)

        with w1.modify_world():
            new_body = Body(name=PrefixedName("b3"))
            b3_id = new_body.id
            w1.add_kinematic_structure_entity(new_body)

        # With synchronous publishing the model must already be propagated
        # by the time modify_world returns.
        assert len(w2.kinematic_structure_entities) == 1
        assert w2.get_kinematic_structure_entity_by_id(b3_id)

        synchronizer_1.close()
        synchronizer_2.close()
    finally:
        receiver_executor.shutdown()
        receiver_thread.join(timeout=2.0)
        receiver_node.destroy_node()


def test_synchronous_publish_blocks_until_receiver_acknowledges(rclpy_node):
    """Test whether synchronous publication genuinely blocks the caller until
    the remote subscriber acknowledges, rather than succeeding by coincidence.

    Uses a second ROS node (distinct ``node_name``) so the acknowledgment protocol can
    distinguish sender from receiver.  The receiver's acknowledgment publisher
    is intercepted so that acknowledgments are captured but not sent.  We then verify
    that the sender thread stays blocked, release the captured acknowledgments, and
    confirm that the sender unblocks.
    """
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    receiver_node = rclpy.create_node("test_receiver_node")
    receiver_executor = SingleThreadedExecutor()
    receiver_executor.add_node(receiver_node)
    receiver_thread = threading.Thread(
        target=receiver_executor.spin, daemon=True, name="receiver-executor"
    )
    receiver_thread.start()
    time.sleep(0.1)

    try:
        w1 = create_dummy_world()
        w2 = create_dummy_world()

        synchronizer_1 = StateSynchronizer(node=rclpy_node, _world=w1, synchronous=True)
        synchronizer_2 = StateSynchronizer(node=receiver_node, _world=w2)

        # Allow time for publishers/subscribers to discover each other
        time.sleep(0.2)

        # Intercept the receiver's acknowledgment publisher: capture outgoing
        # acknowledgments without actually publishing them so the sender never
        # gets an acknowledgment from the receiver node.
        real_acknowledgment_publisher = synchronizer_2.acknowledge_publisher
        captured_acknowledgments = []

        class _AcknowledgmentInterceptor:
            """Drop-in replacement that records but does not send acknowledgments."""

            def publish(self, msg):
                captured_acknowledgments.append(msg)

        synchronizer_2.acknowledge_publisher = _AcknowledgmentInterceptor()

        # Trigger a synchronous state change in a background thread. It
        # should block because the receiver's acknowledgment will never arrive.
        w1.state.data[0, 0] = 1.0
        publish_done = threading.Event()

        def do_publish():
            w1.notify_state_change()
            publish_done.set()

        thread = threading.Thread(target=do_publish, daemon=True)
        thread.start()

        # Give the executor enough time to deliver the message and process
        # the sender's self-acknowledgment.  The sender must still be blocked
        # because the receiver's acknowledgment was intercepted.
        time.sleep(0.5)
        assert (
            not publish_done.is_set()
        ), "Synchronous publish must block until the receiver acknowledges"

        # Now release the captured acknowledgments via the real publisher.
        for msg in captured_acknowledgments:
            real_acknowledgment_publisher.publish(msg)

        # The sender should unblock promptly.
        thread.join(timeout=5)
        assert (
            publish_done.is_set()
        ), "Synchronous publish must unblock after the receiver acknowledges"

        # The state should also be propagated because the receiver's
        # subscription callback still applied the message (only the
        # acknowledgment was intercepted, not message processing).
        assert w1.state.data[0, 0] == w2.state.data[0, 0]

        synchronizer_2.acknowledge_publisher = real_acknowledgment_publisher
        synchronizer_1.close()
        synchronizer_2.close()
    finally:
        receiver_executor.shutdown()
        receiver_thread.join(timeout=2.0)
        receiver_node.destroy_node()


def test_compute_state_changes_no_changes(rclpy_node):
    w = create_dummy_world()
    s = StateSynchronizer(node=rclpy_node, _world=w)
    # Immediately compare without changing state
    changes = s.compute_state_changes()
    assert changes == {}
    s.close()


def test_compute_state_changes_single_change(rclpy_node):
    w = create_dummy_world()
    s = StateSynchronizer(node=rclpy_node, _world=w)
    # change first position
    w.state.data[0, 0] += 1e-3
    changes = s.compute_state_changes()
    names = w.state.keys()
    assert list(changes.keys()) == [names[0]]
    assert np.isclose(changes[names[0]], w.state.positions[0])
    s.close()


def test_compute_state_changes_shape_change_full_snapshot(rclpy_node):
    w = create_dummy_world()
    s = StateSynchronizer(node=rclpy_node, _world=w)
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
    s = StateSynchronizer(node=rclpy_node, _world=w)
    # set both previous and current to NaN for entry 0
    w.state.data[0, 0] = np.nan
    s.previous_world_state_data[0] = np.nan
    assert s.compute_state_changes() == {}
    s.close()


def test_attribute_updates(rclpy_node):
    world1 = World(name="w1")
    world2 = World(name="w2")
    world1._id = uuid.UUID(int=1)
    world2._id = uuid.UUID(int=2)

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        _world=world1,
    )
    synchronizer_2 = ModelSynchronizer(
        node=rclpy_node,
        _world=world2,
    )

    root = Body(name=PrefixedName("root"))
    with world1.modify_world():
        world1.add_body(root)
    time.sleep(1)
    with world1.modify_world():
        fridge = Fridge.create_with_new_body_in_world(
            name=PrefixedName("case"),
            world=world1,
            scale=Scale(1, 1, 2.0),
        )
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("left_door"),
            world=world1,
        )
    time.sleep(1)
    assert [hash(sa) for sa in world1.semantic_annotations] == [
        hash(sa) for sa in world2.semantic_annotations
    ], f"{[sa.name for sa in world1.semantic_annotations]} vs {[sa.name for sa in world2.semantic_annotations]}"

    print(f"{door.id=}")
    with world1.modify_world():
        fridge.add_door(door)

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
    sync1 = ModelSynchronizer(node=rclpy_node, _world=w1)
    sync2 = ModelSynchronizer(node=rclpy_node, _world=w2)

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
        updated_kwargs=[
            JSONAttributeDiff(attribute_name="value", added_values=["direct_value"])
        ],
    )
    mod.apply(w)
    assert anno.value == "direct_value"

    # Test entity reference update
    mod = AttributeUpdateModification(
        entity_id=anno.id,
        updated_kwargs=[
            JSONAttributeDiff(attribute_name="entity", added_values=[b1.id])
        ],
    )
    mod.apply(w)
    assert anno.entity == b1

    # Test list update (add)
    mod = AttributeUpdateModification(
        entity_id=anno.id,
        updated_kwargs=[
            JSONAttributeDiff(attribute_name="entities", added_values=[b1.id])
        ],
    )
    mod.apply(w)
    assert b1 in anno.entities

    # Test list update (remove)
    mod = AttributeUpdateModification(
        entity_id=anno.id,
        updated_kwargs=[
            JSONAttributeDiff(attribute_name="entities", removed_values=[b1.id])
        ],
    )
    mod.apply(w)
    assert b1 not in anno.entities


def test_skipping_incorrect_message(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    with w1.modify_world():
        new_body = Body(name=PrefixedName("b3"))
        w1.add_kinematic_structure_entity(new_body)

    time.sleep(0.2)

    assert len(w1.kinematic_structure_entities) == len(w2.kinematic_structure_entities)

    with w1.modify_world():
        synchronizer_1.apply_missed_messages()
        handle = Handle.create_with_new_body_in_world(PrefixedName("handle"), w1)

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

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
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
            Handle.create_with_new_body_in_world(PrefixedName("handle"), w1)

        # Nested w2 context
        with w2.modify_world():
            for _ in range(in_w2):
                Handle.create_with_new_body_in_world(PrefixedName("handle2"), w2)

        # Create handles after nested context
        for _ in range(after_w2):
            Handle.create_with_new_body_in_world(PrefixedName("handle"), w1)

    w1_ids, w2_ids = wait_for_sync_kse_and_return_ids(w1, w2)
    assert len(w1.kinematic_structure_entities) == len(w2.kinematic_structure_entities)
    assert w1_ids == w2_ids

    synchronizer_1.close()
    synchronizer_2.close()


def test_nested_modify_world_publish_changes_true_false(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
        node=rclpy_node,
        _world=w2,
    )

    with w1.modify_world():
        new_body = Body(name=PrefixedName("b3"))
        w1.add_kinematic_structure_entity(new_body)

    time.sleep(0.2)

    assert len(w1.kinematic_structure_entities) == len(w2.kinematic_structure_entities)

    with pytest.raises(MismatchingPublishChangesAttribute):
        with w1.modify_world():
            handle = Handle.create_with_new_body_in_world(PrefixedName("handle"), w1)

            with w1.modify_world(publish_changes=False):
                handle = Handle.create_with_new_body_in_world(
                    PrefixedName("handle"), w1
                )

    with pytest.raises(MismatchingPublishChangesAttribute):
        with w1.modify_world(publish_changes=False):
            handle = Handle.create_with_new_body_in_world(PrefixedName("handle"), w1)

            with w1.modify_world(publish_changes=True):
                handle = Handle.create_with_new_body_in_world(
                    PrefixedName("handle"), w1
                )

    synchronizer_1.close()
    synchronizer_2.close()


def test_dont_publish_changes(rclpy_node):
    w1 = World(name="w1")
    w2 = World(name="w2")

    synchronizer_1 = ModelSynchronizer(
        node=rclpy_node,
        _world=w1,
    )
    synchronizer_2 = ModelSynchronizer(
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
    )

    serialized = to_json(original)
    restored = from_json(serialized)

    assert isinstance(restored, WorldStateUpdate)
    assert restored.meta_data.node_name == original.meta_data.node_name
    assert restored.meta_data.process_id == original.meta_data.process_id
    assert restored.ids == original.ids
    assert restored.states == original.states
    assert restored.publication_event_id == original.publication_event_id


def test_load_model_serialization_round_trip():
    """
    Verify that LoadModel survives a to_json/from_json round trip.
    """
    meta = MetaData(node_name="loader", process_id=99)
    original = LoadModel(meta_data=meta, primary_key=7)

    serialized = to_json(original)
    restored = from_json(serialized)

    assert isinstance(restored, LoadModel)
    assert restored.primary_key == 7
    assert restored.meta_data.node_name == "loader"
    assert restored.publication_event_id == original.publication_event_id


def test_acknowledgment_serialization_round_trip():
    """
    Verify that Acknowledgment survives a to_json/from_json round trip.
    """
    event_id = uuid.uuid4()
    meta = MetaData(node_name="acknowledgment_node", process_id=1)
    original = Acknowledgment(publication_event_id=event_id, node_meta_data=meta)

    serialized = to_json(original)
    restored = from_json(serialized)

    assert isinstance(restored, Acknowledgment)
    assert restored.publication_event_id == event_id
    assert restored.node_meta_data.node_name == "acknowledgment_node"
    assert restored.node_meta_data.process_id == 1


def test_acknowledgement_with_missed_messages(rclpy_node):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    receiver_node = rclpy.create_node("test_sync_state_receiver")
    receiver_executor = SingleThreadedExecutor()
    receiver_executor.add_node(receiver_node)
    receiver_thread = threading.Thread(
        target=receiver_executor.spin, daemon=True, name="sync-state-receiver"
    )
    receiver_thread.start()
    time.sleep(0.1)

    try:
        w1 = create_dummy_world()
        w2 = create_dummy_world()

        synchronizer_1 = StateSynchronizer(
            node=rclpy_node,
            _world=w1,
            synchronous=True,
        )
        synchronizer_2 = StateSynchronizer(
            node=receiver_node,
            _world=w2,
        )
        synchronizer_2.pause()

        # Allow time for publishers/subscribers to discover each other
        time.sleep(0.5)

        w1.state.data[0, 0] = 1.0
        w1.notify_state_change()

        # the notify should time out giving us the old state
        assert w2.state.data[0, 0] == 0
        synchronizer_2.apply_missed_messages()
        # after apply message we should have the correct state
        assert w1.state.data[0, 0] == w2.state.data[0, 0]

        synchronizer_1.close()
        synchronizer_2.close()
    finally:
        receiver_executor.shutdown()
        receiver_thread.join(timeout=2.0)
        receiver_node.destroy_node()


if __name__ == "__main__":
    unittest.main()
