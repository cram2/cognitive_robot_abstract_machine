import unittest
from copy import deepcopy

import numpy as np
import pytest

from krrood.adapters.json_serializer import from_json, to_json, shallow_diff_json
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    InsufficientModificationHistoryError,
    InvalidRollbackVersionError,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Door,
    Oven,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    Vector3,
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    Connection6DoF,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Actuator
from semantic_digital_twin.world_description.world_modification import (
    WorldModelModificationBlock,
    AddKinematicStructureEntityModification,
    RemoveKinematicStructureEntityModification,
    AddConnectionModification,
    RemoveConnectionModification,
    AddDegreeOfFreedomModification,
    RemoveDegreeOfFreedomModification,
    AddSemanticAnnotationModification,
    RemoveSemanticAnnotationModification,
    RemoveActuatorModification,
    SetDofHasHardwareInterface,
    AttributeUpdateModification,
)


class ConnectionModificationTestCase(unittest.TestCase):

    def test_single_modification(self):
        w = World()

        with w.modify_world():
            b1 = Body(name=PrefixedName("b1"))
            b2 = Body(name=PrefixedName("b2"))
            w.add_kinematic_structure_entity(b1)
            w.add_kinematic_structure_entity(b2)

            connection = FixedConnection(b1, b2)
            w.add_connection(connection)

    def test_ChangeDofHasHardwareInterface(self):
        w = World()

        with w.modify_world():
            b1 = Body(name=PrefixedName("b1"))
            b2 = Body(name=PrefixedName("b2"))
            w.add_kinematic_structure_entity(b1)
            w.add_kinematic_structure_entity(b2)

            dof = DegreeOfFreedom(name=PrefixedName("dofyboi"))
            w.add_degree_of_freedom(dof)
            connection = RevoluteConnection(
                b1, b2, axis=Vector3.from_iterable([0, 0, 1]), raw_dof=dof
            )
            w.add_connection(connection)
        assert connection.dof.has_hardware_interface is False

        with w.modify_world():
            w.set_dofs_has_hardware_interface(connection.dofs, True)
        assert connection.dof.has_hardware_interface is True

    def test_many_modifications(self):
        world = World()

        with world.modify_world():
            b1 = Body(name=PrefixedName("b1"))
            b2 = Body(name=PrefixedName("b2"))
            b3 = Body(name=PrefixedName("b3"))
            world.add_kinematic_structure_entity(b1)
            world.add_kinematic_structure_entity(b2)
            world.add_kinematic_structure_entity(b3)
            world.add_connection(
                Connection6DoF.create_with_dofs(parent=b1, child=b2, world=world)
            )
            dof = DegreeOfFreedom(name=PrefixedName("dofyboi"))
            world.add_degree_of_freedom(dof)
            world.add_connection(
                PrismaticConnection(
                    parent=b2,
                    child=b3,
                    axis=Vector3.from_iterable([0, 0, 1]),
                    raw_dof=dof,
                )
            )

        modifications = world.get_world_model_manager().model_modification_blocks[-1]
        self.assertEqual(len(modifications.modifications), 13)

        add_body_modifications = [
            m
            for m in modifications.modifications
            if isinstance(m, AddKinematicStructureEntityModification)
        ]
        self.assertEqual(len(add_body_modifications), 3)

        add_dof_modifications = [
            m
            for m in modifications.modifications
            if isinstance(m, AddDegreeOfFreedomModification)
        ]
        self.assertEqual(len(add_dof_modifications), 8)

        add_connection_modifications = [
            m
            for m in modifications.modifications
            if isinstance(m, AddConnectionModification)
        ]
        self.assertEqual(len(add_connection_modifications), 2)

        # reconstruct this world
        w2 = World()

        tracker = WorldEntityWithIDKwargsTracker()
        kwargs = tracker.create_kwargs()
        # copy modifications
        modifications_copy = from_json(to_json(modifications), **kwargs)
        with w2.modify_world():
            modifications_copy.apply(w2)
        self.assertEqual(len(w2.bodies), 3)
        self.assertEqual(len(w2.connections), 2)

        with world.modify_world():
            world.remove_connection(world.connections[-1])
            world.remove_kinematic_structure_entity(
                world.get_kinematic_structure_entity_by_name("b3")
            )

        modifications = world.get_world_model_manager().model_modification_blocks[-1]
        self.assertEqual(len(modifications.modifications), 3)

        tracker = WorldEntityWithIDKwargsTracker.from_world(w2)
        kwargs = tracker.create_kwargs()

        modifications_copy = from_json(to_json(modifications), **kwargs)
        with w2.modify_world():
            modifications_copy.apply(w2)
        self.assertEqual(len(w2.bodies), 2)
        self.assertEqual(len(w2.connections), 1)

    def test_semantic_annotation_modifications(self):
        w = World()
        with w.modify_world():
            w.add_kinematic_structure_entity(b1 := Body(name=PrefixedName("b1")))
        v1 = Handle(root=b1)
        v2 = Door(root=b1, handle=v1)

        add_v1 = AddSemanticAnnotationModification.from_domain_object(v1)
        add_v2 = AddSemanticAnnotationModification.from_domain_object(v2)

        self.assertNotEqual({v1.id, v2.id}, set(a.id for a in w.semantic_annotations))

        with w.modify_world():
            add_v1.apply(w)
            add_v2.apply(w)

        self.assertEqual({v1.id, v2.id}, set(a.id for a in w.semantic_annotations))
        self.assertEqual(v1.root, w.get_kinematic_structure_entity_by_name("b1"))

        rm_v1 = RemoveSemanticAnnotationModification(v1.id)
        rm_v2 = RemoveSemanticAnnotationModification(v2.id)
        with w.modify_world():
            rm_v1.apply(w)
            rm_v2.apply(w)

        self.assertNotEqual({v1.id, v2.id}, set(a.id for a in w.semantic_annotations))

    def test_duplicate_name_modification_serialization(self):
        w = World()
        with w.modify_world():
            b1 = Body(name=PrefixedName("b1"))
            w.add_kinematic_structure_entity(b1)

            b2 = Body(name=PrefixedName("b1"))  # Duplicate name
            w.add_kinematic_structure_entity(b2)

            b3 = Body(name=PrefixedName("b3"))
            w.add_kinematic_structure_entity(b3)

            c1 = Connection6DoF.create_with_dofs(
                name=PrefixedName("name1"), parent=b1, child=b2, world=w
            )
            w.add_connection(c1)
            c2 = Connection6DoF.create_with_dofs(
                name=PrefixedName("name1"), parent=b1, child=b3, world=w
            )
            w.add_connection(c2)

        modifications = w.get_world_model_manager().model_modification_blocks[-1]
        tracker = WorldEntityWithIDKwargsTracker()
        kwargs = tracker.create_kwargs()

        modifications_copy = from_json(to_json(modifications), **kwargs)

        w2 = World()
        with w2.modify_world():
            modifications_copy.apply(w2)
        self.assertEqual(len(w2.bodies), 3)
        self.assertEqual(len(w2.connections), 2)

    def test_actuator_serialization(self):
        w = World()
        with w.modify_world():
            b1 = Body(name=PrefixedName("b1"))
            w.add_kinematic_structure_entity(b1)
            b2 = Body(name=PrefixedName("b2"))
            w.add_kinematic_structure_entity(b2)
            c = Connection6DoF.create_with_dofs(parent=b1, child=b2, world=w)
            w.add_connection(c)
            dof = w.degrees_of_freedom[0]
            actuator = Actuator(name=PrefixedName("actuator"))
            actuator.add_dof(dof)
            w.add_actuator(actuator)

        modifications = w.get_world_model_manager().model_modification_blocks[-1]
        tracker = WorldEntityWithIDKwargsTracker()
        kwargs = tracker.create_kwargs()

        modifications_copy = from_json(to_json(modifications), **kwargs)
        w2 = World()
        with w2.modify_world():
            modifications_copy.apply(w2)

        self.assertEqual(len(w2.actuators), 1)
        self.assertEqual(w2.actuators[0].id, actuator.id)


def test_connection_t_child_expression_survives_json_roundtrip():

    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    child = Body(name=PrefixedName("child", prefix="review"))
    connection_T_child = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=2.0, z=3.0, child_frame=child
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(child)
        connection = FixedConnection(
            parent=root,
            child=child,
            connection_T_child_expression=connection_T_child,
        )
        world.add_connection(connection)

    tracker_kwargs = WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs()
    restored = from_json(connection.to_json(), **tracker_kwargs)

    np.testing.assert_allclose(
        restored.connection_T_child_expression.to_np(),
        connection.connection_T_child_expression.to_np(),
    )


def test_body_inertial_survives_world_deepcopy():
    world = World()
    body = Body(name=PrefixedName("heavy_body", prefix="review"))
    collision = Box(
        scale=Scale(),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
    )
    body.collision = ShapeCollection([collision], reference_frame=body)
    body.inertial = Inertial(mass=5.0)
    with world.modify_world():
        world.add_kinematic_structure_entity(body)

    copied_world = deepcopy(world)
    copied_body = copied_world.get_body_by_name("heavy_body")

    assert copied_body.inertial is not None
    assert copied_body.inertial.mass == pytest.approx(5.0)


def test_design_09_failed_atomic_modification_is_not_recorded():
    """
    world.py:283-289: atomic_world_modification appends the modification to the current
    block *before* executing the function.

    If the function raises and the caller catches the error inside the modify_world
    block, a phantom modification stays in the history.
    """
    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    child = Body(name=PrefixedName("child", prefix="review"))
    collision = Box(
        scale=Scale(),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=child),
    )
    child.collision = ShapeCollection([collision], reference_frame=child)
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(child)
        connection = FixedConnection(parent=root, child=child)
        world.add_connection(connection)

    outside_parent = Body(name=PrefixedName("outside_parent", prefix="review"))
    outside_child = Body(name=PrefixedName("outside_child", prefix="review"))

    with world.modify_world():
        bad_connection = FixedConnection(parent=outside_parent, child=outside_child)
        try:
            # bodies were never added to the world -> their graph indices are None
            world._add_connection(bad_connection)
        except Exception:
            pass

        block = world._model_manager.current_model_modification_block
        assert len(block) == 0, (
            "a failed atomic modification was recorded in the history: "
            f"{block.modifications}"
        )


# %% revert and rollback


def _last_modification_block(world: World) -> WorldModelModificationBlock:
    return world.get_world_model_manager().model_modification_blocks[-1]


def test_revert_add_kinematic_structure_entity():
    world = World()
    body = Body(name=PrefixedName("body"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body)

    modification = _last_modification_block(world)[0]
    assert isinstance(modification, AddKinematicStructureEntityModification)

    with world.modify_world():
        modification.revert(world)

    assert not world.is_kinematic_structure_entity_in_world(body)


def test_revert_remove_kinematic_structure_entity():
    world = World()
    body = Body(name=PrefixedName("body"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
    with world.modify_world():
        world.remove_kinematic_structure_entity(body)

    modification = _last_modification_block(world)[0]
    assert isinstance(modification, RemoveKinematicStructureEntityModification)

    with world.modify_world():
        modification.revert(world)

    assert world.is_kinematic_structure_entity_in_world(body)


def test_revert_add_connection():
    # world.is_connection_in_world() is not used here: Connection.add_to_world()
    # never registers connections in the world's entity-hash table, so that check is
    # always False regardless of revert. Membership in world.connections is the
    # meaningful check, matching how the rest of this file verifies connections.
    world = World()
    with world.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        b2 = Body(name=PrefixedName("b2"))
        world.add_kinematic_structure_entity(b1)
        world.add_kinematic_structure_entity(b2)
        connection = FixedConnection(b1, b2)
        world.add_connection(connection)

    modification = next(
        m
        for m in _last_modification_block(world)
        if isinstance(m, AddConnectionModification)
    )

    with world.modify_world():
        modification.revert(world)
        # reverting the connection alone leaves b2 disconnected from the world's single
        # root, which a modify_world block may not exit with; remove it too.
        world.remove_kinematic_structure_entity(b2)

    assert connection not in world.connections
    assert not world.is_kinematic_structure_entity_in_world(b2)


def test_revert_remove_connection():
    world = World()
    with world.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        b2 = Body(name=PrefixedName("b2"))
        world.add_kinematic_structure_entity(b1)
        world.add_kinematic_structure_entity(b2)
        connection = FixedConnection(b1, b2)
        world.add_connection(connection)
    with world.modify_world():
        # removing the connection alone would leave b2 disconnected from the world's
        # single root, which a modify_world block may not exit with; remove it too.
        world.remove_connection(connection)
        world.remove_kinematic_structure_entity(b2)

    modification = next(
        m
        for m in _last_modification_block(world)
        if isinstance(m, RemoveConnectionModification)
    )

    with world.modify_world():
        modification.revert(world)

    assert connection in world.connections
    assert world.is_kinematic_structure_entity_in_world(b2)


def test_revert_add_degree_of_freedom():
    # A degree of freedom not used by any connection is deleted automatically as an
    # orphan when the modify_world block that adds it closes (World.delete_orphaned_dofs),
    # so it has to be attached to a connection to observe revert() removing it deliberately.
    world = World()
    with world.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        b2 = Body(name=PrefixedName("b2"))
        world.add_kinematic_structure_entity(b1)
        world.add_kinematic_structure_entity(b2)
        dof = DegreeOfFreedom(name=PrefixedName("dof"))
        world.add_degree_of_freedom(dof)
        connection = RevoluteConnection(
            b1, b2, axis=Vector3.from_iterable([0, 0, 1]), raw_dof=dof
        )
        world.add_connection(connection)

    modification = next(
        m
        for m in _last_modification_block(world)
        if isinstance(m, AddDegreeOfFreedomModification)
    )

    with world.modify_world():
        # the connection using this dof has to go too, otherwise forward kinematics
        # compilation fails with a dangling reference to the removed dof.
        world.remove_connection(connection)
        world.remove_kinematic_structure_entity(b2)
        modification.revert(world)

    assert not world.is_degree_of_freedom_in_world(dof)


def test_revert_remove_degree_of_freedom():
    world = World()
    with world.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        b2 = Body(name=PrefixedName("b2"))
        world.add_kinematic_structure_entity(b1)
        world.add_kinematic_structure_entity(b2)
        dof = DegreeOfFreedom(name=PrefixedName("dof"))
        world.add_degree_of_freedom(dof)
        connection = RevoluteConnection(
            b1, b2, axis=Vector3.from_iterable([0, 0, 1]), raw_dof=dof
        )
        world.add_connection(connection)
    with world.modify_world():
        # removing the connection first makes the explicit dof removal legal (a dof
        # still used by a connection cannot safely be removed on its own).
        world.remove_connection(connection)
        world.remove_degree_of_freedom(dof)
        world.remove_kinematic_structure_entity(b2)

    modification = next(
        m
        for m in _last_modification_block(world)
        if isinstance(m, RemoveDegreeOfFreedomModification)
    )

    with world.modify_world():
        modification.revert(world)
        # re-attach the dof so the block does not exit with it orphaned again.
        world.add_kinematic_structure_entity(b2)
        world.add_connection(connection)

    assert world.is_degree_of_freedom_in_world(dof)


def test_revert_add_semantic_annotation():
    world = World()
    with world.modify_world():
        body = Body(name=PrefixedName("body"))
        world.add_kinematic_structure_entity(body)
    handle = Handle(root=body)

    add_handle = AddSemanticAnnotationModification.from_domain_object(handle)
    with world.modify_world():
        add_handle.apply(world)

    assert handle.id in {a.id for a in world.semantic_annotations}

    with world.modify_world():
        add_handle.revert(world)

    assert handle.id not in {a.id for a in world.semantic_annotations}


def test_revert_remove_semantic_annotation():
    world = World()
    with world.modify_world():
        body = Body(name=PrefixedName("body"))
        world.add_kinematic_structure_entity(body)
    handle = Handle(root=body)
    with world.modify_world():
        world.add_semantic_annotation(handle)
    with world.modify_world():
        world.remove_semantic_annotation(handle)

    modification = _last_modification_block(world)[0]
    assert isinstance(modification, RemoveSemanticAnnotationModification)
    assert handle.id not in {a.id for a in world.semantic_annotations}

    with world.modify_world():
        modification.revert(world)

    assert handle.id in {a.id for a in world.semantic_annotations}


def test_revert_add_actuator():
    world = World()
    actuator = Actuator(name=PrefixedName("actuator"))
    with world.modify_world():
        world.add_actuator(actuator)

    modification = _last_modification_block(world)[0]
    with world.modify_world():
        modification.revert(world)

    assert actuator not in world.actuators


def test_revert_remove_actuator():
    world = World()
    actuator = Actuator(name=PrefixedName("actuator"))
    with world.modify_world():
        world.add_actuator(actuator)
    with world.modify_world():
        world.remove_actuator(actuator)

    modification = _last_modification_block(world)[0]
    assert isinstance(modification, RemoveActuatorModification)

    with world.modify_world():
        modification.revert(world)

    assert actuator in world.actuators


def test_revert_set_dofs_has_hardware_interface_restores_per_dof_previous_value():
    # dof1 and dof2 are each attached to a connection so neither is deleted as an
    # orphan when the modify_world block that adds it closes.
    world = World()
    with world.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        b2 = Body(name=PrefixedName("b2"))
        b3 = Body(name=PrefixedName("b3"))
        world.add_kinematic_structure_entity(b1)
        world.add_kinematic_structure_entity(b2)
        world.add_kinematic_structure_entity(b3)
        dof1 = DegreeOfFreedom(name=PrefixedName("dof1"))
        dof2 = DegreeOfFreedom(name=PrefixedName("dof2"))
        world.add_degree_of_freedom(dof1)
        world.add_degree_of_freedom(dof2)
        world.add_connection(
            RevoluteConnection(
                b1, b2, axis=Vector3.from_iterable([0, 0, 1]), raw_dof=dof1
            )
        )
        world.add_connection(
            RevoluteConnection(
                b2, b3, axis=Vector3.from_iterable([0, 0, 1]), raw_dof=dof2
            )
        )
    with world.modify_world():
        world.set_dofs_has_hardware_interface([dof1], True)

    assert dof1.has_hardware_interface is True
    assert dof2.has_hardware_interface is False

    with world.modify_world():
        world.set_dofs_has_hardware_interface([dof1, dof2], True)

    modification = _last_modification_block(world)[0]
    assert isinstance(modification, SetDofHasHardwareInterface)

    with world.modify_world():
        modification.revert(world)

    assert dof1.has_hardware_interface is True
    assert dof2.has_hardware_interface is False


def test_revert_attribute_update_scalar():
    world = World()
    body = Body(name=PrefixedName("body"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body)

    original_name = body.name
    with world.modify_world():
        body.update_name(PrefixedName("renamed"))

    assert body.name == PrefixedName("renamed")

    modification = _last_modification_block(world)[0]
    assert isinstance(modification, AttributeUpdateModification)

    with world.modify_world():
        modification.revert(world)

    assert body.name == original_name


def test_revert_attribute_update_list():
    world = World()
    with world.modify_world():
        oven_body = Body(name=PrefixedName("oven_body"))
        door_body = Body(name=PrefixedName("door_body"))
        world.add_kinematic_structure_entity(oven_body)
        world.add_kinematic_structure_entity(door_body)
        # a world must stay a single connected tree across modify_world blocks.
        world.add_connection(FixedConnection(oven_body, door_body))

    oven = Oven(root=oven_body)
    door = Door(root=door_body)
    with world.modify_world():
        world.add_semantic_annotation(oven)
        world.add_semantic_annotation(door)

    tracker_kwargs = WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs()
    before_json = to_json(oven)
    oven.doors.append(door)
    after_json = to_json(oven)
    diff = shallow_diff_json(before_json, after_json, **tracker_kwargs)

    modification = AttributeUpdateModification(
        entity_id=oven.id, updated_kwargs_json_list=diff
    )
    assert door in oven.doors

    with world.modify_world():
        modification.revert(world)

    assert door not in oven.doors


def test_world_model_modification_block_revert_restores_removed_branch():
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        child = Body(name=PrefixedName("child"))
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(child)
        connection = FixedConnection(root, child)
        world.add_connection(connection)

    world.remove_branch_from_world(child)
    assert not world.is_kinematic_structure_entity_in_world(child)
    assert connection not in world.connections

    block = _last_modification_block(world)
    with world.modify_world():
        block.revert(world)

    assert world.is_kinematic_structure_entity_in_world(child)
    assert connection in world.connections


def test_world_rollback_modification_blocks_reverts_most_recent_block():
    # b2 is connected to b1 so the world stays a single connected tree at the end of
    # each modify_world block, as required.
    world = World()
    with world.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        world.add_kinematic_structure_entity(b1)
    with world.modify_world():
        b2 = Body(name=PrefixedName("b2"))
        world.add_kinematic_structure_entity(b2)
        world.add_connection(FixedConnection(b1, b2))

    rolled_back = world.rollback_modification_blocks()

    assert world.is_kinematic_structure_entity_in_world(b1)
    assert not world.is_kinematic_structure_entity_in_world(b2)
    assert len(rolled_back) == 1
    add_body_modification = next(
        m
        for m in rolled_back[0]
        if isinstance(m, AddKinematicStructureEntityModification)
    )
    assert add_body_modification.kinematic_structure_entity is b2


def test_world_rollback_modification_blocks_reverts_several_blocks_in_order():
    world = World()
    with world.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        world.add_kinematic_structure_entity(b1)
    with world.modify_world():
        b2 = Body(name=PrefixedName("b2"))
        world.add_kinematic_structure_entity(b2)
        world.add_connection(FixedConnection(b1, b2))

    world.rollback_modification_blocks(count=2)

    assert not world.is_kinematic_structure_entity_in_world(b1)
    assert not world.is_kinematic_structure_entity_in_world(b2)


def test_world_rollback_modification_blocks_raises_when_insufficient_history():
    world = World()
    with world.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        world.add_kinematic_structure_entity(b1)

    with pytest.raises(InsufficientModificationHistoryError):
        world.rollback_modification_blocks(count=2)

    assert world.is_kinematic_structure_entity_in_world(b1)


def test_world_rollback_to_version_reverts_blocks_after_it():
    world = World()
    with world.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        world.add_kinematic_structure_entity(b1)
    version_after_b1 = world.get_world_model_manager().version

    with world.modify_world():
        b2 = Body(name=PrefixedName("b2"))
        world.add_kinematic_structure_entity(b2)
        world.add_connection(FixedConnection(b1, b2))
    with world.modify_world():
        b3 = Body(name=PrefixedName("b3"))
        world.add_kinematic_structure_entity(b3)
        world.add_connection(FixedConnection(b2, b3))

    rolled_back = world.rollback_to_version(version_after_b1)

    assert world.is_kinematic_structure_entity_in_world(b1)
    assert not world.is_kinematic_structure_entity_in_world(b2)
    assert not world.is_kinematic_structure_entity_in_world(b3)
    assert len(rolled_back) == 2


def test_world_rollback_to_version_is_noop_at_current_version():
    world = World()
    with world.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        world.add_kinematic_structure_entity(b1)
    current_version = world.get_world_model_manager().version

    rolled_back = world.rollback_to_version(current_version)

    assert rolled_back == []
    assert world.is_kinematic_structure_entity_in_world(b1)


def test_world_rollback_to_version_raises_for_version_ahead_of_current():
    world = World()
    with world.modify_world():
        b1 = Body(name=PrefixedName("b1"))
        world.add_kinematic_structure_entity(b1)
    current_version = world.get_world_model_manager().version

    with pytest.raises(InvalidRollbackVersionError):
        world.rollback_to_version(current_version + 1)

    assert world.is_kinematic_structure_entity_in_world(b1)


def test_world_rollback_to_version_raises_for_negative_version():
    world = World()

    with pytest.raises(InvalidRollbackVersionError):
        world.rollback_to_version(-1)


if __name__ == "__main__":
    unittest.main()
