import unittest
from copy import deepcopy

import numpy as np
import pytest

from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Door,
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
    AddConnectionModification,
    AddDegreeOfFreedomModification,
    AddSemanticAnnotationModification,
    RemoveSemanticAnnotationModification,
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
    """world.py:283-289: atomic_world_modification appends the modification to the
    current block *before* executing the function. If the function raises and the
    caller catches the error inside the modify_world block, a phantom modification
    stays in the history."""

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


if __name__ == "__main__":
    unittest.main()
