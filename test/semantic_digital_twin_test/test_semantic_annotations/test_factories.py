import unittest
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

from random_events.product_algebra import Event
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    CannotBeAPartOf,
    AmbiguousPart,
    MechanicalJointAlreadyMounted,
    UnknownPartWholeRelationshipField,
)
from semantic_digital_twin.exceptions import (
    InvalidPlaneDimensions,
    InvalidHingeActiveAxis,
    InvalidConnectionLimits,
    MissingSemanticAnnotationError,
    MismatchingWorld,
    MissingWorldModificationContextError,
)
from semantic_digital_twin.orm.ormatic_interface import *
from semantic_digital_twin.semantic_annotations.mixins import (
    PartWholeRelationship,
    HasRootBody,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasCaseAsRootBody,
    IsPartWholeRelationship
)
from krrood.patterns.field_metadata import FieldMetadata
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    DoubleDoor,
    Floor,
    Cup,
    Cabinet,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Door,
    Drawer,
    Wall,
    Hinge,
    Fridge,
    Slider,
    Aperture,
    MechanicalJoint,
    Table,
    Milk,
    Cereal,
    Microwave,
    Hood,
    Toaster,
    CoffeeMachine,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
)
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body


class TestFactories(unittest.TestCase):
    def test_handle_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            returned_handle = Handle.create_with_new_body_in_world(
                name=PrefixedName("handle"),
                scale=Scale(0.1, 0.2, 0.03),
                thickness=0.03,
                world=world,
            )
        semantic_handle_annotations = world.get_semantic_annotations_by_type(Handle)
        self.assertEqual(len(semantic_handle_annotations), 1)
        self.assertTrue(
            isinstance(
                semantic_handle_annotations[0].root.parent_connection, FixedConnection
            )
        )

        queried_handle: Handle = semantic_handle_annotations[0]
        self.assertEqual(returned_handle, queried_handle)
        self.assertEqual(
            world.root, queried_handle.root.parent_kinematic_structure_entity
        )

    def test_basic_has_body_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            returned_hinge = Hinge.create_with_new_body_in_world(
                name=PrefixedName("hinge"),
                world=world,
                active_axis=Vector3.Z(),
            )
            returned_slider = Slider.create_with_new_body_in_world(
                name=PrefixedName("slider"),
                world=world,
                active_axis=Vector3.X(),
            )
        semantic_hinge_annotations = world.get_semantic_annotations_by_type(Hinge)
        self.assertEqual(len(semantic_hinge_annotations), 1)

        queried_hinge: Hinge = semantic_hinge_annotations[0]
        self.assertEqual(returned_hinge, queried_hinge)
        self.assertEqual(
            world.root, queried_hinge.root.parent_kinematic_structure_entity
        )
        semantic_slider_annotations = world.get_semantic_annotations_by_type(Slider)
        self.assertEqual(len(semantic_slider_annotations), 1)
        queried_slider: Slider = semantic_slider_annotations[0]
        self.assertEqual(returned_slider, queried_slider)
        self.assertEqual(
            world.root, queried_slider.root.parent_kinematic_structure_entity
        )

    def test_door_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            returned_door = Door.create_with_new_body_in_world(
                name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world
            )
        semantic_door_annotations = world.get_semantic_annotations_by_type(Door)
        self.assertEqual(len(semantic_door_annotations), 1)
        self.assertTrue(
            isinstance(
                semantic_door_annotations[0].root.parent_connection, FixedConnection
            )
        )

        queried_door: Door = semantic_door_annotations[0]
        self.assertEqual(returned_door, queried_door)
        self.assertEqual(
            world.root, queried_door.root.parent_kinematic_structure_entity
        )

    def test_door_factory_invalid(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            with pytest.raises(InvalidPlaneDimensions):
                Door.create_with_new_body_in_world(
                    name=PrefixedName("door"),
                    scale=Scale(1, 1, 2),
                    world=world,
                )

            with pytest.raises(InvalidPlaneDimensions):
                Door.create_with_new_body_in_world(
                    name=PrefixedName("door"),
                    scale=Scale(1, 2, 1),
                    world=world,
                )

        with pytest.raises(MissingWorldModificationContextError):
            Door.create_with_new_body_in_world(
                name=PrefixedName("door"),
                world=world,
            )

    def test_has_hinge_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            door = Door.create_with_new_body_in_world(
                name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world
            )
            hinge = Hinge.create_with_new_body_in_world(
                name=PrefixedName("hinge"), world=world, active_axis=Vector3.Z()
            )
        assert len(world.kinematic_structure_entities) == 4
        assert isinstance(hinge.root.parent_connection, RevoluteConnection)
        assert root == hinge.root.parent_kinematic_structure_entity
        assert root == door.root.parent_kinematic_structure_entity
        with world.modify_world():
            door.add(hinge)
        assert isinstance(hinge.root.parent_connection, RevoluteConnection)
        assert door.root.parent_kinematic_structure_entity == hinge.root
        assert door.mechanical_joint == hinge

    def test_has_handle_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            door = Door.create_with_new_body_in_world(
                name=PrefixedName("door"),
                scale=Scale(0.03, 1, 2),
                world=world,
            )

            handle = Handle.create_with_new_body_in_world(
                name=PrefixedName("handle"),
                world=world,
            )
        assert len(world.kinematic_structure_entities) == 4

        assert root == handle.root.parent_kinematic_structure_entity
        with world.modify_world():
            door.add(handle)

        assert door.root == handle.root.parent_kinematic_structure_entity
        assert door.handle == handle

    def test_case_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            fridge = Fridge.create_with_new_body_in_world(
                name=PrefixedName("case"),
                world=world,
                scale=Scale(1, 1, 2.0),
            )

        assert isinstance(fridge, HasCaseAsRootBody)

        semantic_container_annotations = world.get_semantic_annotations_by_type(Fridge)
        self.assertEqual(len(semantic_container_annotations), 1)

        assert len(world.get_semantic_annotations_by_type(HasCaseAsRootBody)) == 1

    def test_drawer_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            drawer = Drawer.create_with_new_body_in_world(
                name=PrefixedName("drawer"),
                world=world,
                scale=Scale(0.2, 0.3, 0.2),
            )
        assert isinstance(drawer, HasCaseAsRootBody)
        semantic_drawer_annotations = world.get_semantic_annotations_by_type(Drawer)
        self.assertEqual(len(semantic_drawer_annotations), 1)

    def test_has_slider_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            drawer = Drawer.create_with_new_body_in_world(
                name=PrefixedName("drawer"),
                scale=Scale(0.2, 0.3, 0.2),
                world=world,
            )
            slider = Slider.create_with_new_body_in_world(
                name=PrefixedName("slider"), world=world, active_axis=Vector3.X()
            )
        assert len(world.kinematic_structure_entities) == 3
        with world.modify_world():
            drawer.add(slider)

        assert drawer.root.parent_kinematic_structure_entity == slider.root
        assert isinstance(slider.root.parent_connection, PrismaticConnection)
        assert drawer.mechanical_joint == slider

    def test_has_drawer_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            fridge = Fridge.create_with_new_body_in_world(
                name=PrefixedName("case"),
                world=world,
                scale=Scale(1, 1, 2.0),
            )
            drawer = Drawer.create_with_new_body_in_world(
                name=PrefixedName("drawer"), world=world
            )
            fridge.add(drawer)

        semantic_drawer_annotations = world.get_semantic_annotations_by_type(Drawer)
        self.assertEqual(len(semantic_drawer_annotations), 1)
        assert fridge.drawers[0] == drawer

    def test_has_doors_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            fridge = Fridge.create_with_new_body_in_world(
                name=PrefixedName("case"),
                world=world,
                scale=Scale(1, 1, 2.0),
            )
            door = Door.create_with_new_body_in_world(
                name=PrefixedName("left_door"),
                world=world,
            )
            fridge.add(door)

        semantic_door_annotations = world.get_semantic_annotations_by_type(Door)
        self.assertEqual(len(semantic_door_annotations), 1)
        assert fridge.doors[0] == door

    def test_floor_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            floor = Floor.create_with_new_body_in_world(
                name=PrefixedName("floor"),
                world=world,
                scale=Scale(5, 5, 0.01),
            )
        semantic_floor_annotations = world.get_semantic_annotations_by_type(Floor)
        self.assertEqual(len(semantic_floor_annotations), 1)
        self.assertTrue(isinstance(floor.root.parent_connection, FixedConnection))
        self.assertEqual(floor, semantic_floor_annotations[0])

    def test_wall_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            wall = Wall.create_with_new_body_in_world(
                name=PrefixedName("wall"),
                scale=Scale(0.1, 4, 2),
                world=world,
            )
        semantic_wall_annotations = world.get_semantic_annotations_by_type(Wall)
        self.assertEqual(len(semantic_wall_annotations), 1)
        self.assertTrue(isinstance(wall.root.parent_connection, FixedConnection))
        self.assertEqual(wall, semantic_wall_annotations[0])

    def test_aperture_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            aperture = Aperture.create_with_new_region_in_world(
                name=PrefixedName("wall"),
                scale=Scale(0.1, 4, 2),
                world=world,
            )
        semantic_aperture_annotations = world.get_semantic_annotations_by_type(Aperture)
        self.assertEqual(len(semantic_aperture_annotations), 1)
        self.assertTrue(isinstance(aperture.root.parent_connection, FixedConnection))
        self.assertEqual(aperture, semantic_aperture_annotations[0])

    def test_aperture_from_body_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            door = Door.create_with_new_body_in_world(
                name=PrefixedName("door"),
                scale=Scale(0.03, 1, 2),
                world=world,
            )
            aperture = Aperture.create_with_new_region_in_world_from_body(
                name=PrefixedName("wall"),
                world=world,
                body=door.root,
            )
        semantic_aperture_annotations = world.get_semantic_annotations_by_type(Aperture)
        self.assertEqual(len(semantic_aperture_annotations), 2)
        self.assertIn(aperture, semantic_aperture_annotations)
        self.assertIn(door.entry_way, semantic_aperture_annotations)

    def test_has_aperture_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            wall = Wall.create_with_new_body_in_world(
                name=PrefixedName("wall"),
                scale=Scale(0.1, 4, 2),
                world=world,
            )
            door = Door.create_with_new_body_in_world(
                name=PrefixedName("door"),
                scale=Scale(0.03, 1, 2),
                world=world,
            )
            aperture = Aperture.create_with_new_region_in_world_from_body(
                name=PrefixedName("wall"),
                world=world,
                body=door.root,
            )
            wall.add(aperture)

        assert wall.apertures[0] == aperture
        assert aperture.root.parent_kinematic_structure_entity == wall.root

    def _setup_door(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            door = Door.create_with_new_body_in_world(
                name=PrefixedName("door"), scale=Scale(0.03, 1.0, 2.0), world=world
            )
        return world, door

    def test_calculate_world_T_hinge_no_handle(self):
        world, door = self._setup_door()
        with self.assertRaises(MissingSemanticAnnotationError):
            door.calculate_world_T_hinge_based_on_handle(Vector3.Z())

    def test_calculate_world_T_hinge_vertical(self):
        world, door = self._setup_door()
        # Add handle at y=0.4 (right side of door center)
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name=PrefixedName("handle"),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.4),
            )
            door.add(handle)

        # Test Z-axis rotation (vertical hinge)
        # handle is at y=0.4, door width is 1.0. Hinge should be at opposite side: y=-0.5
        world_T_hinge = door.calculate_world_T_hinge_based_on_handle(Vector3.Z())
        expected_T_hinge = (
            door.root.global_transform
            @ HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.5)
        )
        self.assertTrue(np.allclose(world_T_hinge.to_np(), expected_T_hinge.to_np()))

        world, door = self._setup_door()
        # Add handle at y=-0.4 (left side of door center)
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name=PrefixedName("handle"),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.4),
            )
            door.add(handle)

        world_T_hinge = door.calculate_world_T_hinge_based_on_handle(Vector3.Z())
        expected_T_hinge = (
            door.root.global_transform
            @ HomogeneousTransformationMatrix.from_xyz_rpy(y=0.5)
        )
        self.assertTrue(np.allclose(world_T_hinge.to_np(), expected_T_hinge.to_np()))

    def test_calculate_world_T_hinge_horizontal(self):
        world, door = self._setup_door()
        # Add handle
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name=PrefixedName("handle"),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=0.4, z=0.0
                ),
            )
            door.add(handle)

        # Test Y-axis rotation (horizontal hinge)
        # handle z=0. Hinge should be at z=1.0 (opposite of default sign 1 if z=0)
        world_T_hinge = door.calculate_world_T_hinge_based_on_handle(Vector3.Y())
        expected_T_hinge = (
            door.root.global_transform
            @ HomogeneousTransformationMatrix.from_xyz_rpy(z=1.0)
        )
        self.assertTrue(np.allclose(world_T_hinge.to_np(), expected_T_hinge.to_np()))

        world, door = self._setup_door()
        # Add handle
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name=PrefixedName("handle"),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=0.5, z=0.0
                ),
            )
            door.add(handle)

        # handle at z=0.5. Hinge should be at z=-1.0
        handle.root.parent_connection.parent_T_connection_expression = (
            HomogeneousTransformationMatrix.from_xyz_rpy(z=0.5)
        )
        world_T_hinge = door.calculate_world_T_hinge_based_on_handle(Vector3.Y())
        expected_T_hinge = (
            door.root.global_transform
            @ HomogeneousTransformationMatrix.from_xyz_rpy(z=-1.0)
        )
        self.assertTrue(np.allclose(world_T_hinge.to_np(), expected_T_hinge.to_np()))

    def test_calculate_world_T_hinge_invalid_axis(self):
        world, door = self._setup_door()
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name=PrefixedName("handle"),
                world=world,
            )
            door.add(handle)
        with self.assertRaises(InvalidHingeActiveAxis):
            door.calculate_world_T_hinge_based_on_handle(Vector3(1, 1, 0))

    def test_calculate_supporting_surface(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            table = Table.create_with_new_body_in_world(
                name=PrefixedName("table"), world=world
            )
        table_scale = Scale(1.0, 1.0, 0.1)
        table.root.collision = BoundingBoxCollection.from_event(
            table.root, table_scale.to_simple_event().as_composite_set()
        ).as_shapes()
        table.root.visual = table.root.collision

        with world.modify_world():
            surface = table.calculate_supporting_surface()

        self.assertIsNotNone(surface)
        self.assertEqual(surface, table.supporting_surface)
        self.assertEqual(len(world.regions), 1)
        self.assertTrue(len(surface.area.combined_mesh.vertices) > 0)

    def test_supporting_surface_position_on_top_of_table(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            table = Table.create_with_new_body_in_world(
                name=PrefixedName("table"),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=1.5),
            )
        table_scale = Scale(1.0, 1.0, 0.5)
        table.root.collision = BoundingBoxCollection.from_event(
            table.root, table_scale.to_simple_event().as_composite_set()
        ).as_shapes()
        table.root.visual = table.root.collision

        with world.modify_world():
            surface = table.calculate_supporting_surface()

        _, max_point = table.min_max_points
        # supporting surface should be at the height of the table's global z + the max z of the table's bounding box (since the table's origin is at its center)
        expected_z = table.root.global_transform.z + max_point.z

        self.assertIsNotNone(surface)
        self.assertEqual(surface, table.supporting_surface)
        self.assertEqual(expected_z, surface.global_transform.z)

    def test_sample_points_from_surface(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            milk = Milk.create_with_new_body_in_world(
                name=PrefixedName("milk"),
                world=world,
                scale=Scale(0.03, 0.03, 0.1),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.5),
            )
            cereal = Cereal.create_with_new_body_in_world(
                name=PrefixedName("cereal"),
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.5),
            )
            table = Table.create_with_new_body_in_world(
                name=PrefixedName("table"), world=world, scale=Scale(1.0, 1.0, 0.1)
            )
            table.add_object(milk)
            table.add_object(cereal)

            cereal_to_place = Cereal.create_with_new_body_in_world(
                name=PrefixedName("cereal_to_place"),
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
            )

        points = table.sample_points_from_surface(
            amount=10,
        )
        self.assertEqual(len(points), 10)

        min_point, max_point = table.min_max_points
        assert all(p.reference_frame == table.supporting_surface for p in points)
        assert all(p.x >= min_point.x for p in points)
        assert all(p.x <= max_point.x for p in points)
        assert all(p.y >= min_point.y for p in points)
        assert all(p.y <= max_point.y for p in points)
        assert np.allclose([p.z for p in points], 0.0025)

    def test_sample_points_from_surface_with_category_of_interest(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            milk = Milk.create_with_new_body_in_world(
                name=PrefixedName("milk"),
                world=world,
                scale=Scale(0.03, 0.03, 0.1),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.5),
            )
            cereal = Cereal.create_with_new_body_in_world(
                name=PrefixedName("cereal"),
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.5),
            )
            cereal2 = Cereal.create_with_new_body_in_world(
                name=PrefixedName("cereal"),
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.2),
            )
            table = Table.create_with_new_body_in_world(
                name=PrefixedName("table"), world=world, scale=Scale(1.0, 1.0, 0.1)
            )
            table.add_object(milk)
            table.add_object(cereal)
            table.add_object(cereal2)

        with world.modify_world():
            table.calculate_supporting_surface()
        objects_of_interest = [cereal, cereal2]
        sampler = table._untruncated_2d_gaussian_sampler(
            objects_of_interest=objects_of_interest, variance=1
        )
        [object_variable, x_variable, y_variable] = sampler.variables
        for object in objects_of_interest:
            conditional, _ = sampler.conditional({object_variable: object})
            expectation = conditional.expectation([x_variable, y_variable])
            surface_T_object = world.transform(
                object.global_transform, table.supporting_surface
            )
            assert expectation[x_variable] == surface_T_object.x
            assert expectation[y_variable] == surface_T_object.y

    def test_remove_objects_from_sampling_event(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            milk = Milk.create_with_new_body_in_world(
                name=PrefixedName("milk"),
                world=world,
                scale=Scale(0.03, 0.03, 0.1),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.5),
            )
            cereal = Cereal.create_with_new_body_in_world(
                name=PrefixedName("cereal"),
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.5),
            )
            table = Table.create_with_new_body_in_world(
                name=PrefixedName("table"), world=world, scale=Scale(1.0, 1.0, 0.1)
            )
            table.add_object(milk)
            table.add_object(cereal)

        with world.modify_world():
            table.calculate_supporting_surface()

        surface_event: Event = table._2d_surface_sample_space_excluding_objects(0)

        surface_P_milk = world.transform(
            milk.root.global_transform, table.supporting_surface
        ).to_position()
        surface_P_cereal = world.transform(
            cereal.root.global_transform, table.supporting_surface
        ).to_position()

        assert not surface_event.contains(surface_P_milk[:2])
        assert not surface_event.contains(surface_P_cereal[:2])

    def test_sample_points_from_surface_with_object_and_category_of_interest(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            milk = Milk.create_with_new_body_in_world(
                name=PrefixedName("milk"),
                world=world,
                scale=Scale(0.03, 0.03, 0.1),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.5),
            )
            cereal = Cereal.create_with_new_body_in_world(
                name=PrefixedName("cereal"),
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.5),
            )
            cereal2 = Cereal.create_with_new_body_in_world(
                name=PrefixedName("cereal"),
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.2),
            )
            table = Table.create_with_new_body_in_world(
                name=PrefixedName("table"), world=world, scale=Scale(1.0, 1.0, 0.1)
            )
            table.add_object(milk)
            table.add_object(cereal)
            table.add_object(cereal2)

            cereal_to_place = Cereal.create_with_new_body_in_world(
                name=PrefixedName("cereal_to_place"),
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
            )

        points = table.sample_points_from_surface(
            cereal_to_place,
            type(cereal),
            amount=100,
        )
        self.assertEqual(len(points), 100)

        min_point, max_point = table.min_max_points
        assert all(p.reference_frame == table.supporting_surface for p in points)
        assert all(p.x >= min_point.x for p in points)
        assert all(p.x <= max_point.x for p in points)
        assert all(p.y >= min_point.y for p in points)
        assert all(p.y <= max_point.y for p in points)
        assert np.allclose([p.z for p in points], 0.1025)

    def test_floor_polytope(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        points = [
            Point3(0, 0, 0, reference_frame=root),
            Point3(1, 0, 0, reference_frame=root),
            Point3(1, 1, 0, reference_frame=root),
            Point3(0, 1, 0, reference_frame=root),
        ]
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            floor = Floor.create_with_new_body_from_polytope_in_world(
                name=PrefixedName("floor"), world=world, floor_polytope=points
            )
        self.assertEqual(len(world.get_semantic_annotations_by_type(Floor)), 1)
        self.assertTrue(len(floor.root.collision) > 0)

    def test_wall_doors(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            wall = Wall.create_with_new_body_in_world(
                name=PrefixedName("wall"), scale=Scale(0.1, 4, 2), world=world
            )

            door_scale = Scale(0.01, 1, 1)
            door = Door.create_with_new_body_in_world(
                name=PrefixedName("door"), scale=door_scale, world=world
            )

            door2 = Door.create_with_new_body_in_world(
                name=PrefixedName("door2"),
                scale=door_scale,
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=2),
            )

        doors = list(wall.doors)
        self.assertIn(door, doors)
        self.assertNotIn(door2, doors)

    def test_handle_with_thickness(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name=PrefixedName("handle"), world=world, thickness=0.005
            )
        self.assertTrue(len(handle.root.collision) > 1)

    def test_add_aperture_geometry(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            wall = Wall.create_with_new_body_in_world(
                name=PrefixedName("wall"), scale=Scale(0.01, 4, 2), world=world
            )
            initial_shapes_count = len(wall.root.collision)

            aperture = Aperture.create_with_new_region_in_world(
                name=PrefixedName("aperture"), scale=Scale(0.1, 1, 1), world=world
            )
            wall.add(aperture)
        self.assertIn(aperture, wall.apertures)
        self.assertTrue(len(wall.root.collision) > initial_shapes_count)

    def test_create_with_connection_limits(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        lower = DerivativeMap[float]()
        lower.position = -0.5
        upper = DerivativeMap[float]()
        upper.position = 0.5
        limits = DegreeOfFreedomLimits(lower=lower, upper=upper)

        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            Hinge.create_with_new_body_in_world(
                name=PrefixedName("hinge"),
                world=world,
                connection_limits=limits,
                active_axis=Vector3.Z(),
            )

        dof = world.degrees_of_freedom[0]
        self.assertEqual(dof.limits.lower.position, -0.5)
        self.assertEqual(dof.limits.upper.position, 0.5)

    def test_create_with_invalid_connection_limits(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        lower = DerivativeMap[float]()
        lower.position = 0.5
        upper = DerivativeMap[float]()
        upper.position = -0.5
        limits = DegreeOfFreedomLimits(lower=lower, upper=upper)

        with self.assertRaises(InvalidConnectionLimits), world.modify_world():
            Hinge.create_with_new_body_in_world(
                name=PrefixedName("hinge"),
                world=world,
                connection_limits=limits,
                active_axis=Vector3.Z(),
            )

    def test_perceivable_cup(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            cup = Cup.create_with_new_body_in_world(
                name=PrefixedName("cup"), world=world
            )
        cup.class_label = "plastic_cup"
        self.assertEqual(cup.class_label, "plastic_cup")

    def test_is_storage_space(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            cabinet = Cabinet.create_with_new_body_in_world(
                name=PrefixedName("cabinet"), world=world, scale=Scale(0.5, 0.5, 1.0)
            )
            cup = Cup.create_with_new_body_in_world(
                name=PrefixedName("cup"), world=world
            )

            cabinet.add_object(cup)

        self.assertIn(cup, cabinet.objects)
        self.assertEqual(cup.root.parent_kinematic_structure_entity, cabinet.root)

    def test_has_objects_mismatching_world(self):
        world1 = World()
        root1 = Body(name=PrefixedName("root1"))
        with world1.modify_world():
            world1.add_body(root1)
        with world1.modify_world():
            cabinet = Cabinet.create_with_new_body_in_world(
                name=PrefixedName("cabinet"), world=world1, scale=Scale(0.5, 0.5, 1.0)
            )
        world2 = World()
        root2 = Body(name=PrefixedName("root2"))
        with world2.modify_world():
            world2.add_body(root2)
        with world2.modify_world():
            cup = Cup.create_with_new_body_in_world(
                name=PrefixedName("cup"), world=world2
            )

        with self.assertRaises(MismatchingWorld):
            cabinet.add_object(cup)

    def test_double_door_view_point(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            door_left = Door.create_with_new_body_in_world(
                name=PrefixedName("door_left"),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, y=0.5
                ),
            )
            door_right = Door.create_with_new_body_in_world(
                name=PrefixedName("door_right"),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, y=-0.5
                ),
            )
            double_door = DoubleDoor(
                door_0=door_left, door_1=door_right, name=PrefixedName("double_door")
            )
            world.add_semantic_annotation(double_door)

        # View point at origin looking forward (identity)
        view_point_front = HomogeneousTransformationMatrix.from_xyz_rpy()
        self.assertEqual(
            double_door.calculate_left_right_door_from_view_point(view_point_front),
            (door_left, door_right),
        )

        # View point at x=2 looking back (180 deg around Z)
        view_point_back = HomogeneousTransformationMatrix.from_xyz_rpy(x=2, yaw=np.pi)
        self.assertEqual(
            double_door.calculate_left_right_door_from_view_point(view_point_back),
            (door_right, door_left),
        )

    def test_microwave_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            microwave = Microwave.create_with_new_body_in_world(
                name=PrefixedName("microwave"), world=world
            )
            door = Door.create_with_new_body_in_world(
                name=PrefixedName("microwave_door"),
                scale=Scale(0.03, 0.3, 0.3),
                world=world,
            )
            microwave.add(door)

        semantic_microwave_annotations = world.get_semantic_annotations_by_type(
            Microwave
        )
        self.assertEqual(len(semantic_microwave_annotations), 1)
        self.assertEqual(microwave.doors[0], door)

    def test_hood_toaster_coffee_machine_factories(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            hood = Hood.create_with_new_body_in_world(
                name=PrefixedName("hood"), world=world
            )
            toaster = Toaster.create_with_new_body_in_world(
                name=PrefixedName("toaster"), world=world
            )
            coffee_machine = CoffeeMachine.create_with_new_body_in_world(
                name=PrefixedName("coffee_machine"), world=world
            )

        self.assertEqual(len(world.get_semantic_annotations_by_type(Hood)), 1)
        self.assertEqual(len(world.get_semantic_annotations_by_type(Toaster)), 1)
        self.assertEqual(
            len(world.get_semantic_annotations_by_type(CoffeeMachine)), 1
        )
        self.assertEqual(
            world.root, hood.root.parent_kinematic_structure_entity
        )
        self.assertEqual(
            world.root, toaster.root.parent_kinematic_structure_entity
        )
        self.assertEqual(
            world.root, coffee_machine.root.parent_kinematic_structure_entity
        )


@dataclass(eq=False)
class _AnnotationWithOverlappingPartWholeRelationshipFields(
    HasRootBody, PartWholeRelationship
):
    """
    Throwaway whole whose two part-whole relationship fields have overlapping element types
    (``Hinge`` is a subclass of ``MechanicalJoint``), so a ``Hinge`` matches both.
    """

    joint: Optional[MechanicalJoint] = field(
        default=None,
        metadata=FieldMetadata(other_metadata=[IsPartWholeRelationship()]).as_dict(),
    )
    specific_joint: Optional[Hinge] = field(
        default=None,
        metadata=FieldMetadata(other_metadata=[IsPartWholeRelationship()]).as_dict(),
    )


def _world_with_root() -> World:
    world = World()
    root = Body(name=PrefixedName("root"))
    with world.modify_world():
        world.add_body(root)
    return world


def test_add_routes_handle_as_child():
    """add(handle) mounts the handle as a child of the door (default strategy)."""
    world = _world_with_root()
    with world.modify_world():
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world
        )
        handle = Handle.create_with_new_body_in_world(
            name=PrefixedName("handle"), world=world
        )
        door.add(handle)

    assert door.handle == handle
    assert door.root == handle.root.parent_kinematic_structure_entity


def test_add_routes_hinge_by_reparenting_self():
    """add(hinge) re-parents the door under the hinge (Hinge._mount_strategy)."""
    world = _world_with_root()
    with world.modify_world():
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world
        )
        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"), world=world, active_axis=Vector3.Z()
        )
        door.add(hinge)

    assert door.mechanical_joint == hinge
    assert door.root.parent_kinematic_structure_entity == hinge.root
    assert isinstance(hinge.root.parent_connection, RevoluteConnection)


def test_add_routes_slider_by_reparenting_self():
    """add(slider) re-parents the drawer under the slider (Slider._mount_strategy)."""
    world = _world_with_root()
    with world.modify_world():
        drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"), scale=Scale(0.2, 0.3, 0.2), world=world
        )
        slider = Slider.create_with_new_body_in_world(
            name=PrefixedName("slider"), world=world, active_axis=Vector3.X()
        )
        drawer.add(slider)

    assert drawer.mechanical_joint == slider
    assert drawer.root.parent_kinematic_structure_entity == slider.root
    assert isinstance(slider.root.parent_connection, PrismaticConnection)


def test_add_routes_plural_drawer_and_door():
    """add() appends to the right list when the matching part-whole relationship field is plural."""
    world = _world_with_root()
    with world.modify_world():
        fridge = Fridge.create_with_new_body_in_world(
            name=PrefixedName("fridge"), world=world, scale=Scale(1, 1, 2.0)
        )
        drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"), world=world
        )
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world
        )
        fridge.add(drawer)
        fridge.add(door)

    assert drawer in fridge.drawers
    assert door in fridge.doors
    assert drawer not in fridge.doors
    assert door not in fridge.drawers


def test_add_routes_aperture_with_cut():
    """add(aperture) cuts the wall geometry and mounts the aperture (Aperture._mount_strategy)."""
    world = _world_with_root()
    with world.modify_world():
        wall = Wall.create_with_new_body_in_world(
            name=PrefixedName("wall"), scale=Scale(0.1, 4, 2), world=world
        )
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world
        )
        aperture = Aperture.create_with_new_region_in_world_from_body(
            name=PrefixedName("aperture"), world=world, body=door.root
        )
        wall.add(aperture)

    assert wall.apertures[0] == aperture
    assert aperture.root.parent_kinematic_structure_entity == wall.root


def test_add_object_stores_occupants():
    """Containment occupants are stored via add_object (occupancy, not parthood)."""
    world = _world_with_root()
    with world.modify_world():
        table = Table.create_with_new_body_in_world(
            name=PrefixedName("table"), world=world, scale=Scale(1.0, 1.0, 0.1)
        )
        milk = Milk.create_with_new_body_in_world(
            name=PrefixedName("milk"), world=world, scale=Scale(0.03, 0.03, 0.1)
        )
        cereal = Cereal.create_with_new_body_in_world(
            name=PrefixedName("cereal"), world=world, scale=Scale(0.1, 0.03, 0.2)
        )
        table.add_object(milk)
        table.add_object(cereal)

    assert milk in table.objects
    assert cereal in table.objects
    assert table.root == milk.root.parent_kinematic_structure_entity


def test_add_does_not_route_occupants():
    """An occupant matches no part-whole relationship field, so add() rejects it (it must use place)."""
    world = _world_with_root()
    with world.modify_world():
        fridge = Fridge.create_with_new_body_in_world(
            name=PrefixedName("fridge"), world=world, scale=Scale(1, 1, 2.0)
        )
        milk = Milk.create_with_new_body_in_world(
            name=PrefixedName("milk"), world=world, scale=Scale(0.03, 0.03, 0.1)
        )
        with pytest.raises(CannotBeAPartOf):
            fridge.add(milk)
        fridge.add_object(milk)

    assert milk in fridge.objects


def test_add_rejects_unsupported_part_type():
    """add() of a part type the annotation has no part-whole relationship field for raises CannotBeAPartOf."""
    world = _world_with_root()
    with world.modify_world():
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world
        )
        drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"), world=world
        )
        # A Door has handle/hinge part-whole relationship fields but no drawer field.
        with pytest.raises(CannotBeAPartOf):
            door.add(drawer)


def test_add_raises_on_ambiguous_part():
    """add() of a part matching more than one part-whole relationship field raises AmbiguousPart."""
    world = _world_with_root()
    with world.modify_world():
        whole = _AnnotationWithOverlappingPartWholeRelationshipFields.create_with_new_body_in_world(
            name=PrefixedName("whole"), world=world
        )
        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"), world=world, active_axis=Vector3.Z()
        )
        # A Hinge is both a MechanicalJoint (joint field) and a Hinge (specific_joint field).
        with pytest.raises(AmbiguousPart):
            whole.add(hinge)


def test_add_field_name_resolves_ambiguity_to_base_field():
    """add(part, field_name=...) routes to the named field even when the type matches several."""
    world = _world_with_root()
    with world.modify_world():
        whole = _AnnotationWithOverlappingPartWholeRelationshipFields.create_with_new_body_in_world(
            name=PrefixedName("whole"), world=world
        )
        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"), world=world, active_axis=Vector3.Z()
        )
        whole.add(hinge, field_name="joint")
    assert whole.joint is hinge
    assert whole.specific_joint is None


def test_add_field_name_resolves_ambiguity_to_specific_field():
    """add(part, field_name=...) can route the same part to the other matching field."""
    world = _world_with_root()
    with world.modify_world():
        whole = _AnnotationWithOverlappingPartWholeRelationshipFields.create_with_new_body_in_world(
            name=PrefixedName("whole"), world=world
        )
        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"), world=world, active_axis=Vector3.Z()
        )
        whole.add(hinge, field_name="specific_joint")
    assert whole.specific_joint is hinge
    assert whole.joint is None


def test_add_unknown_field_name_raises():
    """add(part, field_name=...) with a name that is not a part-whole field raises."""
    world = _world_with_root()
    with world.modify_world():
        whole = _AnnotationWithOverlappingPartWholeRelationshipFields.create_with_new_body_in_world(
            name=PrefixedName("whole"), world=world
        )
        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"), world=world, active_axis=Vector3.Z()
        )
        with pytest.raises(UnknownPartWholeRelationshipField):
            whole.add(hinge, field_name="not_a_field")


def test_add_field_name_with_mismatching_type_raises():
    """add(part, field_name=...) still type-checks: a part the named field rejects raises."""
    world = _world_with_root()
    with world.modify_world():
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world
        )
        handle = Handle.create_with_new_body_in_world(
            name=PrefixedName("handle"), world=world
        )
        # 'mechanical_joint' is a real part-whole field of Door, but a Handle is not a MechanicalJoint.
        with pytest.raises(CannotBeAPartOf):
            door.add(handle, field_name="mechanical_joint")


def test_containment_only_annotation_has_no_add():
    """A pure-containment annotation (Table) exposes add_object but not the part-whole add()."""
    assert not hasattr(Table, "add")
    assert hasattr(Table, "add_object")


def test_mechanical_joint_mount_splices_under_whole_parent():
    """
    When the whole already sits under a non-root parent, mounting a mechanical joint splices the joint
    between the whole and that parent (parent -> joint -> whole): the whole's ancestry is preserved and
    the joint keeps its active (revolute) connection, now anchored at the whole's parent.
    """
    world = _world_with_root()
    with world.modify_world():
        fridge = Fridge.create_with_new_body_in_world(
            name=PrefixedName("fridge"), world=world, scale=Scale(1, 1, 2.0)
        )
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world
        )
        # Place the door inside the fridge first, so its parent is the fridge (not the world root).
        fridge.add(door)
        assert door.root.parent_kinematic_structure_entity == fridge.root

        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"), world=world, active_axis=Vector3.Z()
        )
        door.add(hinge)

    # Spliced topology: fridge -> hinge -> door.
    assert door.root.parent_kinematic_structure_entity == hinge.root
    assert hinge.root.parent_kinematic_structure_entity == fridge.root
    # The joint kept its active connection (it was not collapsed to a FixedConnection).
    assert isinstance(hinge.root.parent_connection, RevoluteConnection)

    # The fridge is still upstream of the door (its parent was not dropped).
    ancestors = []
    entity = door.root.parent_kinematic_structure_entity
    while entity is not None and entity != world.root:
        ancestors.append(entity)
        entity = entity.parent_kinematic_structure_entity
    assert fridge.root in ancestors


def test_mechanical_joint_mount_onto_same_whole_is_idempotent():
    """Mounting the same joint onto the whole it already connects is a no-op (no self-loop, no error)."""
    world = _world_with_root()
    with world.modify_world():
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world
        )
        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"), world=world, active_axis=Vector3.Z()
        )
        door.add(hinge)
        door.add(hinge)

    assert door.mechanical_joint == hinge
    assert door.root.parent_kinematic_structure_entity == hinge.root


def test_mechanical_joint_cannot_be_mounted_onto_a_second_whole():
    """A joint already connecting one whole rejects being mounted onto a different whole."""
    world = _world_with_root()
    with world.modify_world():
        door1 = Door.create_with_new_body_in_world(
            name=PrefixedName("door1"), scale=Scale(0.03, 1, 2), world=world
        )
        door2 = Door.create_with_new_body_in_world(
            name=PrefixedName("door2"), scale=Scale(0.03, 1, 2), world=world
        )
        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"), world=world, active_axis=Vector3.Z()
        )
        door1.add(hinge)
        with pytest.raises(MechanicalJointAlreadyMounted):
            door2.add(hinge)


if __name__ == "__main__":
    unittest.main()
