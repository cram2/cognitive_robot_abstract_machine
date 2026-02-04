import time
import unittest
from dataclasses import dataclass
from semantic_digital_twin.orm.ormatic_interface import *

import numpy as np
import pytest

from krrood.adapters.json_serializer import to_json
from krrood.ormatic.dao import to_dao
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    InvalidPlaneDimensions,
    InvalidHingeActiveAxis,
    InvalidConnectionLimits,
    MissingSemanticAnnotationError,
    MismatchingWorld,
    MissingWorldModificationContextError,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasCaseAsRootBody,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Door,
    Drawer,
    Wall,
    Hinge,
    Fridge,
    DoubleDoor,
    Slider,
    Floor,
    Aperture,
    Table,
    Cup,
    Cabinet,
)
from semantic_digital_twin.spatial_types import (
    Vector3,
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Scale, Box
from semantic_digital_twin.world_description.shape_collection import (
    ShapeCollection,
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body, Region
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)


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
            door.add_hinge(hinge)
        assert isinstance(hinge.root.parent_connection, RevoluteConnection)
        assert door.root.parent_kinematic_structure_entity == hinge.root
        assert door.hinge == hinge

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
            door.add_handle(handle)

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
            drawer.add_slider(slider)

        assert drawer.root.parent_kinematic_structure_entity == slider.root
        assert isinstance(slider.root.parent_connection, PrismaticConnection)
        assert drawer.slider == slider

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
            fridge.add_drawer(drawer)

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
            fridge.add_door(door)

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
            wall.add_aperture(aperture)

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
            door.add_handle(handle)

        # Test Z-axis rotation (vertical hinge)
        # handle is at y=0.4, door width is 1.0. Hinge should be at opposite side: y=-0.5
        world_T_hinge = door.calculate_world_T_hinge_based_on_handle(Vector3.Z())
        expected_T_hinge = (
            door.root.global_pose @ HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.5)
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
            door.add_handle(handle)

        world_T_hinge = door.calculate_world_T_hinge_based_on_handle(Vector3.Z())
        expected_T_hinge = (
            door.root.global_pose @ HomogeneousTransformationMatrix.from_xyz_rpy(y=0.5)
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
            door.add_handle(handle)

        # Test Y-axis rotation (horizontal hinge)
        # handle z=0. Hinge should be at z=1.0 (opposite of default sign 1 if z=0)
        world_T_hinge = door.calculate_world_T_hinge_based_on_handle(Vector3.Y())
        expected_T_hinge = (
            door.root.global_pose @ HomogeneousTransformationMatrix.from_xyz_rpy(z=1.0)
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
            door.add_handle(handle)

        # handle at z=0.5. Hinge should be at z=-1.0
        handle.root.parent_connection.parent_T_connection_expression = (
            HomogeneousTransformationMatrix.from_xyz_rpy(z=0.5)
        )
        world_T_hinge = door.calculate_world_T_hinge_based_on_handle(Vector3.Y())
        expected_T_hinge = (
            door.root.global_pose @ HomogeneousTransformationMatrix.from_xyz_rpy(z=-1.0)
        )
        self.assertTrue(np.allclose(world_T_hinge.to_np(), expected_T_hinge.to_np()))

    def test_calculate_world_T_hinge_invalid_axis(self):
        world, door = self._setup_door()
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name=PrefixedName("handle"),
                world=world,
            )
            door.add_handle(handle)
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
        self.assertEqual(len(world.regions), 1)
        self.assertEqual(table.supporting_surface, surface)
        self.assertTrue(len(surface.area.combined_mesh.vertices) > 0)

    def test_points_on_surface(self):
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

        points = table.points_on_supporting_surface(amount=10)
        self.assertEqual(len(points), 10)
        for p in points:
            self.assertEqual(p.reference_frame, table.root)
            self.assertAlmostEqual(float(p.z), 0.05 + 0.01)

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

    def test_kinematic_helpers(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)

        mid = Body(name=PrefixedName("mid"))
        child = Body(name=PrefixedName("child"))
        with world.modify_world():
            world.add_body(mid)
            world.add_connection(
                FixedConnection(
                    root,
                    mid,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=1
                    ),
                )
            )
            world.add_body(child)
            world.add_connection(
                FixedConnection(
                    mid,
                    child,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=1
                    ),
                )
            )

        slider = Slider(name=PrefixedName("slider"), root=child)
        with world.modify_world():
            world.add_semantic_annotation(slider)

        parent_T_self = root.global_pose.inverse() @ slider.root.global_pose
        self.assertAlmostEqual(parent_T_self[0, 3], 2.0)

        self_T_child = slider.root.global_pose.inverse() @ root.global_pose
        self.assertAlmostEqual(self_T_child[0, 3], -2.0)

        self.assertEqual(slider.get_new_grandparent(mid), root)

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
            wall.add_aperture(aperture)
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

    def test_has_storage_space(self):
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


if __name__ == "__main__":
    unittest.main()
