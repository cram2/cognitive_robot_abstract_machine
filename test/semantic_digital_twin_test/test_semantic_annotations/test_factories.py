import unittest
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

from random_events.product_algebra import Event
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    CannotBeAPartOf,
    AmbiguousPart,
    UnknownPartWholeRelationshipField,
)
from semantic_digital_twin.exceptions import (
    InvalidPlaneDimensions,
    InvalidHingeActiveAxis,
    InvalidConnectionLimits,
    MissingMovableJointError,
    MissingSemanticAnnotationError,
    MismatchingWorld,
)
from semantic_digital_twin.orm.ormatic_interface import *
from semantic_digital_twin.semantic_annotations.mixins import (
    PartWholeRelationship,
    HasRootBody,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasCaseAsRootBody,
)
from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    DoubleDoor,
    Elevator,
    FirstFloor,
    Floor,
    GroundFloor,
    Cup,
    Cabinet,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Door,
    Drawer,
    Wall,
    Fridge,
    BottleCap,
    DoorWithType,
    Aperture,
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
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
)
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
    ScrewConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    VolumetricBoundingBox,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    ShapeCollection,
)
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.specifications.connections import (
    PrismaticConnectionSpecification,
    RevoluteConnectionSpecification,
    ScrewConnectionSpecification,
)
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)


class TestFactories(unittest.TestCase):
    def test_handle_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            returned_handle = Handle.get_annotation_specification(
                "handle",
                Handle.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(0.1, 0.2, 0.03), thickness=0.03
                ),
            ).spawn(world)
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

    def test_active_has_body_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            returned_door = Door.create_with_new_body_in_world(
                name="door",
                world=world,
                parent_connection_specification=RevoluteConnectionSpecification(
                    axis=Vector3.Z()
                ),
            )
            returned_drawer = Drawer.create_with_new_body_in_world(
                name="drawer",
                world=world,
                parent_connection_specification=PrismaticConnectionSpecification(
                    axis=Vector3.X()
                ),
            )
        [queried_door] = world.get_semantic_annotations_by_type(Door)
        self.assertEqual(returned_door, queried_door)
        self.assertEqual(world.root, queried_door.movable_joint.parent)
        self.assertIsInstance(queried_door.movable_joint, RevoluteConnection)
        [queried_drawer] = world.get_semantic_annotations_by_type(Drawer)
        self.assertEqual(returned_drawer, queried_drawer)
        self.assertEqual(world.root, queried_drawer.movable_joint.parent)
        self.assertIsInstance(queried_drawer.movable_joint, PrismaticConnection)

    def test_door_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            returned_door = Door.create_with_new_body_in_world(
                name="door", scale=Scale(0.03, 1, 2), world=world
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

    def test_door_fixed_to_its_parent_has_no_joint(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            door = Door.create_with_new_body_in_world(
                name="door", scale=Scale(0.03, 1, 2), world=world
            )
        assert isinstance(door.root.parent_connection, FixedConnection)
        assert door.movable_joint is None

    def test_door_factory_invalid(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            with pytest.raises(InvalidPlaneDimensions):
                Door.create_with_new_body_in_world(
                    name="door",
                    scale=Scale(1, 1, 2),
                    world=world,
                )

            with pytest.raises(InvalidPlaneDimensions):
                Door.create_with_new_body_in_world(
                    name="door",
                    scale=Scale(1, 2, 1),
                    world=world,
                )

    def test_door_on_hinge_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            door = Door.create_with_new_body_in_world(
                name="door",
                scale=Scale(0.03, 1, 2),
                world=world,
                parent_connection_specification=RevoluteConnectionSpecification(
                    axis=Vector3.Z()
                ),
            )
        assert set(world.kinematic_structure_entities) == {
            world.root,
            door.root,
            door.entry_way.root,
        }
        assert isinstance(door.movable_joint, RevoluteConnection)
        assert door.movable_joint.parent == world.root

    def test_bottle_cap_on_screw_factory(self):
        world = World.create_with_root_body("root")
        screw_pitch = 0.005
        with world.modify_world():
            bottle_cap = BottleCap.create_with_new_body_in_world(
                name="bottle_cap",
                world=world,
                scale=Scale(0.03, 0.03, 0.02),
                parent_connection_specification=ScrewConnectionSpecification(
                    axis=Vector3.Z(), screw_pitch=screw_pitch
                ),
            )
        assert isinstance(bottle_cap.movable_joint, ScrewConnection)
        assert bottle_cap.movable_joint.screw_pitch == screw_pitch
        assert bottle_cap.movable_joint.parent == world.root

    def test_has_handle_factory(self):
        world = World.create_with_root_body("root")
        root = world.root
        with world.modify_world():
            door = Door.create_with_new_body_in_world(
                name="door",
                scale=Scale(0.03, 1, 2),
                world=world,
            )

            handle = Handle.create_with_new_body_in_world(
                name="handle",
                world=world,
            )
        assert len(world.kinematic_structure_entities) == 4

        assert root == handle.root.parent_kinematic_structure_entity
        with world.modify_world():
            door.add(handle)

        assert door.root == handle.root.parent_kinematic_structure_entity
        assert door.handle == handle

    def test_case_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            fridge = Fridge.create_with_new_body_in_world(
                name="case",
                world=world,
                scale=Scale(1, 1, 2.0),
            )

        assert isinstance(fridge, HasCaseAsRootBody)

        semantic_container_annotations = world.get_semantic_annotations_by_type(Fridge)
        self.assertEqual(len(semantic_container_annotations), 1)

        assert len(world.get_semantic_annotations_by_type(HasCaseAsRootBody)) == 1

    def test_drawer_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            drawer = Drawer.create_with_new_body_in_world(
                name="drawer",
                world=world,
                scale=Scale(0.2, 0.3, 0.2),
            )
        assert isinstance(drawer, HasCaseAsRootBody)
        semantic_drawer_annotations = world.get_semantic_annotations_by_type(Drawer)
        self.assertEqual(len(semantic_drawer_annotations), 1)

    def test_hole_direction_carries_the_entity_own_root_as_reference_frame(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            drawer = Drawer.create_with_new_body_in_world(
                name="drawer",
                world=world,
                scale=Scale(0.2, 0.3, 0.2),
            )

        assert drawer.hole_direction.reference_frame is drawer.root
        np.testing.assert_allclose(drawer.hole_direction.to_np()[:3], [0, 0, 1])

    def test_drawer_on_rails_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            drawer = Drawer.create_with_new_body_in_world(
                name="drawer",
                scale=Scale(0.2, 0.3, 0.2),
                world=world,
                parent_connection_specification=PrismaticConnectionSpecification(
                    axis=Vector3.X()
                ),
            )
        assert set(world.kinematic_structure_entities) == {world.root, drawer.root}
        assert isinstance(drawer.movable_joint, PrismaticConnection)
        assert drawer.movable_joint.parent == world.root

    def test_has_drawer_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            fridge = Fridge.create_with_new_body_in_world(
                name="case",
                world=world,
                scale=Scale(1, 1, 2.0),
            )
            drawer = Drawer.create_with_new_body_in_world(name="drawer", world=world)
            fridge.add(drawer)

        semantic_drawer_annotations = world.get_semantic_annotations_by_type(Drawer)
        self.assertEqual(len(semantic_drawer_annotations), 1)
        assert fridge.drawers[0] == drawer

    def test_has_doors_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            fridge = Fridge.create_with_new_body_in_world(
                name="case",
                world=world,
                scale=Scale(1, 1, 2.0),
            )
            door = Door.create_with_new_body_in_world(
                name="left_door",
                world=world,
            )
            fridge.add(door)

        semantic_door_annotations = world.get_semantic_annotations_by_type(Door)
        self.assertEqual(len(semantic_door_annotations), 1)
        assert fridge.doors[0] == door

    def test_floor_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            floor = Floor.create_with_new_body_in_world(
                name="floor",
                world=world,
                scale=Scale(5, 5, 0.01),
            )
        semantic_floor_annotations = world.get_semantic_annotations_by_type(Floor)
        self.assertEqual(len(semantic_floor_annotations), 1)
        self.assertTrue(isinstance(floor.root.parent_connection, FixedConnection))
        self.assertEqual(floor, semantic_floor_annotations[0])

    def test_wall_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            wall = Wall.create_with_new_body_in_world(
                name="wall",
                scale=Scale(0.1, 4, 2),
                world=world,
            )
        semantic_wall_annotations = world.get_semantic_annotations_by_type(Wall)
        self.assertEqual(len(semantic_wall_annotations), 1)
        self.assertTrue(isinstance(wall.root.parent_connection, FixedConnection))
        self.assertEqual(wall, semantic_wall_annotations[0])

    def test_aperture_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            aperture = Aperture.create_with_new_region_in_world(
                name="wall",
                scale=Scale(0.1, 4, 2),
                world=world,
            )
        semantic_aperture_annotations = world.get_semantic_annotations_by_type(Aperture)
        self.assertEqual(len(semantic_aperture_annotations), 1)
        self.assertTrue(isinstance(aperture.root.parent_connection, FixedConnection))
        self.assertEqual(aperture, semantic_aperture_annotations[0])

    def test_aperture_from_body_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            door = Door.create_with_new_body_in_world(
                name="door",
                scale=Scale(0.03, 1, 2),
                world=world,
            )
            aperture = Aperture.create_with_new_region_in_world_from_body(
                name="wall",
                world=world,
                body=door.root,
            )
        semantic_aperture_annotations = world.get_semantic_annotations_by_type(Aperture)
        self.assertEqual(len(semantic_aperture_annotations), 2)
        self.assertIn(aperture, semantic_aperture_annotations)
        self.assertIn(door.entry_way, semantic_aperture_annotations)

    def test_has_aperture_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            wall = Wall.create_with_new_body_in_world(
                name="wall",
                scale=Scale(0.1, 4, 2),
                world=world,
            )
            door = Door.create_with_new_body_in_world(
                name="door",
                scale=Scale(0.03, 1, 2),
                world=world,
            )
            aperture = Aperture.create_with_new_region_in_world_from_body(
                name="wall",
                world=world,
                body=door.root,
            )
            wall.add(aperture)

        assert wall.apertures[0] == aperture
        assert aperture.root.parent_kinematic_structure_entity == wall.root

    def _setup_door(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            door = Door.create_with_new_body_in_world(
                name="door", scale=Scale(0.03, 1.0, 2.0), world=world
            )
        return world, door

    def test_door_movable_joint_needs_a_handle(self):
        world, door = self._setup_door()
        with self.assertRaises(MissingSemanticAnnotationError):
            door.calculate_self_T_movable_joint(Vector3.Z())

    def test_door_movable_joint_is_on_the_vertical_edge_opposite_the_handle(self):
        world, door = self._setup_door()
        # Add handle at y=0.4 (right side of door center)
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name="handle",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.4),
            )
            door.add(handle)

        # Test Z-axis rotation (vertical hinge)
        # handle is at y=0.4, door width is 1.0. Hinge should be at opposite side: y=-0.5
        door_T_hinge = door.calculate_self_T_movable_joint(Vector3.Z())
        np.testing.assert_allclose(
            door_T_hinge.to_np(),
            HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.5).to_np(),
        )

        world, door = self._setup_door()
        # Add handle at y=-0.4 (left side of door center)
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name="handle",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.4),
            )
            door.add(handle)

        door_T_hinge = door.calculate_self_T_movable_joint(Vector3.Z())
        np.testing.assert_allclose(
            door_T_hinge.to_np(),
            HomogeneousTransformationMatrix.from_xyz_rpy(y=0.5).to_np(),
        )

    def test_door_movable_joint_is_on_the_same_edge_for_an_axis_of_either_sign(self):
        world, door = self._setup_door()
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name="handle",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.4),
            )
            door.add(handle)

        np.testing.assert_allclose(
            door.calculate_self_T_movable_joint(Vector3.NEGATIVE_Z()).to_np(),
            door.calculate_self_T_movable_joint(Vector3.Z()).to_np(),
        )

    def test_door_mounts_on_a_hinge_opposite_its_handle(self):
        world, door = self._setup_door()
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name="handle",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.4),
            )
            door.add(handle)
        world_T_door = door.root.global_transform
        world_T_hinge = world_T_door @ door.calculate_self_T_movable_joint(Vector3.Z())
        limits = DegreeOfFreedomLimits.from_position_range_and_speed(
            lower_position=0.0, upper_position=np.pi / 2
        )

        specification = RevoluteConnectionSpecification(
            axis=Vector3.Z(), dof_limits=limits
        )

        joint = door.mount_on_movable_joint(specification)

        assert joint is door.movable_joint
        assert isinstance(joint, RevoluteConnection)
        assert joint.dof.limits.upper.position == limits.upper.position
        assert specification.connection_T_child is None
        np.testing.assert_allclose(
            door.root.global_transform.to_np(), world_T_door.to_np(), atol=1e-12
        )
        joint.position = np.pi / 2
        world.notify_state_change()
        np.testing.assert_allclose(
            door.root.global_transform.to_np(),
            (
                world_T_hinge
                @ HomogeneousTransformationMatrix.from_xyz_rpy(yaw=np.pi / 2)
                @ world_T_hinge.inverse()
                @ world_T_door
            ).to_np(),
            atol=1e-12,
        )

    def test_door_movable_joint_is_on_the_horizontal_edge_opposite_the_handle(self):
        world, door = self._setup_door()
        # Add handle
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name="handle",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=0.4, z=0.0
                ),
            )
            door.add(handle)

        # Test Y-axis rotation (horizontal hinge)
        # handle z=0. Hinge should be at z=1.0 (opposite of default sign 1 if z=0)
        door_T_hinge = door.calculate_self_T_movable_joint(Vector3.Y())
        np.testing.assert_allclose(
            door_T_hinge.to_np(),
            HomogeneousTransformationMatrix.from_xyz_rpy(z=1.0).to_np(),
        )

        world, door = self._setup_door()
        # Add handle
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name="handle",
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
        door_T_hinge = door.calculate_self_T_movable_joint(Vector3.Y())
        np.testing.assert_allclose(
            door_T_hinge.to_np(),
            HomogeneousTransformationMatrix.from_xyz_rpy(z=-1.0).to_np(),
        )

    def test_door_movable_joint_rejects_an_axis_off_the_door_plane(self):
        world, door = self._setup_door()
        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name="handle",
                world=world,
            )
            door.add(handle)
        with self.assertRaises(InvalidHingeActiveAxis):
            door.calculate_self_T_movable_joint(Vector3(1, 1, 0))

    def test_calculate_supporting_surface(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            table = Table.create_with_new_body_in_world(name="table", world=world)
        table_scale = Scale(1.0, 1.0, 0.1)
        table.root.collision = BoundingBoxCollection.from_event(
            VolumetricBoundingBox,
            table.root,
            table_scale.to_simple_event().as_composite_set(),
        ).as_shapes()
        table.root.visual = table.root.collision

        with world.modify_world():
            surface = table.calculate_supporting_surface()

        self.assertIsNotNone(surface)
        self.assertEqual(surface, table.supporting_surface)
        self.assertEqual(len(world.regions), 1)
        self.assertTrue(len(surface.area.combined_mesh.vertices) > 0)

    def test_supporting_surface_position_on_top_of_table(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            table = Table.create_with_new_body_in_world(
                name="table",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=1.5),
            )
        table_scale = Scale(1.0, 1.0, 0.5)
        table.root.collision = BoundingBoxCollection.from_event(
            VolumetricBoundingBox,
            table.root,
            table_scale.to_simple_event().as_composite_set(),
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

    def test_supporting_surface_on_top_of_table_with_origin_at_a_corner(self):
        """
        A table whose origin is a corner on the floor, as a scanned or vendor asset
        often has it, still gets its supporting surface on its top.
        """
        world = World.create_with_root_body("root")
        with world.modify_world():
            table = Table.create_with_new_body_in_world(
                name="table",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.0, y=-1.0, yaw=np.pi / 2
                ),
            )
        table.root.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=0.5, y=0.3, z=0.25, reference_frame=table.root
                    ),
                    scale=Scale(1.0, 0.6, 0.5),
                )
            ],
            reference_frame=table.root,
        )
        table.root.visual = table.root.collision

        with world.modify_world():
            surface = table.calculate_supporting_surface()

        self.assertIsNotNone(surface)
        table_top = Point3(0.5, 0.3, 0.5, reference_frame=table.root)
        expected = world.transform(table_top, world.root).to_np()[:3]
        np.testing.assert_allclose(
            surface.global_transform.position.to_np()[:3], expected, atol=1e-9
        )
        surface_box = surface.area.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
        table_box = table.root.collision.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
        np.testing.assert_allclose(
            [
                surface_box.min_x,
                surface_box.max_x,
                surface_box.min_y,
                surface_box.max_y,
            ],
            [table_box.min_x, table_box.max_x, table_box.min_y, table_box.max_y],
            atol=1e-9,
        )

    def test_sample_points_from_surface(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            milk = Milk.create_with_new_body_in_world(
                name="milk",
                world=world,
                scale=Scale(0.03, 0.03, 0.1),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.5),
            )
            cereal = Cereal.create_with_new_body_in_world(
                name="cereal",
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.5),
            )
            table = Table.create_with_new_body_in_world(
                name="table", world=world, scale=Scale(1.0, 1.0, 0.1)
            )
            table.add_object(milk)
            table.add_object(cereal)

            cereal_to_place = Cereal.create_with_new_body_in_world(
                name="cereal_to_place",
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
        world = World.create_with_root_body("root")
        with world.modify_world():
            milk = Milk.create_with_new_body_in_world(
                name="milk",
                world=world,
                scale=Scale(0.03, 0.03, 0.1),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.5),
            )
            cereal = Cereal.create_with_new_body_in_world(
                name="cereal",
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.5),
            )
            cereal2 = Cereal.create_with_new_body_in_world(
                name="cereal",
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.2),
            )
            table = Table.create_with_new_body_in_world(
                name="table", world=world, scale=Scale(1.0, 1.0, 0.1)
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
        world = World.create_with_root_body("root")
        with world.modify_world():
            milk = Milk.create_with_new_body_in_world(
                name="milk",
                world=world,
                scale=Scale(0.03, 0.03, 0.1),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.5),
            )
            cereal = Cereal.create_with_new_body_in_world(
                name="cereal",
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.5),
            )
            table = Table.create_with_new_body_in_world(
                name="table", world=world, scale=Scale(1.0, 1.0, 0.1)
            )
            table.add_object(milk)
            table.add_object(cereal)

        with world.modify_world():
            table.calculate_supporting_surface()

        surface_event: Event = table._2d_surface_sample_space_excluding_objects(0)

        surface_P_milk = world.transform(
            milk.root.global_transform, table.supporting_surface
        ).position
        surface_P_cereal = world.transform(
            cereal.root.global_transform, table.supporting_surface
        ).position

        assert not surface_event.contains(surface_P_milk[:2])
        assert not surface_event.contains(surface_P_cereal[:2])

    def test_sample_points_from_surface_with_object_and_category_of_interest(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            milk = Milk.create_with_new_body_in_world(
                name="milk",
                world=world,
                scale=Scale(0.03, 0.03, 0.1),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.5),
            )
            cereal = Cereal.create_with_new_body_in_world(
                name="cereal",
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.5),
            )
            cereal2 = Cereal.create_with_new_body_in_world(
                name="cereal",
                world=world,
                scale=Scale(0.1, 0.03, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.2),
            )
            table = Table.create_with_new_body_in_world(
                name="table", world=world, scale=Scale(1.0, 1.0, 0.1)
            )
            table.add_object(milk)
            table.add_object(cereal)
            table.add_object(cereal2)

            cereal_to_place = Cereal.create_with_new_body_in_world(
                name="cereal_to_place",
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
                name="floor", world=world, floor_polytope=points
            )
        self.assertEqual(len(world.get_semantic_annotations_by_type(Floor)), 1)
        self.assertTrue(len(floor.root.collision) > 0)

    def test_wall_doors(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            wall = Wall.create_with_new_body_in_world(
                name="wall", scale=Scale(0.1, 4, 2), world=world
            )

            door_scale = Scale(0.01, 1, 1)
            door = Door.create_with_new_body_in_world(
                name="door", scale=door_scale, world=world
            )

            door2 = Door.create_with_new_body_in_world(
                name="door2",
                scale=door_scale,
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=2),
            )

        doors = list(wall.doors)
        self.assertIn(door, doors)
        self.assertNotIn(door2, doors)

    def test_handle_with_thickness(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            handle = Handle.get_annotation_specification(
                "handle",
                Handle.get_default_root_kinematic_structure_entity_specification(
                    thickness=0.005
                ),
            ).spawn(world)
        self.assertTrue(len(handle.root.collision) > 1)

    def test_add_aperture_geometry(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            wall = Wall.create_with_new_body_in_world(
                name="wall", scale=Scale(0.01, 4, 2), world=world
            )
            initial_shapes_count = len(wall.root.collision)

            aperture = Aperture.create_with_new_region_in_world(
                name="aperture", scale=Scale(0.1, 1, 1), world=world
            )
            wall.add(aperture)
        self.assertIn(aperture, wall.apertures)
        self.assertTrue(len(wall.root.collision) > initial_shapes_count)

    def test_create_with_connection_limits(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        limits = DegreeOfFreedomLimits.from_position_range_and_speed(
            lower_position=-0.5, upper_position=0.5
        )

        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            Door.create_with_new_body_in_world(
                name="door",
                world=world,
                parent_connection_specification=RevoluteConnectionSpecification(
                    dof_limits=limits, axis=Vector3.Z()
                ),
            )

        dof = world.degrees_of_freedom[0]
        self.assertEqual(dof.limits.lower.position, -0.5)
        self.assertEqual(dof.limits.upper.position, 0.5)

    def test_create_with_invalid_connection_limits(self):
        world = World.create_with_root_body("root")
        limits = DegreeOfFreedomLimits.from_position_range_and_speed(
            lower_position=0.5, upper_position=-0.5
        )

        with self.assertRaises(InvalidConnectionLimits), world.modify_world():
            Door.create_with_new_body_in_world(
                name="door",
                world=world,
                parent_connection_specification=RevoluteConnectionSpecification(
                    dof_limits=limits, axis=Vector3.Z()
                ),
            )

    def test_perceivable_cup(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            cup = Cup.create_with_new_body_in_world(name="cup", world=world)
        cup.class_label = "plastic_cup"
        self.assertEqual(cup.class_label, "plastic_cup")

    def test_is_storage_space(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            cabinet = Cabinet.create_with_new_body_in_world(
                name="cabinet", world=world, scale=Scale(0.5, 0.5, 1.0)
            )
            cup = Cup.create_with_new_body_in_world(name="cup", world=world)

            cabinet.add_object(cup)

        self.assertIn(cup, cabinet.objects)
        self.assertEqual(cup.root.parent_kinematic_structure_entity, cabinet.root)

    def test_has_objects_mismatching_world(self):
        world1 = World.create_with_root_body("root1")
        with world1.modify_world():
            cabinet = Cabinet.create_with_new_body_in_world(
                name="cabinet", world=world1, scale=Scale(0.5, 0.5, 1.0)
            )
        world2 = World.create_with_root_body("root2")
        with world2.modify_world():
            cup = Cup.create_with_new_body_in_world(name="cup", world=world2)

        with self.assertRaises(MismatchingWorld):
            cabinet.add_object(cup)

    def test_double_door_view_point(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            door_left = Door.create_with_new_body_in_world(
                name="door_left",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, y=0.5
                ),
            )
            door_right = Door.create_with_new_body_in_world(
                name="door_right",
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

    #################################################################
    # Characterization of the scale -> geometry generation.
    # These pin the geometry that create_with_new_body_in_world(scale=...)
    # currently produces, so the get_default_root_kinematic_structure_entity_specification /
    # get_default_root_kinematic_structure_entity_specification extraction (and the later factory
    # rewire) provably preserves it.
    #################################################################

    @staticmethod
    def _world_with_root() -> World:
        world = World.create_with_root_body("root")
        return world

    def test_characterize_base_body_geometry(self):
        world = self._world_with_root()
        with world.modify_world():
            milk = Milk.create_with_new_body_in_world(
                name="milk", world=world, scale=Scale(0.2, 0.3, 0.4)
            )
        collision = milk.root.collision
        # base path assigns one collection to both collision and visual
        self.assertIs(collision, milk.root.visual)
        self.assertEqual(len(collision), 1)
        np.testing.assert_allclose(
            collision.combined_mesh.bounds,
            [[-0.1, -0.15, -0.2], [0.1, 0.15, 0.2]],
        )

    def test_characterize_case_body_geometry(self):
        world = self._world_with_root()
        with world.modify_world():
            drawer = Drawer.create_with_new_body_in_world(
                name="drawer", world=world, scale=Scale(0.3, 0.4, 0.5)
            )
        collision = drawer.root.collision
        self.assertIs(collision, drawer.root.visual)
        # hollow container -> more than one box
        self.assertGreater(len(collision), 1)
        # outer extents still equal the scale
        np.testing.assert_allclose(
            collision.combined_mesh.bounds,
            [[-0.15, -0.2, -0.25], [0.15, 0.2, 0.25]],
        )

    def test_characterize_handle_geometry(self):
        world = self._world_with_root()
        with world.modify_world():
            handle = Handle.get_annotation_specification(
                "handle",
                Handle.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(0.1, 0.05, 0.05), thickness=0.01
                ),
            ).spawn(world)
        collision = handle.root.collision
        self.assertIs(collision, handle.root.visual)
        self.assertGreater(len(collision), 1)
        np.testing.assert_allclose(
            collision.combined_mesh.bounds,
            [[-0.1, -0.025, -0.025], [0.0, 0.025, 0.025]],
        )

    def test_characterize_door_geometry(self):
        world = self._world_with_root()
        with world.modify_world():
            door = Door.create_with_new_body_in_world(
                name="door", world=world, scale=Scale(0.03, 1, 2)
            )
        collision = door.root.collision
        self.assertIs(collision, door.root.visual)
        self.assertEqual(len(collision), 1)
        np.testing.assert_allclose(
            collision.combined_mesh.bounds,
            [[-0.015, -0.5, -1.0], [0.015, 0.5, 1.0]],
        )

    def test_characterize_door_invalid_plane(self):
        world = self._world_with_root()
        with self.assertRaises(InvalidPlaneDimensions):
            with world.modify_world():
                Door.create_with_new_body_in_world(
                    name="door", world=world, scale=Scale(2, 1, 1)
                )

    def test_characterize_floor_geometry(self):
        world = self._world_with_root()
        with world.modify_world():
            floor = Floor.create_with_new_body_in_world(
                name="floor", world=world, scale=Scale(2, 2, 0.1)
            )
        collision = floor.root.collision
        self.assertIs(collision, floor.root.visual)
        # floor is a single polytope mesh
        self.assertEqual(len(collision), 1)
        np.testing.assert_allclose(
            collision.combined_mesh.bounds,
            [[-1.0, -1.0, -0.05], [1.0, 1.0, 0.05]],
        )

    def test_characterize_wall_geometry(self):
        world = self._world_with_root()
        with world.modify_world():
            wall = Wall.create_with_new_body_in_world(
                name="wall", world=world, scale=Scale(0.1, 4, 2)
            )
        collision = wall.root.collision
        self.assertIs(collision, wall.root.visual)
        # wall event runs z from 0..scale.z, not centered
        np.testing.assert_allclose(
            collision.combined_mesh.bounds,
            [[-0.05, -2.0, 0.0], [0.05, 2.0, 2.0]],
        )

    def test_characterize_wall_invalid_plane(self):
        world = self._world_with_root()
        with self.assertRaises(InvalidPlaneDimensions):
            with world.modify_world():
                Wall.create_with_new_body_in_world(
                    name="wall", world=world, scale=Scale(2, 1, 1)
                )

    def test_characterize_aperture_region_geometry(self):
        world = self._world_with_root()
        with world.modify_world():
            aperture = Aperture.create_with_new_region_in_world(
                name="aperture", world=world, scale=Scale(0.1, 1, 2)
            )
        # region geometry lives on .area, not .collision
        area = aperture.root.area
        self.assertEqual(len(area), 1)
        np.testing.assert_allclose(
            area.combined_mesh.bounds,
            [[-0.05, -0.5, -1.0], [0.05, 0.5, 1.0]],
        )

    def test_microwave_factory(self):
        world = World.create_with_root_body("root")
        with world.modify_world():
            microwave = Microwave.create_with_new_body_in_world(
                name="microwave", world=world
            )
            door = Door.create_with_new_body_in_world(
                name="microwave_door",
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
        world = World.create_with_root_body("root")
        with world.modify_world():
            hood = Hood.create_with_new_body_in_world(name="hood", world=world)
            toaster = Toaster.create_with_new_body_in_world(name="toaster", world=world)
            coffee_machine = CoffeeMachine.create_with_new_body_in_world(
                name="coffee_machine", world=world
            )

        self.assertEqual(len(world.get_semantic_annotations_by_type(Hood)), 1)
        self.assertEqual(len(world.get_semantic_annotations_by_type(Toaster)), 1)
        self.assertEqual(len(world.get_semantic_annotations_by_type(CoffeeMachine)), 1)
        self.assertEqual(world.root, hood.root.parent_kinematic_structure_entity)
        self.assertEqual(world.root, toaster.root.parent_kinematic_structure_entity)
        self.assertEqual(
            world.root, coffee_machine.root.parent_kinematic_structure_entity
        )


@dataclass(eq=False)
class _AnnotationWithOverlappingPartWholeRelationshipFields(
    HasRootBody, PartWholeRelationship
):
    """
    Throwaway whole whose two part-whole relationship fields have overlapping element
    types (``DoorWithType`` is a subclass of ``Door``), so a ``DoorWithType`` matches
    both.
    """

    door: Optional[Door] = field(
        default=None,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    typed_door: Optional[DoorWithType] = field(
        default=None,
        metadata=IsPartWholeRelationship().as_dict(),
    )


def _world_with_root() -> World:
    world = World.create_with_root_body("root")
    return world


def test_add_routes_handle_as_child():
    """
    Add(handle) mounts the handle as a child of the door (default strategy).
    """
    world = _world_with_root()
    with world.modify_world():
        door = Door.create_with_new_body_in_world(
            name="door", scale=Scale(0.03, 1, 2), world=world
        )
        handle = Handle.create_with_new_body_in_world(name="handle", world=world)
        door.add(handle)

    assert door.handle == handle
    assert door.root == handle.root.parent_kinematic_structure_entity


def test_add_routes_plural_drawer_and_door():
    """
    Add() appends to the right list when the matching part-whole relationship field is
    plural.
    """
    world = _world_with_root()
    with world.modify_world():
        fridge = Fridge.create_with_new_body_in_world(
            name="fridge", world=world, scale=Scale(1, 1, 2.0)
        )
        drawer = Drawer.create_with_new_body_in_world(name="drawer", world=world)
        door = Door.create_with_new_body_in_world(
            name="door", scale=Scale(0.03, 1, 2), world=world
        )
        fridge.add(drawer)
        fridge.add(door)

    assert drawer in fridge.drawers
    assert door in fridge.doors
    assert drawer not in fridge.doors
    assert door not in fridge.drawers


def test_add_routes_aperture_with_cut():
    """
    Add(aperture) cuts the wall geometry and mounts the aperture
    (Aperture._mount_strategy).
    """
    world = _world_with_root()
    with world.modify_world():
        wall = Wall.create_with_new_body_in_world(
            name="wall", scale=Scale(0.1, 4, 2), world=world
        )
        door = Door.create_with_new_body_in_world(
            name="door", scale=Scale(0.03, 1, 2), world=world
        )
        aperture = Aperture.create_with_new_region_in_world_from_body(
            name="aperture", world=world, body=door.root
        )
        wall.add(aperture)

    assert wall.apertures[0] == aperture
    assert aperture.root.parent_kinematic_structure_entity == wall.root


def test_add_object_stores_occupants():
    """
    Containment occupants are stored via add_object (occupancy, not parthood).
    """
    world = _world_with_root()
    with world.modify_world():
        table = Table.create_with_new_body_in_world(
            name="table", world=world, scale=Scale(1.0, 1.0, 0.1)
        )
        milk = Milk.create_with_new_body_in_world(
            name="milk", world=world, scale=Scale(0.03, 0.03, 0.1)
        )
        cereal = Cereal.create_with_new_body_in_world(
            name="cereal", world=world, scale=Scale(0.1, 0.03, 0.2)
        )
        table.add_object(milk)
        table.add_object(cereal)

    assert milk in table.objects
    assert cereal in table.objects
    assert table.root == milk.root.parent_kinematic_structure_entity


def test_add_does_not_route_occupants():
    """
    An occupant matches no part-whole relationship field, so add() rejects it (it must
    use place).
    """
    world = _world_with_root()
    with world.modify_world():
        fridge = Fridge.create_with_new_body_in_world(
            name="fridge", world=world, scale=Scale(1, 1, 2.0)
        )
        milk = Milk.create_with_new_body_in_world(
            name="milk", world=world, scale=Scale(0.03, 0.03, 0.1)
        )
        with pytest.raises(CannotBeAPartOf):
            fridge.add(milk)
        fridge.add_object(milk)

    assert milk in fridge.objects


def test_add_rejects_unsupported_part_type():
    """
    Add() of a part type the annotation has no part-whole relationship field for raises
    CannotBeAPartOf.
    """
    world = _world_with_root()
    with world.modify_world():
        door = Door.create_with_new_body_in_world(
            name="door", scale=Scale(0.03, 1, 2), world=world
        )
        drawer = Drawer.create_with_new_body_in_world(name="drawer", world=world)
        # A Door has handle/entry way part-whole relationship fields but no drawer field.
        with pytest.raises(CannotBeAPartOf):
            door.add(drawer)


def test_add_raises_on_ambiguous_part():
    """
    Add() of a part matching more than one part-whole relationship field raises
    AmbiguousPart.
    """
    world = _world_with_root()
    with world.modify_world():
        whole = _AnnotationWithOverlappingPartWholeRelationshipFields.create_with_new_body_in_world(
            name="whole", world=world
        )
        typed_door = DoorWithType.create_with_new_body_in_world(
            name="typed_door", scale=Scale(0.03, 1, 2), world=world
        )
        # A DoorWithType is both a Door (door field) and a DoorWithType (typed_door field).
        with pytest.raises(AmbiguousPart):
            whole.add(typed_door)


def test_add_field_name_resolves_ambiguity_to_base_field():
    """
    Add(part, field_name=...) routes to the named field even when the type matches
    several.
    """
    world = _world_with_root()
    with world.modify_world():
        whole = _AnnotationWithOverlappingPartWholeRelationshipFields.create_with_new_body_in_world(
            name="whole", world=world
        )
        typed_door = DoorWithType.create_with_new_body_in_world(
            name="typed_door", scale=Scale(0.03, 1, 2), world=world
        )
        whole.add(typed_door, field_name="door")
    assert whole.door is typed_door
    assert whole.typed_door is None


def test_add_field_name_resolves_ambiguity_to_specific_field():
    """
    Add(part, field_name=...) can route the same part to the other matching field.
    """
    world = _world_with_root()
    with world.modify_world():
        whole = _AnnotationWithOverlappingPartWholeRelationshipFields.create_with_new_body_in_world(
            name="whole", world=world
        )
        typed_door = DoorWithType.create_with_new_body_in_world(
            name="typed_door", scale=Scale(0.03, 1, 2), world=world
        )
        whole.add(typed_door, field_name="typed_door")
    assert whole.typed_door is typed_door
    assert whole.door is None


def test_add_unknown_field_name_raises():
    """
    Add(part, field_name=...) with a name that is not a part-whole field raises.
    """
    world = _world_with_root()
    with world.modify_world():
        whole = _AnnotationWithOverlappingPartWholeRelationshipFields.create_with_new_body_in_world(
            name="whole", world=world
        )
        typed_door = DoorWithType.create_with_new_body_in_world(
            name="typed_door", scale=Scale(0.03, 1, 2), world=world
        )
        with pytest.raises(UnknownPartWholeRelationshipField):
            whole.add(typed_door, field_name="not_a_field")


def test_add_field_name_with_mismatching_type_raises():
    """
    Add(part, field_name=...) still type-checks: a part the named field rejects raises.
    """
    world = _world_with_root()
    with world.modify_world():
        door = Door.create_with_new_body_in_world(
            name="door", scale=Scale(0.03, 1, 2), world=world
        )
        handle = Handle.create_with_new_body_in_world(name="handle", world=world)
        # 'entry_way' is a real part-whole field of Door, but a Handle is not an EntryWay.
        with pytest.raises(CannotBeAPartOf):
            door.add(handle, field_name="entry_way")


def test_containment_only_annotation_has_no_add():
    """
    A pure-containment annotation (Table) exposes add_object but not the part-whole
    add().
    """
    assert not hasattr(Table, "add")
    assert hasattr(Table, "add_object")


# %% movable joints
# A part that moves relative to its whole says where its joint sits, so it can be mounted
# on one, and refuses to be moved while nothing moves it.


@pytest.mark.parametrize(
    "annotation_type, scale",
    [
        (Drawer, Scale(0.4, 0.5, 0.6)),
        (Elevator, Scale(2, 2, 2)),
        (BottleCap, Scale(0.02, 0.04, 0.04)),
    ],
)
def test_movable_joint_of_a_part_that_slides_or_screws_sits_at_its_origin(
    annotation_type, scale
):
    world = _world_with_root()
    with world.modify_world():
        part = annotation_type.create_with_new_body_in_world(
            name="part", world=world, scale=scale
        )

    np.testing.assert_allclose(
        part.calculate_self_T_movable_joint(Vector3.X()).to_np(),
        HomogeneousTransformationMatrix().to_np(),
    )


def test_drawer_mounts_on_its_movable_joint_where_it_stands():
    world = _world_with_root()
    with world.modify_world():
        drawer = Drawer.create_with_new_body_in_world(
            name="drawer",
            world=world,
            scale=Scale(0.4, 0.5, 0.6),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, yaw=0.3
            ),
        )
    world_T_drawer = drawer.root.global_transform

    joint = drawer.mount_on_movable_joint(
        PrismaticConnectionSpecification(axis=Vector3.X())
    )

    assert joint is drawer.movable_joint
    assert isinstance(joint, PrismaticConnection)
    np.testing.assert_allclose(
        drawer.root.global_transform.to_np(), world_T_drawer.to_np(), atol=1e-12
    )


def test_drawer_without_a_movable_joint_has_no_opening_ratio():
    world = _world_with_root()
    with world.modify_world():
        drawer = Drawer.create_with_new_body_in_world(
            name="drawer", world=world, scale=Scale(0.4, 0.5, 0.6)
        )

    with pytest.raises(MissingMovableJointError):
        drawer.opening_ratio


@pytest.mark.parametrize("operate_doors", [Elevator.open, Elevator.close])
def test_elevator_doors_without_a_movable_joint_cannot_be_operated(operate_doors):
    world = _world_with_root()
    with world.modify_world():
        elevator = Elevator.create_with_new_body_in_world(
            name="elevator", world=world, scale=Scale(2, 2, 2)
        )
        door = Door.create_with_new_body_in_world(
            name="door", world=world, scale=Scale(0.05, 1, 2)
        )
        elevator.add(door)

    with pytest.raises(MissingMovableJointError):
        operate_doors(elevator)


def test_elevator_without_a_movable_joint_cannot_drive():
    world = _world_with_root()
    with world.modify_world():
        elevator = Elevator.create_with_new_body_in_world(
            name="elevator", world=world, scale=Scale(2, 2, 2)
        )
        ground_floor = GroundFloor.create_with_new_region_in_world(
            name="ground_floor", world=world, scale=Scale(4, 4, 0.1)
        )

    with pytest.raises(MissingMovableJointError):
        elevator.drive_to_floor(ground_floor)


# %% an elevator's cabin floor

CABIN_SCALE = Scale(2, 2, 2)
"""
Outer size of the test elevator's cabin.
"""

CABIN_WALL_THICKNESS = 0.05
"""
Thickness of the test elevator cabin's walls, floor and roof.
"""


def test_an_elevator_supports_what_stands_on_its_cabin_floor():
    """
    What rides an elevator stands on the cabin's floor, not on its roof, so the
    supporting surface lies on the inside bottom of the cabin.
    """
    world = _world_with_root()
    with world.modify_world():
        elevator = Elevator.get_annotation_specification(
            "elevator",
            Elevator.get_default_root_kinematic_structure_entity_specification(
                scale=CABIN_SCALE, wall_thickness=CABIN_WALL_THICKNESS
            ),
        ).spawn(world)

    with world.modify_world():
        surface = elevator.calculate_supporting_surface()

    elevator_T_surface = world.transform(surface.global_pose, elevator.root)
    assert float(elevator_T_surface.z) == pytest.approx(
        -(CABIN_SCALE.z - CABIN_WALL_THICKNESS) / 2
    )


def test_an_elevator_stops_with_its_cabin_floor_flush_with_the_level(
    multi_story_building,
):
    """
    Something rolls in and out of the cabin only if the cabin's floor is level with the
    floor it stops at.
    """
    world = deepcopy(multi_story_building)
    elevator = world.get_semantic_annotations_by_type(Elevator)[0]
    first_floor = world.get_semantic_annotations_by_type(FirstFloor)[0]

    elevator.drive_to_floor(first_floor)
    world.notify_state_change()

    cabin_floor = elevator.require_supporting_surface()
    assert float(cabin_floor.global_pose.z) == pytest.approx(
        float(first_floor.floor_plane[0].z)
    )


if __name__ == "__main__":
    unittest.main()
