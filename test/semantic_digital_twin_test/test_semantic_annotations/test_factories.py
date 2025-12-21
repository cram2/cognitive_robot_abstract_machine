import unittest

import pytest
import rclpy

from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import InvalidPlaneDimensions
from semantic_digital_twin.semantic_annotations.mixins import HasCaseAsMainBody

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Door,
    Drawer,
    Dresser,
    Wall,
    Hinge,
    DoubleDoor,
    Fridge,
    Slider,
    Floor,
    Aperture,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    TransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body


class TestFactories(unittest.TestCase):
    def test_handle_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        returned_handle = Handle.create_with_new_body_in_world(
            name=PrefixedName("handle"),
            scale=Scale(0.1, 0.2, 0.03),
            thickness=0.03,
            world=world,
            parent=root,
        )
        semantic_handle_annotations = world.get_semantic_annotations_by_type(Handle)
        self.assertEqual(len(semantic_handle_annotations), 1)

        queried_handle: Handle = semantic_handle_annotations[0]
        self.assertEqual(returned_handle, queried_handle)
        self.assertEqual(
            world.root, queried_handle.body.parent_kinematic_structure_entity
        )

    def test_basic_has_body_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        returned_hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"),
            world=world,
            parent=root,
        )
        returned_slider = Slider.create_with_new_body_in_world(
            name=PrefixedName("slider"),
            world=world,
            parent=root,
        )
        semantic_hinge_annotations = world.get_semantic_annotations_by_type(Hinge)
        self.assertEqual(len(semantic_hinge_annotations), 1)

        queried_hinge: Hinge = semantic_hinge_annotations[0]
        self.assertEqual(returned_hinge, queried_hinge)
        self.assertEqual(
            world.root, queried_hinge.body.parent_kinematic_structure_entity
        )
        semantic_slider_annotations = world.get_semantic_annotations_by_type(Slider)
        self.assertEqual(len(semantic_slider_annotations), 1)
        queried_slider: Slider = semantic_slider_annotations[0]
        self.assertEqual(returned_slider, queried_slider)
        self.assertEqual(
            world.root, queried_slider.body.parent_kinematic_structure_entity
        )

    def test_door_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        returned_door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world, parent=root
        )
        semantic_door_annotations = world.get_semantic_annotations_by_type(Door)
        self.assertEqual(len(semantic_door_annotations), 1)

        queried_door: Door = semantic_door_annotations[0]
        self.assertEqual(returned_door, queried_door)
        self.assertEqual(
            world.root, queried_door.body.parent_kinematic_structure_entity
        )

    def test_door_factory_invalid(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with pytest.raises(InvalidPlaneDimensions):
            Door.create_with_new_body_in_world(
                name=PrefixedName("door"),
                scale=Scale(1, 1, 2),
                world=world,
                parent=root,
            )

        with pytest.raises(InvalidPlaneDimensions):
            Door.create_with_new_body_in_world(
                name=PrefixedName("door"),
                scale=Scale(1, 2, 1),
                world=world,
                parent=root,
            )

    def test_has_hinge_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world, parent=root
        )
        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"), world=world, parent=root
        )
        assert len(world.kinematic_structure_entities) == 3

        door.add_hinge(hinge)

        assert door.body.parent_kinematic_structure_entity == hinge.body
        assert isinstance(hinge.body.parent_connection, RevoluteConnection)
        assert door.hinge == hinge

    def test_reverse_has_hinge_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"), scale=Scale(0.03, 1, 2), world=world, parent=root
        )
        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"), world=world, parent=door.body
        )
        assert len(world.kinematic_structure_entities) == 3

        door.add_hinge(hinge)

        assert door.body.parent_kinematic_structure_entity == hinge.body
        assert isinstance(hinge.body.parent_connection, RevoluteConnection)
        assert door.hinge == hinge

    def test_has_handle_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)

        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"),
            scale=Scale(0.03, 1, 2),
            world=world,
            parent=root,
        )

        handle = Handle.create_with_new_body_in_world(
            name=PrefixedName("handle"),
            world=world,
            parent=root,
        )
        assert len(world.kinematic_structure_entities) == 3

        assert root == handle.body.parent_kinematic_structure_entity

        door.add_handle(handle)

        assert door.body == handle.body.parent_kinematic_structure_entity
        assert door.handle == handle

    def test_double_door_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        left_door = Door.create_with_new_body_in_world(
            name=PrefixedName("left_door"), world=world, parent=root
        )
        right_door = Door.create_with_new_body_in_world(
            name=PrefixedName("right_door"), world=world, parent=root
        )
        double_door = DoubleDoor.create_with_left_right_door_in_world(
            left_door, right_door
        )
        semantic_double_door_annotations = world.get_semantic_annotations_by_type(
            DoubleDoor
        )

        self.assertEqual(len(semantic_double_door_annotations), 1)

    def test_case_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        fridge = Fridge.create_with_new_body_in_world(
            name=PrefixedName("case"),
            world=world,
            parent=root,
            scale=Scale(1, 1, 2.0),
        )

        assert isinstance(fridge, HasCaseAsMainBody)

        semantic_container_annotations = world.get_semantic_annotations_by_type(Fridge)
        self.assertEqual(len(semantic_container_annotations), 1)

        assert len(world.get_semantic_annotations_by_type(HasCaseAsMainBody)) == 1

    def test_drawer_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"),
            world=world,
            parent=root,
            scale=Scale(0.2, 0.3, 0.2),
        )
        assert isinstance(drawer, HasCaseAsMainBody)
        semantic_drawer_annotations = world.get_semantic_annotations_by_type(Drawer)
        self.assertEqual(len(semantic_drawer_annotations), 1)

    def test_has_slider_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"),
            world=world,
            parent=root,
            scale=Scale(0.2, 0.3, 0.2),
        )
        slider = Slider.create_with_new_body_in_world(
            name=PrefixedName("slider"), world=world, parent=root
        )
        assert len(world.kinematic_structure_entities) == 3

        drawer.add_slider(slider)

        assert drawer.body.parent_kinematic_structure_entity == slider.body
        assert isinstance(slider.body.parent_connection, PrismaticConnection)
        assert drawer.slider == slider

    def test_reverse_has_slider_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"),
            scale=Scale(0.2, 0.3, 0.2),
            world=world,
            parent=root,
        )
        slider = Slider.create_with_new_body_in_world(
            name=PrefixedName("slider"), world=world, parent=drawer.body
        )
        assert len(world.kinematic_structure_entities) == 3

        drawer.add_slider(slider)

        assert drawer.body.parent_kinematic_structure_entity == slider.body
        assert isinstance(slider.body.parent_connection, PrismaticConnection)
        assert drawer.slider == slider

    def test_has_drawer_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        fridge = Fridge.create_with_new_body_in_world(
            name=PrefixedName("case"),
            world=world,
            parent=root,
            scale=Scale(1, 1, 2.0),
        )
        drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"), world=world, parent=fridge.body
        )
        fridge.add_drawer(drawer)

        semantic_drawer_annotations = world.get_semantic_annotations_by_type(Drawer)
        self.assertEqual(len(semantic_drawer_annotations), 1)
        assert fridge.drawers[0] == drawer

    def test_has_doors_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        fridge = Fridge.create_with_new_body_in_world(
            name=PrefixedName("case"),
            world=world,
            parent=root,
            scale=Scale(1, 1, 2.0),
        )
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("left_door"), world=world, parent=fridge.body
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
        floor = Floor.create_with_new_body_in_world(
            name=PrefixedName("floor"),
            world=world,
            parent=root,
            scale=Scale(5, 5, 0.01),
        )
        semantic_floor_annotations = world.get_semantic_annotations_by_type(Floor)
        self.assertEqual(len(semantic_floor_annotations), 1)

    def test_wall_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        wall = Wall.create_with_new_body_in_world(
            name=PrefixedName("wall"),
            scale=Scale(0.1, 4, 2),
            world=world,
            parent=root,
        )
        semantic_wall_annotations = world.get_semantic_annotations_by_type(Wall)
        self.assertEqual(len(semantic_wall_annotations), 1)

    def test_aperture_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        aperture = Aperture.create_with_new_region_in_world(
            name=PrefixedName("wall"),
            scale=Scale(0.1, 4, 2),
            world=world,
            parent=root,
        )
        semantic_aperture_annotations = world.get_semantic_annotations_by_type(Aperture)
        self.assertEqual(len(semantic_aperture_annotations), 1)

    def test_aperture_from_body_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"),
            scale=Scale(0.03, 1, 2),
            world=world,
            parent=root,
        )
        aperture = Aperture.create_with_new_region_in_world_from_body(
            name=PrefixedName("wall"),
            world=world,
            parent=root,
            body=door.body,
        )
        semantic_aperture_annotations = world.get_semantic_annotations_by_type(Aperture)
        self.assertEqual(len(semantic_aperture_annotations), 1)

    def test_has_aperture_factory(self):
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        wall = Wall.create_with_new_body_in_world(
            name=PrefixedName("wall"),
            scale=Scale(0.1, 4, 2),
            world=world,
            parent=root,
        )
        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"),
            scale=Scale(0.03, 1, 2),
            world=world,
            parent=root,
        )
        aperture = Aperture.create_with_new_region_in_world_from_body(
            name=PrefixedName("wall"),
            world=world,
            parent=root,
            body=door.body,
        )
        wall.add_aperture(aperture)

        assert wall.apertures[0] == aperture
        assert aperture.region.parent_kinematic_structure_entity == wall.body


if __name__ == "__main__":
    unittest.main()
