import json
import os
import unittest
from dataclasses import asdict

import numpy as np
from pkg_resources import resource_filename
from sqlalchemy.orm import Session

from krrood.ormatic.utils import create_engine
from semantic_digital_twin.adapters.procthor.procthor_parser import (
    ProcTHORParser,
    ProcthorRoom,
    unity_to_semantic_digital_twin_transform,
    ProcthorDoor,
    ProcthorWall,
    ProcthorObject,
)
from semantic_digital_twin.adapters.procthor.procthor_resolver import (
    ProcthorResolver,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bread,
    Floor,
    Door,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Region, Body


class ProcTHORTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        json_dir = os.path.join(
            resource_filename("semantic_digital_twin", "../../"),
            "resources",
            "procthor_json",
        )
        cls.file_path = os.path.join(json_dir, "house_test.json")
        with open(cls.file_path) as f:
            cls.house_json = json.load(f)

        cls.world = World(name="TestWorld")
        world_root = Body(name=PrefixedName("TestWorldRoot"))
        with cls.world.modify_world():
            cls.world.add_kinematic_structure_entity(world_root)

    def test_unity_to_semantic_digital_twin_transform_identity_matrix(self):
        m = np.eye(4)
        result = unity_to_semantic_digital_twin_transform(
            HomogeneousTransformationMatrix(data=m)
        )
        np.testing.assert_allclose(result.to_np(), np.eye(4), rtol=1e-6, atol=1e-6)

    def test_unity_to_semantic_digital_twin_transform_translation_along_x(self):
        """Unity +X should map to semantic –Y (because of reflection)."""
        m = np.eye(4)
        m[0, 3] = 1.0
        result = unity_to_semantic_digital_twin_transform(
            HomogeneousTransformationMatrix(data=m)
        )
        self.assertAlmostEqual(result.to_position().to_np()[1], -1.0)
        np.testing.assert_allclose(
            result.to_rotation_matrix().to_np()[:3, :3], np.eye(3)
        )

    def test_unity_to_semantic_digital_twin_transform_translation_along_z(self):
        """Unity +Z should map to semantic +X."""
        m = np.eye(4)
        m[2, 3] = 2.0
        result = unity_to_semantic_digital_twin_transform(
            HomogeneousTransformationMatrix(data=m)
        )
        self.assertAlmostEqual(result.to_position().to_np()[0], 2.0, places=6)
        np.testing.assert_allclose(
            result.to_rotation_matrix().to_np()[:3, :3], np.eye(3)
        )

    def test_unity_to_semantic_digital_twin_transform_rotation_y_90_degrees(self):
        """Unity +90° about Y should become –90° about Z in semantic frame."""
        theta = np.pi / 2
        m = np.eye(4)
        m[:3, :3] = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
        result = unity_to_semantic_digital_twin_transform(
            HomogeneousTransformationMatrix(data=m)
        )

        expected = np.eye(4)
        expected[:3, :3] = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(result.to_np(), expected, rtol=1e-6, atol=1e-6)

    def test_room_centered_polytope(self):
        room = self.house_json["rooms"][0]
        procthor_room = ProcthorRoom(room_dict=room)
        point_array = np.asarray(
            [point.to_np()[:3] for point in procthor_room.centered_polytope]
        )
        self.assertTrue(np.allclose(point_array.mean(axis=0), np.zeros(3), atol=1e-6))

    def test_world_T_room(self):
        room = self.house_json["rooms"][0]
        procthor_room = ProcthorRoom(room_dict=room)
        np.testing.assert_array_equal(
            procthor_room.world_T_room.to_rotation_matrix().to_np(), np.eye(4)
        )
        np.testing.assert_array_equal(
            procthor_room.world_T_room.to_translation_matrix().to_np()[:3, 3],
            np.array([1.5, -1.5, 0]),
        )

    def test_room_get_world(self):
        room = self.house_json["rooms"][0]
        procthor_room = ProcthorRoom(room_dict=room)
        procthor_room.add_to_world(self.world)
        self.assertEqual(
            self.world.root.child_kinematic_structure_entities[0].name.name,
            "Kitchen_4_floor",
        )

        with self.world.modify_world():
            supporting_surface = self.world.get_semantic_annotations_by_type(Floor)[
                0
            ].calculate_supporting_surface()

        queried_supporting_surface = self.world.get_kinematic_structure_entity_by_name(
            "Kitchen_4_floor_supporting_surface_region"
        )
        self.assertEqual(supporting_surface, queried_supporting_surface)
        self.assertIsInstance(queried_supporting_surface, Region)
        region_area_mesh = queried_supporting_surface.area[0]
        self.assertAlmostEqual(region_area_mesh.mesh.area, 18.06)

    def test_door_polygon(self):
        door = self.house_json["doors"][0]
        procthor_door = ProcthorDoor(
            door_dict=door,
            parent_wall_width=0.05,
            thickness=0.03,
            world_T_parent_wall=HomogeneousTransformationMatrix(),
        )

        bounds = (
            procthor_door.x_min,
            procthor_door.x_max,
            procthor_door.y_min,
            procthor_door.y_max,
        )
        expected_bounds = (0.25, 2.25, 0.0, 2.1)
        np.testing.assert_allclose(bounds, expected_bounds, rtol=1e-6, atol=1e-6)

    def test_door_scale(self):
        door = self.house_json["doors"][0]
        procthor_door = ProcthorDoor(
            door_dict=door,
            parent_wall_width=0.05,
            thickness=0.03,
            world_T_parent_wall=HomogeneousTransformationMatrix(),
        )

        scale = procthor_door.scale
        expected_scale = Scale(0.03, 2.0, 2.1)
        self.assertEqual(expected_scale, scale)

    def test_wall_T_door(self):
        door = self.house_json["doors"][0]
        procthor_door = ProcthorDoor(
            door_dict=door,
            parent_wall_width=0.05,
            thickness=0.03,
            world_T_parent_wall=HomogeneousTransformationMatrix(),
        )

        expected_rotation = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, -1.225],
                [0, 0, 1, 1.05],
                [0, 0, 0, 1],
            ]
        )
        np.testing.assert_allclose(
            procthor_door.wall_T_door.to_np(),
            expected_rotation,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_door_add_to_world(self):
        door = self.house_json["doors"][0]
        procthor_door = ProcthorDoor(
            door_dict=door,
            parent_wall_width=0.05,
            thickness=0.03,
            world_T_parent_wall=HomogeneousTransformationMatrix(),
        )

        double_door = procthor_door.add_to_world(self.world)
        doors = self.world.get_semantic_annotations_by_type(Door)

        self.assertEqual(double_door.name.name, "Doorway_Double_7_room1_room4")
        self.assertEqual(len(doors), 2)

    def test_wall_creation(self):
        parser = ProcTHORParser.from_file(self.file_path, None)
        doors = self.house_json["doors"]
        walls = self.house_json["walls"]

        procthor_wall = ProcthorWall(wall_dicts=walls, door_dicts=doors)

        procthor_walls_from_door, used_wall_ids = parser._build_procthor_wall_from_door(
            walls, doors
        )

        self.assertEqual(procthor_walls_from_door, [procthor_wall])

        procthor_walls_from_polygon = parser._build_procthor_wall_from_polygon(walls)[0]

        self.assertFalse(procthor_walls_from_polygon.door_dicts)

        self.assertEqual(
            asdict(procthor_walls_from_door[0], dict_factory=dict),
            {
                **asdict(procthor_walls_from_polygon, dict_factory=dict),
                "door_dicts": procthor_walls_from_door[0].door_dicts,
            },
        )

    def test_wall_coords(self):
        doors = self.house_json["doors"]
        walls = self.house_json["walls"]

        procthor_wall = ProcthorWall(wall_dicts=walls, door_dicts=doors)

        x_coords, y_coords, z_coords = (
            procthor_wall.x_coords,
            procthor_wall.y_coords,
            procthor_wall.z_coords,
        )

        np.testing.assert_array_equal(x_coords, [3.5])
        np.testing.assert_array_equal(y_coords, [0.0, 4.2])
        np.testing.assert_array_equal(z_coords, [3.6, 0.0])
        self.assertEqual(procthor_wall.delta_x, 0.0)
        self.assertEqual(procthor_wall.delta_z, 3.6)

    def test_wall_scale(self):
        doors = self.house_json["doors"]
        walls = self.house_json["walls"]

        procthor_wall = ProcthorWall(
            wall_dicts=walls, door_dicts=doors, wall_thickness=0.05
        )

        self.assertEqual(procthor_wall.scale, Scale(0.05, 3.6, 4.2))

    def test_world_T_wall(self):
        doors = self.house_json["doors"]
        walls = self.house_json["walls"]

        procthor_wall = ProcthorWall(
            wall_dicts=walls, door_dicts=doors, wall_thickness=0.05
        )

        expected_translation = np.array(
            [
                [0, 1, 0, 1.8],
                [-1, 0, 0, -3.5],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        np.testing.assert_allclose(
            procthor_wall.world_T_wall.to_np(),
            expected_translation,
            rtol=1e-6,
            atol=1e-6,
        )

    @unittest.skip("Requires Database, TBD")
    def test_world_T_obj(self):
        objects = self.house_json["objects"][0]

        semantic_digital_twin_database_uri = os.environ.get(
            "semantic_digital_twin_DATABASE_URI"
        )

        # Create database engine and session
        engine = create_engine(semantic_digital_twin_database_uri)
        session = Session(engine)

        procthor_object = ProcthorObject(object_dict=objects, session=session)

        desired = np.array(
            [
                [-1.0, 0.0, 0.0, 6.7],
                [0.0, -1.0, 0.0, -6.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_almost_equal(
            procthor_object.world_T_obj.to_np(),
            desired,
            decimal=6,
        )

    @unittest.skip("Requires Database, TBD")
    def test_object_get_world(self):
        objects = self.house_json["objects"][0]

        semantic_digital_twin_database_uri = os.environ.get(
            "SEMANTIC_DIGITAL_TWIN_DATABASE_URI"
        )

        # Create database engine and session
        engine = create_engine(semantic_digital_twin_database_uri)
        session = Session(engine)

        procthor_object = ProcthorObject(object_dict=objects, session=session)
        world = procthor_object.get_world()
        ...

    @unittest.skip("Requires Database, TBD")
    def test_duplicate_object(self):
        semantic_digital_twin_database_uri = os.environ.get(
            "SEMANTIC_DIGITAL_TWIN_DATABASE_URI"
        )

        # Create database engine and session
        engine = create_engine(semantic_digital_twin_database_uri)
        session = Session(engine)

        house = self.house_json
        objects = house["objects"]

        procthor_object = ProcthorObject(object_dict=objects[0], session=session)
        obj_world = procthor_object.get_world()

        world = ProcTHORParser(
            "test_house", house=self.house_json, session=session
        ).parse()

        world.merge_world(obj_world)

        ...

    def test_parse_full_world(self):
        world = ProcTHORParser.from_file(
            os.path.join(
                resource_filename("semantic_digital_twin", "../../"),
                "resources",
                "procthor_json",
                "house_987654321.json",
            )
        ).parse()

        assert world is not None

    def test_procthor_views(self):
        """
        Simple krrood_test case to check that the ProcthorResolver works correctly with the additional_names attribute.
        """
        resolver = ProcthorResolver()
        resolver.classes = [Bread]

        resolved_string = resolver.resolve("whitebread_a_slice_8_mesh")
        self.assertEqual(Bread, resolved_string)


if __name__ == "__main__":
    unittest.main()
