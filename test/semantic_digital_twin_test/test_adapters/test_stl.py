import os
import unittest

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.world_description.geometry import FileMesh


class STLAdapterTestCase(unittest.TestCase):

    def setUp(self):
        # Set up any necessary resources or state before each test
        self.milk_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "stl", "milk.stl"
        )
        self.cup = os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "stl", "jeroen_cup.stl"
        )

    def test_stl_parsing_construct(self):
        stl_parser = STLParser(self.milk_path)
        self.assertEqual(stl_parser.file_path, self.milk_path)

    def test_stl_parsing(self):
        stl_parser = STLParser(self.milk_path)
        world = stl_parser.parse()
        world.validate()

        self.assertEqual(len(world.kinematic_structure_entities), 1)
        self.assertEqual(len(world.kinematic_structure_entities[0].collision), 1)
        self.assertEqual(len(world.kinematic_structure_entities[0].visual), 1)
        self.assertEqual(
            FileMesh, type(world.kinematic_structure_entities[0].collision[0])
        )

    def test_parse_and_merge(self):
        milk_world = STLParser(self.milk_path).parse()
        cup_world = STLParser(self.cup).parse()

        milk_world.merge_world(cup_world)
        self.assertEqual(len(milk_world.kinematic_structure_entities), 2)
        self.assertEqual(len(milk_world.kinematic_structure_entities[0].collision), 1)
