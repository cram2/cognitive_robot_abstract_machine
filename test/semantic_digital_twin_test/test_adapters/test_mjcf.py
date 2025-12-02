import os.path
import unittest

from semantic_digital_twin.adapters.mjcf import MJCFParser

from semantic_digital_twin.world_description.connections import FixedConnection


class MjcfParserTestCase(unittest.TestCase):
    mjcf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "semantic_digital_twin", "resources", "mjcf"
    )
    table_xml = os.path.join(mjcf_dir, "table.xml")
    kitchen_xml = os.path.join(mjcf_dir, "kitchen-small.xml")
    apartment_xml = os.path.join(mjcf_dir, "iai_apartment.xml")
    pr2_xml = os.path.join(mjcf_dir, "pr2_kinematic_tree.xml")

    def setUp(self):
        self.table_xml_parser = MJCFParser(self.table_xml)
        self.kitchen_xml_parser = MJCFParser(self.kitchen_xml)
        self.apartment_xml_parser = MJCFParser(self.apartment_xml)
        self.pr2_xml_parser = MJCFParser(self.pr2_xml)

    def test_table_parsing(self):
        body_num = 7
        world = self.table_xml_parser.parse()
        world.validate()
        self.assertTrue(len(world.kinematic_structure_entities) == body_num)

        origin_left_front_leg_joint = world.get_connection(
            world.root, world.kinematic_structure_entities[1]
        )
        self.assertIsInstance(origin_left_front_leg_joint, FixedConnection)

    def test_kitchen_parsing(self):
        world = self.kitchen_xml_parser.parse()
        world.validate()
        self.assertTrue(len(world.kinematic_structure_entities) > 0)
        self.assertTrue(len(world.connections) > 0)

    def test_apartment_parsing(self):
        world = self.apartment_xml_parser.parse()
        world.validate()
        self.assertTrue(len(world.kinematic_structure_entities) > 0)
        self.assertTrue(len(world.connections) > 0)

    def test_pr2_parsing(self):
        world = self.pr2_xml_parser.parse()
        world.validate()
        self.assertTrue(len(world.kinematic_structure_entities) > 0)
        self.assertTrue(len(world.connections) > 0)
        self.assertTrue(world.root.name.name == "world")


if __name__ == "__main__":
    unittest.main()
