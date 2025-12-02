import os.path
import unittest

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world_description.connections import FixedConnection


class URDFParserTestCase(unittest.TestCase):
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "semantic_digital_twin", "resources", "urdf"
    )
    table = os.path.join(urdf_dir, "table.urdf")
    kitchen = os.path.join(urdf_dir, "kitchen-small.urdf")
    apartment = os.path.join(urdf_dir, "apartment.urdf")
    pr2 = os.path.join(urdf_dir, "pr2_kinematic_tree.urdf")

    def setUp(self):
        self.table_parser = URDFParser.from_file(file_path=self.table)
        self.kitchen_parser = URDFParser.from_file(file_path=self.kitchen)
        self.apartment_parser = URDFParser.from_file(file_path=self.apartment)
        self.pr2_parser = URDFParser.from_file(file_path=self.pr2)

    def test_table_parsing(self):
        world = self.table_parser.parse()
        world.validate()
        self.assertEqual(len(world.kinematic_structure_entities), 6)

        origin_left_front_leg_joint = world.get_connection(
            world.root, world.kinematic_structure_entities[1]
        )
        self.assertIsInstance(origin_left_front_leg_joint, FixedConnection)

    def test_kitchen_parsing(self):
        world = self.kitchen_parser.parse()
        world.validate()
        self.assertTrue(len(world.kinematic_structure_entities) > 0)
        self.assertTrue(len(world.connections) > 0)

    def test_apartment_parsing(self):
        world = self.apartment_parser.parse()
        world.validate()
        self.assertTrue(len(world.kinematic_structure_entities) > 0)
        self.assertTrue(len(world.connections) > 0)

    def test_pr2_parsing(self):
        world = self.pr2_parser.parse()
        world.validate()
        self.assertTrue(len(world.kinematic_structure_entities) > 0)
        self.assertTrue(len(world.connections) > 0)
        self.assertTrue(world.root.name.name == "base_footprint")

    def test_mimic_joints(self):
        world = self.pr2_parser.parse()
        joint_to_be_mimiced = world.get_connection_by_name("l_gripper_l_finger_joint")
        mimic_joint = world.get_connection_by_name("l_gripper_r_finger_joint")

        self.assertEqual(joint_to_be_mimiced.dofs, mimic_joint.dofs)


if __name__ == "__main__":
    unittest.main()
