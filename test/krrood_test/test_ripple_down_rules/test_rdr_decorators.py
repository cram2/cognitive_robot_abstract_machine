import unittest
from .datasets import PhysicalObject, Part, Robot


class RDRDecoratorsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.robot = Robot("Robot1", parts=[Part("Part1"), Part("Part2")])
        cls.physical_object = PhysicalObject("Object1", [cls.robot])
        cls.part_3 = Part("Part3")

    def test_is_a_robot(self):
        self.physical_object._is_a_robot_rdr.fit = False
        self.assertFalse(self.physical_object.is_a_robot())
        self.assertTrue(self.robot.is_a_robot())

    def test_select_objects_that_are_parts_of_robots(self):
        self.physical_object._select_parts_rdr.fit = False
        selected_parts = self.physical_object.select_objects_that_are_parts_of_robot([self.part_3, self.robot,
                                                                                      *self.robot.parts], self.robot)
        for part in selected_parts:
            self.assertIn(part, self.robot.parts)
        self.assertNotIn(self.part_3, selected_parts)
        self.assertNotIn(self.robot, selected_parts)
