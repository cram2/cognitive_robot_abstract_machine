import unittest

from random_events.variable import Continuous
from sortedcontainers import SortedSet

from semantic_digital_twin.datastructures.variables import SpatialVariables


class TestSpatialVariables(unittest.TestCase):
    def test_enum_members_exist(self):

        self.assertTrue(hasattr(SpatialVariables, "x"))
        self.assertTrue(hasattr(SpatialVariables, "y"))
        self.assertTrue(hasattr(SpatialVariables, "z"))

        self.assertEqual(SpatialVariables.x.value, Continuous("x"))
        self.assertEqual(SpatialVariables.y.value, Continuous("y"))
        self.assertEqual(SpatialVariables.z.value, Continuous("z"))

        # Distinct members and distinct underlying values
        self.assertNotEqual(SpatialVariables.x, SpatialVariables.y)
        self.assertNotEqual(SpatialVariables.x, SpatialVariables.z)
        self.assertNotEqual(SpatialVariables.y, SpatialVariables.z)

        values = [
            SpatialVariables.x.value,
            SpatialVariables.y.value,
            SpatialVariables.z.value,
        ]
        self.assertEqual(
            len(set(values)), 3, "Underlying values for x, y, z should be distinct"
        )

    def test_xy_contains_correct_members(self):
        xy = SpatialVariables.xy
        expected = {SpatialVariables.x.value, SpatialVariables.y.value}
        self.assertEqual(set(xy), expected, "xy should contain exactly x and y values")
        if SortedSet is not None:
            self.assertIsInstance(xy, SortedSet)

    def test_xz_contains_correct_members(self):
        xz = SpatialVariables.xz
        expected = {SpatialVariables.x.value, SpatialVariables.z.value}
        self.assertEqual(set(xz), expected, "xz should contain exactly x and z values")
        if SortedSet is not None:
            self.assertIsInstance(xz, SortedSet)

    def test_yz_contains_correct_members(self):
        yz = SpatialVariables.yz
        expected = {SpatialVariables.y.value, SpatialVariables.z.value}
        self.assertEqual(set(yz), expected, "yz should contain exactly y and z values")
        if SortedSet is not None:
            self.assertIsInstance(yz, SortedSet)


if __name__ == "__main__":
    unittest.main()
