from krrood.adapters.json_serializer import to_json, from_json
from random_events.set import SetElement, Set
import enum
import unittest

from random_events.sigma_algebra import AbstractSimpleSet


str_set = {"a", "c", "b"}
int_set = {1, 2, 3}
float_set = {1.001, 2.3845, 3.4345345}


class SetElementTestCase(unittest.TestCase):

    def test_intersection_with(self):
        a = SetElement.from_data("a", str_set)
        b = SetElement.from_data("b", str_set)

        intersection_a_b = a.intersection_with(b)
        self.assertTrue(intersection_a_b.is_empty())
        self.assertEqual(a.intersection_with(a), a)

    def test_complement(self):
        a = SetElement.from_data("a", str_set)
        b = SetElement.from_data("b", str_set)
        c = SetElement.from_data("c", str_set)
        # a1 = SetElement.from_data(1, int_set)
        # a2 = SetElement.from_data(2, int_set)
        # a3 = SetElement.from_data(3, int_set)
        complement_a = a.complement()
        complement_b = b.complement()
        complement_c = c.complement()
        # complement_a1 = a1.complement()
        self.assertEqual({*complement_a}, {b, c})
        self.assertEqual({*complement_b}, {a, c})
        self.assertEqual({*complement_c}, {a, b})
        # self.assertEqual(complement_a1, {a2, a3})

    def test_contains(self):
        a = SetElement.from_data("a", str_set)
        b = SetElement.from_data("b", str_set)
        c = SetElement.from_data("c", str_set)
        self.assertTrue(a.contains(a))
        self.assertFalse(a.contains(b))
        self.assertFalse(a.contains(c))

    def test_to_json(self):
        a = SetElement.from_data("a", str_set)
        b = from_json(to_json(a))
        self.assertEqual(a, b)
        self.assertEqual(a.all_elements, b.all_elements)


class SetTestCase(unittest.TestCase):

    def test_simplify(self):
        a = SetElement.from_data("a", str_set)
        b = SetElement.from_data("b", str_set)
        c = SetElement.from_data("c", str_set)
        s = Set.from_simple_sets(a, b, c, c)
        self.assertEqual(3, len(s.simple_sets))
        ss = Set.from_simple_sets(a, b, c)
        self.assertEqual(ss, s.simplify())

    def test_difference(self):
        a = SetElement.from_data("a", str_set)
        b = SetElement.from_data("b", str_set)
        s = Set.from_simple_sets(a, b)
        s_ = Set.from_simple_sets(a)
        self.assertEqual(s.difference_with(s_), Set.from_simple_sets(b))

    def test_complement(self):
        a = SetElement.from_data("a", str_set)
        b = SetElement.from_data("b", str_set)
        c = SetElement.from_data("c", str_set)
        s = Set.from_simple_sets(a, b)
        self.assertEqual(s.complement(), Set.from_simple_sets(c))

    def test_to_json(self):
        a = SetElement.from_data("a", str_set)
        b = SetElement.from_data("b", str_set)
        s = Set.from_simple_sets(a, b)
        s_ = from_json(to_json(s))
        self.assertEqual(s, s_)

    def test_from_iterable(self):
        s = Set.from_iterable(str_set)
        self.assertEqual(
            s,
            Set.from_simple_sets(
                SetElement.from_data("a", str_set),
                SetElement.from_data("b", str_set),
                SetElement.from_data("c", str_set),
            ),
        )

    def test_from_enum(self):
        class TestEnum(enum.IntEnum):
            A = 1
            B = 2

        s = Set.from_iterable(TestEnum)
        self.assertIsNotNone(s.__str__())

    def test_hashmap(self):
        a = SetElement.from_data("a", str_set)
        b = SetElement.from_data("b", str_set)
        s = Set.from_simple_sets(a, b)
        self.assertEqual(s.hash_map, {hash(a): "a", hash(b): "b", hash("c"): "c"})


class SetTypeTestCase(unittest.TestCase):
    def test_int(self):
        a = SetElement.from_data(1, int_set)
        b = SetElement.from_data(2, int_set)
        c = SetElement.from_data(3, int_set)

        s = Set.from_simple_sets(a, b, c)
        s2 = Set.from_simple_sets(a, b)
        inter = s.intersection_with(s2)
        self.assertEqual(inter, s2)

    def test_call_temporary_result(self):
        def make_set():
            x = {"a", "b"}
            return Set.from_iterable(x)

        self.assertEqual(len(make_set().simple_sets), 2)

    def test_str(self):
        a = SetElement.from_data("a", str_set)
        b = SetElement.from_data("b", str_set)
        c = SetElement.from_data("c", str_set)

        s = Set.from_simple_sets(a, b, c)
        s2 = Set.from_simple_sets(a, b)
        inter = s.intersection_with(s2)
        self.assertEqual(inter, s2)

    def test_float(self):
        a = SetElement.from_data(1.001, float_set)
        b = SetElement.from_data(2.3845, float_set)
        c = SetElement.from_data(3.4345345, float_set)

        s = Set.from_simple_sets(a, b, c)
        s2 = Set.from_simple_sets(a, b)
        inter = s.intersection_with(s2)
        self.assertEqual(inter, s2)


if __name__ == "__main__":
    unittest.main()
