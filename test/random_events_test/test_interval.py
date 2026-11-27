import math
import unittest

from krrood.adapters.json_serializer import to_json, from_json
from random_events.interval import *


class SimpleIntervalTestCase(unittest.TestCase):
    def test_intersection(self):
        a = SimpleInterval.from_data(lower=0, upper=1)
        b = SimpleInterval.from_data(lower=0.5, upper=2)
        c = SimpleInterval.from_data(
            lower=0.5, upper=0.75, left=Bound.OPEN, right=Bound.CLOSED
        )

        intersection_a_b = a.intersection_with(other=b)
        intersection_a_b_ = SimpleInterval.from_data(
            lower=0.5, upper=1, left=Bound.OPEN, right=Bound.OPEN
        )
        self.assertEqual(intersection_a_b, intersection_a_b_)

        intersection_a_c = a.intersection_with(other=c)
        self.assertEqual(intersection_a_c, c)

    def test_is_empty(self):
        a = SimpleInterval.from_data()
        b = SimpleInterval.from_data(lower=3, upper=1)
        c = SimpleInterval.from_data(lower=0, upper=1)
        self.assertTrue(a.is_empty())
        self.assertTrue(b.is_empty())
        self.assertFalse(c.is_empty())

    def test_complement(self):
        a = SimpleInterval.from_data()
        complement_a = a.complement()
        self.assertEqual(
            complement_a[0],
            SimpleInterval.from_data(
                lower=-float("inf"),
                upper=float("inf"),
                left=Bound.OPEN,
                right=Bound.OPEN,
            ),
        )
        b = SimpleInterval.from_data(lower=0, upper=1)
        complement_b = b.complement()
        self.assertSetEqual(
            {*complement_b},
            {
                SimpleInterval.from_data(
                    lower=-float("inf"), upper=0, left=Bound.OPEN, right=Bound.CLOSED
                ),
                SimpleInterval.from_data(
                    lower=1, upper=float("inf"), left=Bound.CLOSED, right=Bound.OPEN
                ),
            },
        )

    def test_contains(self):
        a = SimpleInterval.from_data(lower=0, upper=1)
        self.assertFalse(a.contains(item=0))
        self.assertTrue(a.contains(item=0.5))
        self.assertFalse(a.contains(item=1))
        self.assertFalse(a.contains(item=-1))
        self.assertFalse(a.contains(item=1.1))

    def test_to_json(self):
        a = SimpleInterval.from_data(lower=0, upper=1)
        b = from_json(to_json(a))
        self.assertIsInstance(b, SimpleInterval)
        self.assertEqual(a, b)

    def test_setter(self):
        a = SimpleInterval.from_data(lower=0, upper=1)
        a.lower = 0.5
        self.assertEqual(a.lower, 0.5)

    def test_eq(self):
        a = SimpleInterval.from_data(lower=0, upper=1)
        b = SimpleInterval.from_data(lower=0, upper=1)
        self.assertEqual(a, b)

    def test_size(self):
        self.assertEqual(SimpleInterval.from_data(lower=0, upper=2.5).size, 2.5)
        self.assertEqual(
            SimpleInterval.from_data(
                lower=1, upper=1, left=Bound.CLOSED, right=Bound.CLOSED
            ).size,
            0,
        )

    def test_size_of_empty_interval(self):
        self.assertEqual(SimpleInterval.from_data().size, 0)
        self.assertEqual(SimpleInterval.from_data(lower=3, upper=1).size, 0)

    def test_size_of_unbounded_interval(self):
        self.assertEqual(
            SimpleInterval.from_data(lower=0, upper=float("inf")).size, float("inf")
        )


class NearestContainedValueTestCase(unittest.TestCase):
    interval = SimpleInterval.from_data(
        lower=0, upper=1, left=Bound.OPEN, right=Bound.CLOSED
    )

    def test_a_contained_value_is_its_own_nearest(self):
        self.assertEqual(self.interval.nearest_contained_value(value=0.5), 0.5)

    def test_a_value_beyond_an_included_end_is_moved_onto_it(self):
        self.assertEqual(self.interval.nearest_contained_value(value=2.0), 1.0)

    def test_a_value_beyond_an_excluded_end_is_moved_just_inside_it(self):
        nearest = self.interval.nearest_contained_value(value=-1.0)
        self.assertEqual(nearest, math.nextafter(0.0, math.inf))
        self.assertTrue(self.interval.contains(item=nearest))


class IntervalTestCase(unittest.TestCase):
    def test_simplify(self):
        a = SimpleInterval.from_data(lower=0, upper=1)
        b = SimpleInterval.from_data(lower=0.5, upper=1.5)
        c = SimpleInterval.from_data(lower=1.5, upper=2, left=Bound.CLOSED)
        d = SimpleInterval.from_data(lower=3, upper=4)
        a_b = Interval.from_simple_sets(d, a, b, c)
        a_b_simplified = a_b.simplify()
        a_b_simplified_ = Interval.from_simple_sets(
            SimpleInterval.from_data(lower=0, upper=2),
            SimpleInterval.from_data(lower=3, upper=4),
        )
        self.assertEqual(a_b_simplified, a_b_simplified_)

    def test_intersection_with_self(self):
        a = SimpleInterval.from_data(lower=0, upper=1)
        b = SimpleInterval.from_data(lower=0.5, upper=1.5)
        c = SimpleInterval.from_data(lower=1.5, upper=2, left=Bound.CLOSED)
        d = SimpleInterval.from_data(lower=3, upper=4)
        a_d = Interval.from_simple_sets(a, d)
        b_c = Interval.from_simple_sets(b, c)

        intersection_a_d_b_c = a_d.intersection_with(other=b_c)
        intersection_expected = Interval.from_simple_sets(
            SimpleInterval.from_data(lower=0.5, upper=1)
        )
        self.assertEqual(intersection_a_d_b_c, intersection_expected)

    def test_complement(self):
        a = SimpleInterval.from_data(lower=0, upper=1)
        d = SimpleInterval.from_data(lower=3, upper=4)
        a_d = Interval.from_simple_sets(a, d)

        complement_a_d = a_d.complement()
        complement_a_d_ = Interval.from_simple_sets(
            SimpleInterval.from_data(
                lower=-float("inf"), upper=0, left=Bound.OPEN, right=Bound.CLOSED
            ),
            SimpleInterval.from_data(
                lower=1, upper=3, left=Bound.CLOSED, right=Bound.CLOSED
            ),
            SimpleInterval.from_data(
                lower=4, upper=float("inf"), left=Bound.CLOSED, right=Bound.OPEN
            ),
        )
        self.assertEqual(complement_a_d, complement_a_d_)

    def test_make_disjoint(self):
        a = SimpleInterval.from_data(
            lower=0, upper=1, left=Bound.CLOSED, right=Bound.CLOSED
        )
        b = SimpleInterval.from_data(
            lower=0.5, upper=1.5, left=Bound.CLOSED, right=Bound.CLOSED
        )
        c = SimpleInterval.from_data(
            lower=1.5, upper=2, left=Bound.CLOSED, right=Bound.CLOSED
        )
        d = SimpleInterval.from_data(
            lower=2, upper=3, left=Bound.CLOSED, right=Bound.CLOSED
        )
        a_b_c_d = Interval.from_simple_sets(a, b, c, d)

        disjoint = a_b_c_d.make_disjoint()

        a_b_c_d_expected = Interval.from_simple_sets(
            SimpleInterval.from_data(
                lower=0, upper=3, left=Bound.CLOSED, right=Bound.CLOSED
            )
        )
        self.assertEqual(disjoint, a_b_c_d_expected)

    def test_union(self):
        a = SimpleInterval.from_data(lower=0, upper=1)
        b = SimpleInterval.from_data(lower=0.5, upper=1.5)
        c = SimpleInterval.from_data(lower=1.5, upper=2, left=Bound.CLOSED)
        d = SimpleInterval.from_data(lower=3, upper=4)
        a_d = Interval.from_simple_sets(a, d)
        b_c = Interval.from_simple_sets(b, c)

        union_a_d_b_c = a_d.union_with(other=b_c)
        union_a_d_b_c_ = Interval.from_simple_sets(
            SimpleInterval.from_data(lower=0, upper=2),
            SimpleInterval.from_data(lower=3, upper=4),
        )
        self.assertSetEqual({*union_a_d_b_c}, {*union_a_d_b_c_})
        self.assertTrue(union_a_d_b_c.is_disjoint())

    def test_to_json(self):
        a = SimpleInterval.from_data(lower=0, upper=1)
        b = Interval.from_simple_sets(a)
        c = from_json(to_json(b))
        self.assertIsInstance(c, Interval)
        self.assertEqual(b, c)

    def test_contained_integers(self):
        a = open(2, 4) | closed_open(4.5, 6)
        self.assertEqual(list(a.contained_integers()), [3, 5])

    def test_size_sums_the_simple_intervals(self):
        a = closed(0, 1) | closed(3, 4.5)
        self.assertEqual(a.size, 2.5)

    def test_size_counts_overlapping_intervals_once(self):
        a = Interval.from_simple_sets(
            SimpleInterval.from_data(lower=0, upper=2),
            SimpleInterval.from_data(lower=1, upper=3),
        )
        self.assertEqual(a.size, 3)

    def test_size_of_the_reals(self):
        self.assertEqual(reals().size, float("inf"))


if __name__ == "__main__":
    unittest.main()
