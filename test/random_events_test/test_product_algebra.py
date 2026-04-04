import unittest

import plotly.graph_objects as go
from sortedcontainers import SortedSet

from krrood.adapters.json_serializer import to_json, from_json
from random_events.interval import *
from random_events.product_algebra import SimpleEvent, Event
from random_events.set import SetElement, Set
from random_events.sigma_algebra import AbstractSimpleSet
from random_events.variable import Continuous, Symbolic

str_set = {"a", "c", "b"}
str_set_domain = Set.from_iterable(str_set)


class EventTestCase(unittest.TestCase):
    x = Continuous(name="x")
    y = Continuous(name="y")
    z = Continuous(name="z")
    a = Symbolic(name="a", domain=str_set_domain)
    b = Symbolic(name="b", domain=str_set_domain)

    # def setUp(self):
    #     print("=" * 80)
    #     print(str_set)
    #     print(str_set_domain)
    #
    # def tearDown(self):
    #     print(str_set)
    #     print(str_set_domain)
    #     print("=" * 80)

    def test_constructor(self):
        sa = SetElement.from_data("a", str_set)
        sb = SetElement.from_data("b", str_set)
        event = SimpleEvent.from_data(
            {
                self.a: Set.from_simple_sets(sa),
                self.x: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
                self.y: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
            }
        )

        self.assertEqual(event[self.x], Interval.from_simple_sets(SimpleInterval.from_data(0, 1)))
        self.assertEqual(event[self.y], Interval.from_simple_sets(SimpleInterval.from_data(0, 1)))
        self.assertEqual(event[self.a], Set.from_simple_sets(sa))

        self.assertFalse(event.is_empty())

    def test_intersection_with(self):
        sa = SetElement.from_data("a", str_set)
        sb = SetElement.from_data("b", str_set)
        sc = SetElement.from_data("c", str_set)
        event_1 = SimpleEvent.from_data(
            {
                self.a: Set.from_simple_sets(sa, sb),
                self.x: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
                self.y: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
            }
        )
        event_2 = SimpleEvent.from_data(
            {self.a: Set.from_simple_sets(sa), self.x: Interval.from_simple_sets(SimpleInterval.from_data(0.5, 1))}
        )
        event_3 = SimpleEvent.from_data({self.a: Set.from_simple_sets(sc)})
        intersection = event_1.intersection_with(event_2)

        intersection_ = SimpleEvent.from_data(
            {
                self.a: Set.from_simple_sets(sa),
                self.x: Interval.from_simple_sets(SimpleInterval.from_data(0.5, 1)),
                self.y: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
            }
        )

        self.assertEqual(intersection, intersection_)
        self.assertNotEqual(intersection, event_1)

        second_intersection = event_1.intersection_with(event_3)
        self.assertTrue(second_intersection.is_empty())

    def test_complement(self):
        sa = SetElement.from_data("a", str_set)
        sb = SetElement.from_data("b", str_set)
        sc = SetElement.from_data("c", str_set)
        event = SimpleEvent.from_data(
            {
                self.a: Set.from_simple_sets(sa, sb),
                self.x: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
                self.y: self.y.domain,
            }
        )
        complement = event.complement()
        self.assertEqual(len(complement), 2)
        complement_1 = SimpleEvent.from_data(
            {self.a: Set.from_simple_sets(sc), self.x: self.x.domain, self.y: self.y.domain}
        )
        complement_2 = SimpleEvent.from_data(
            {
                self.a: event[self.a],
                self.x: event[self.x].complement(),
                self.y: self.y.domain,
            }
        )
        e_c = Event.from_simple_sets(complement_1, complement_2)
        self.assertEqual(e_c, Event.from_simple_sets(*complement))

    def test_simplify(self):
        sa = SetElement.from_data("a", str_set)
        sb = SetElement.from_data("b", str_set)
        sc = SetElement.from_data("c", str_set)
        event_1 = SimpleEvent.from_data(
            {
                self.a: Set.from_simple_sets(sa, sb),
                self.x: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
                self.y: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
            }
        )
        event_2 = SimpleEvent.from_data(
            {
                self.a: Set.from_simple_sets(sc),
                self.x: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
                self.y: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
            }
        )
        event = Event.from_simple_sets(event_1, event_2)
        simplified = event.simplify()
        self.assertEqual(len(simplified.simple_sets), 1)

        result = Event.from_simple_sets(
            SimpleEvent.from_data(
                {
                    self.a: self.a.domain,
                    self.x: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
                    self.y: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
                }
            )
        )
        self.assertEqual(simplified, result)

    def test_to_json(self):
        sa = SetElement.from_data("a", str_set)
        sb = SetElement.from_data("b", str_set)
        event = SimpleEvent.from_data(
            {
                self.a: Set.from_simple_sets(sa, sb),
                self.x: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
                self.y: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
            }
        )
        event_ = from_json(to_json(event))
        self.assertEqual(event_, event)

    def test_plot_2d(self):
        event_1 = SimpleEvent.from_data(
            {
                self.x: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
                self.y: Interval.from_simple_sets(SimpleInterval.from_data(0, 1)),
            }
        )
        event_2 = SimpleEvent.from_data(
            {
                self.x: Interval.from_simple_sets(SimpleInterval.from_data(1, 2)),
                self.y: Interval.from_simple_sets(SimpleInterval.from_data(1, 2)),
            }
        )
        event = Event.from_simple_sets(event_1, event_2)
        fig = go.Figure(event.plot(), event.plotly_layout())
        self.assertIsNotNone(fig)  # fig.show()

    def test_plot_3d(self):
        event_1 = SimpleEvent.from_data(
            {
                self.x: SimpleInterval.from_data(0, 1),
                self.y: SimpleInterval.from_data(0, 1),
                self.z: SimpleInterval.from_data(0, 1),
            }
        )
        event_2 = SimpleEvent.from_data(
            {
                self.x: SimpleInterval.from_data(1, 2),
                self.y: SimpleInterval.from_data(1, 2),
                self.z: SimpleInterval.from_data(1, 2),
            }
        )
        event = Event.from_simple_sets(event_1, event_2)
        fig = go.Figure(event.plot(), event.plotly_layout())
        self.assertIsNotNone(fig)  # fig.show()

    def test_union(self):
        sa = SetElement.from_data("a", str_set)
        sb = SetElement.from_data("b", str_set)
        sc = SetElement.from_data("c", str_set)
        event = Event.from_simple_sets(SimpleEvent.from_data({self.a: sa, self.x: open(-float("inf"), 2)}))
        second_event = SimpleEvent.from_data(
            {self.a: Set.from_simple_sets(sa, sb), self.x: open(1, 4)}
        ).as_composite_set()
        union = event | second_event
        result = Event.from_simple_sets(
            SimpleEvent.from_data({self.a: sa, self.x: open(-float("inf"), 4)}),
            SimpleEvent.from_data({self.a: sb, self.x: open(1, 4)}),
        )
        self.assertEqual(union, result)

    def test_marginal_event(self):
        event_1 = SimpleEvent.from_data(
            {self.x: closed(0, 1), self.y: Interval.from_simple_sets(SimpleInterval.from_data(0, 1))}
        )
        event_2 = SimpleEvent.from_data(
            {self.x: closed(1, 2), self.y: Interval.from_simple_sets(SimpleInterval.from_data(3, 4))}
        )
        event_3 = SimpleEvent.from_data(
            {self.x: closed(5, 6), self.y: Interval.from_simple_sets(SimpleInterval.from_data(5, 6))}
        )
        event = Event.from_simple_sets(event_1, event_2, event_3)
        marginal = event.marginal({self.x})
        self.assertEqual(
            marginal,
            SimpleEvent.from_data({self.x: closed(0, 2) | closed(5, 6)}).as_composite_set(),
        )
        fig = go.Figure(marginal.plot())  # fig.show()

    def test_marginal_event_symbolic(self):
        a = Symbolic(name="a", domain=str_set_domain)
        event = (
            SimpleEvent.from_data({self.a: "a", self.b: "b"}).as_composite_set()
            | SimpleEvent.from_data({self.a: "b", self.b: "b"}).as_composite_set()
        )
        e_a = event.marginal({a})
        self.assertEqual(e_a, SimpleEvent.from_data({self.a: ("a", "b")}).as_composite_set())

    def test_variable_comparison(self):
        a1 = Symbolic(name="a", domain=str_set_domain)
        a2 = Symbolic(name="a", domain=str_set_domain)
        self.assertEqual(a1, a2)
        self.assertEqual(a1.cpp_object, a2.cpp_object)

    def test_to_json_multiple_events(self):
        sa = SetElement.from_data("a", str_set)
        sb = SetElement.from_data("b", str_set)
        # sc = SetElement.from_data("c", str_set)
        event = SimpleEvent.from_data(
            {self.x: closed(0, 1), self.y: SimpleInterval.from_data(3, 5), self.a: Set.from_simple_sets(sa, sb)}
        ).as_composite_set()
        event_ = from_json(to_json(event))
        self.assertEqual(event_, event)

    def test_bounding_box(self):
        event_1 = SimpleEvent.from_data(
            {self.x: closed(0, 1), self.y: SimpleInterval.from_data(0, 1)}
        ).as_composite_set()
        event_2 = SimpleEvent.from_data(
            {self.x: closed(1, 2), self.y: Interval.from_simple_sets(SimpleInterval.from_data(3, 4))}
        ).as_composite_set()
        event = event_1 | event_2

        event_before_bounding_box = event.__deepcopy__()
        bounding_box = event.bounding_box()
        result = SimpleEvent.from_data(
            {
                self.x: closed(0, 2),
                self.y: SimpleInterval.from_data(0, 1).as_composite_set()
                | SimpleInterval.from_data(3, 4).as_composite_set(),
            }
        )
        self.assertEqual(bounding_box, result)
        self.assertEqual(event, event_before_bounding_box)

    def test_complex_event_bounding_box_with_references(self):
        event1 = SimpleEvent.from_data(
            {self.x: closed(0, 1) | closed(2, 3), self.y: closed(0, 1) | closed(2, 3)}
        ).as_composite_set()
        event2 = SimpleEvent.from_data(
            {self.x: closed(1, 2) | closed(3, 4), self.y: closed(1, 2) | closed(3, 4)}
        ).as_composite_set()
        event = event1 | event2
        event_before_bb = event.__deepcopy__()
        bb = event.bounding_box()
        self.assertEqual(event, event_before_bb)

    def test_setitem(self):
        event = SimpleEvent.from_data()
        event[self.a] = "a"
        self.assertEqual(event[self.a], SetElement.from_data("a", str_set).as_composite_set())
        event[self.a] = ("a", "b")
        self.assertEqual(
            event[self.a],
            SetElement.from_data("a", str_set).as_composite_set()
            | SetElement.from_data("b", str_set).as_composite_set(),
        )

        with self.assertRaises(ValueError):
            event[self.a] = 1

    def test_fill_missing_variables(self):
        e = SimpleEvent.from_data({self.x: closed(0, 1) | closed(3, 4)}).as_composite_set()
        y = Continuous("y")
        e.fill_missing_variables((y,))
        self.assertTrue(y in e.variables)

    def test_fill_missing_variables_pure(self):
        e = SimpleEvent.from_data({self.x: closed(0, 1) | closed(3, 4)}).as_composite_set()
        y = Continuous("y")
        e = e.fill_missing_variables_pure((y,))
        self.assertTrue(y in e.variables)

    def test_update_variables(self):
        e = (
            SimpleEvent.from_data({self.x: closed(0, 1), self.y: closed(3, 4)})
            .as_composite_set()
            .complement()
        )
        y2 = Continuous("y2")
        e2 = e.update_variables({self.y: y2})
        self.assertEqual(e2.variables, SortedSet([self.x, y2]))


class OperationsWithEmptySetsTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")
    y: Continuous = Continuous("y")

    def test_empty_union(self):
        empty_event = SimpleEvent.from_data(
            {self.x: SimpleInterval.from_data(0, 0), self.y: SimpleInterval.from_data(0, 0)}
        ).as_composite_set()
        event = SimpleEvent.from_data(
            {self.x: SimpleInterval.from_data(0, 1), self.y: SimpleInterval.from_data(0, 1)}
        ).as_composite_set()
        union = empty_event.union_with(event)
        union2 = event.union_with(empty_event)
        self.assertEqual(union, event)
        self.assertEqual(union2, event)

    def test_union_different_variables(self):
        simple_event1 = SimpleEvent.from_data({self.x: closed(0, 1)})
        simple_event2 = SimpleEvent.from_data({self.y: closed(3, 4)})
        event_1 = Event.from_simple_sets(simple_event1)
        event_2 = Event.from_simple_sets(simple_event2)

        event_1.fill_missing_variables(event_2.variables)
        event_2.fill_missing_variables(event_1.variables)

        union = event_1.union_with(event_2)

        exp_se1 = SimpleEvent.from_data(
            {self.x: closed(0, 1), self.y: open(float("-inf"), float("inf"))}
        )
        exp_se2 = SimpleEvent.from_data(
            {self.x: open(float("-inf"), float("inf")), self.y: closed(3, 4)}
        )
        exp_result = Event.from_simple_sets(exp_se1, exp_se2).make_disjoint()

        self.assertEqual(union, exp_result)

    def test_difference_with_empty_set(self):
        event = SimpleEvent.from_data(
            {self.x: SimpleInterval.from_data(0, 1), self.y: SimpleInterval.from_data(0, 1)}
        ).as_composite_set()
        empty_event = Event()
        diff = event.difference_with(empty_event)
        self.assertEqual(diff, event)


if __name__ == "__main__":
    unittest.main()
