import unittest

import plotly.graph_objects as go
from sortedcontainers import SortedSet

from random_events.interval import *
from random_events.product_algebra import SimpleEvent, Event
from random_events.set import SetElement, Set
from random_events.sigma_algebra import AbstractSimpleSet
from random_events.variable import Continuous, Symbolic

str_set = {"a", "c", "b"}
str_set_domain = Set.from_iterable(str_set)


class EventTestCase(unittest.TestCase):
    x = Continuous("x")
    y = Continuous("y")
    z = Continuous("z")
    a = Symbolic("a", str_set_domain)
    b = Symbolic("b", str_set_domain)

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
        sa = SetElement("a", str_set)
        sb = SetElement("b", str_set)
        event = SimpleEvent(
            {
                self.a: Set(sa),
                self.x: Interval(SimpleInterval(0, 1)),
                self.y: Interval(SimpleInterval(0, 1)),
            }
        )

        self.assertEqual(event[self.x], Interval(SimpleInterval(0, 1)))
        self.assertEqual(event[self.y], Interval(SimpleInterval(0, 1)))
        self.assertEqual(event[self.a], Set(sa))

        self.assertFalse(event.is_empty())

    def test_intersection_with(self):
        sa = SetElement("a", str_set)
        sb = SetElement("b", str_set)
        sc = SetElement("c", str_set)
        event_1 = SimpleEvent(
            {
                self.a: Set(sa, sb),
                self.x: Interval(SimpleInterval(0, 1)),
                self.y: Interval(SimpleInterval(0, 1)),
            }
        )
        event_2 = SimpleEvent(
            {self.a: Set(sa), self.x: Interval(SimpleInterval(0.5, 1))}
        )
        event_3 = SimpleEvent({self.a: Set(sc)})
        intersection = event_1.intersection_with(event_2)

        intersection_ = SimpleEvent(
            {
                self.a: Set(sa),
                self.x: Interval(SimpleInterval(0.5, 1)),
                self.y: Interval(SimpleInterval(0, 1)),
            }
        )

        self.assertEqual(intersection, intersection_)
        self.assertNotEqual(intersection, event_1)

        second_intersection = event_1.intersection_with(event_3)
        self.assertTrue(second_intersection.is_empty())

    def test_complement(self):
        sa = SetElement("a", str_set)
        sb = SetElement("b", str_set)
        sc = SetElement("c", str_set)
        event = SimpleEvent(
            {
                self.a: Set(sa, sb),
                self.x: Interval(SimpleInterval(0, 1)),
                self.y: self.y.domain,
            }
        )
        complement = event.complement()
        self.assertEqual(len(complement), 2)
        complement_1 = SimpleEvent(
            {self.a: Set(sc), self.x: self.x.domain, self.y: self.y.domain}
        )
        complement_2 = SimpleEvent(
            {
                self.a: event[self.a],
                self.x: event[self.x].complement(),
                self.y: self.y.domain,
            }
        )
        e_c = Event(complement_1, complement_2)
        self.assertEqual(e_c, Event(*complement))

    def test_simplify(self):
        sa = SetElement("a", str_set)
        sb = SetElement("b", str_set)
        sc = SetElement("c", str_set)
        event_1 = SimpleEvent(
            {
                self.a: Set(sa, sb),
                self.x: Interval(SimpleInterval(0, 1)),
                self.y: Interval(SimpleInterval(0, 1)),
            }
        )
        event_2 = SimpleEvent(
            {
                self.a: Set(sc),
                self.x: Interval(SimpleInterval(0, 1)),
                self.y: Interval(SimpleInterval(0, 1)),
            }
        )
        event = Event(event_1, event_2)
        simplified = event.simplify()
        self.assertEqual(len(simplified.simple_sets), 1)

        result = Event(
            SimpleEvent(
                {
                    self.a: self.a.domain,
                    self.x: Interval(SimpleInterval(0, 1)),
                    self.y: Interval(SimpleInterval(0, 1)),
                }
            )
        )
        self.assertEqual(simplified, result)

    def test_to_json(self):
        sa = SetElement("a", str_set)
        sb = SetElement("b", str_set)
        sc = SetElement("c", str_set)
        event = SimpleEvent(
            {
                self.a: Set(sa, sb),
                self.x: Interval(SimpleInterval(0, 1)),
                self.y: Interval(SimpleInterval(0, 1)),
            }
        )
        event_ = AbstractSimpleSet.from_json(event.to_json())
        self.assertEqual(event_, event)

    def test_plot_2d(self):
        event_1 = SimpleEvent(
            {
                self.x: Interval(SimpleInterval(0, 1)),
                self.y: Interval(SimpleInterval(0, 1)),
            }
        )
        event_2 = SimpleEvent(
            {
                self.x: Interval(SimpleInterval(1, 2)),
                self.y: Interval(SimpleInterval(1, 2)),
            }
        )
        event = Event(event_1, event_2)
        fig = go.Figure(event.plot(), event.plotly_layout())
        self.assertIsNotNone(fig)  # fig.show()

    def test_plot_3d(self):
        event_1 = SimpleEvent(
            {
                self.x: SimpleInterval(0, 1),
                self.y: SimpleInterval(0, 1),
                self.z: SimpleInterval(0, 1),
            }
        )
        event_2 = SimpleEvent(
            {
                self.x: SimpleInterval(1, 2),
                self.y: SimpleInterval(1, 2),
                self.z: SimpleInterval(1, 2),
            }
        )
        event = Event(event_1, event_2)
        fig = go.Figure(event.plot(), event.plotly_layout())
        self.assertIsNotNone(fig)  # fig.show()

    def test_union(self):
        sa = SetElement("a", str_set)
        sb = SetElement("b", str_set)
        sc = SetElement("c", str_set)
        event = Event(SimpleEvent({self.a: sa, self.x: open(-float("inf"), 2)}))
        second_event = SimpleEvent(
            {self.a: Set(sa, sb), self.x: open(1, 4)}
        ).as_composite_set()
        union = event | second_event
        result = Event(
            SimpleEvent({self.a: sa, self.x: open(-float("inf"), 4)}),
            SimpleEvent({self.a: sb, self.x: open(1, 4)}),
        )
        self.assertEqual(union, result)

    def test_marginal_event(self):
        event_1 = SimpleEvent(
            {self.x: closed(0, 1), self.y: Interval(SimpleInterval(0, 1))}
        )
        event_2 = SimpleEvent(
            {self.x: closed(1, 2), self.y: Interval(SimpleInterval(3, 4))}
        )
        event_3 = SimpleEvent(
            {self.x: closed(5, 6), self.y: Interval(SimpleInterval(5, 6))}
        )
        event = Event(event_1, event_2, event_3)
        marginal = event.marginal({self.x})
        self.assertEqual(
            marginal,
            SimpleEvent({self.x: closed(0, 2) | closed(5, 6)}).as_composite_set(),
        )
        fig = go.Figure(marginal.plot())  # fig.show()

    def test_marginal_event_symbolic(self):
        a = Symbolic("a", str_set_domain)
        event = (
            SimpleEvent({self.a: "a", self.b: "b"}).as_composite_set()
            | SimpleEvent({self.a: "b", self.b: "b"}).as_composite_set()
        )
        e_a = event.marginal({a})
        self.assertEqual(e_a, SimpleEvent({self.a: ("a", "b")}).as_composite_set())

    def test_variable_comparison(self):
        a1 = Symbolic("a", str_set_domain)
        a2 = Symbolic("a", str_set_domain)
        self.assertEqual(a1, a2)
        self.assertEqual(a1._cpp_object, a2._cpp_object)

    def test_to_json_multiple_events(self):
        sa = SetElement("a", str_set)
        sb = SetElement("b", str_set)
        # sc = SetElement("c", str_set)
        event = SimpleEvent(
            {self.x: closed(0, 1), self.y: SimpleInterval(3, 5), self.a: Set(sa, sb)}
        ).as_composite_set()
        event_ = AbstractSimpleSet.from_json(event.to_json())
        self.assertEqual(event_, event)

    def test_bounding_box(self):
        event_1 = SimpleEvent(
            {self.x: closed(0, 1), self.y: SimpleInterval(0, 1)}
        ).as_composite_set()
        event_2 = SimpleEvent(
            {self.x: closed(1, 2), self.y: Interval(SimpleInterval(3, 4))}
        ).as_composite_set()
        event = event_1 | event_2

        event_before_bounding_box = event.__deepcopy__()
        bounding_box = event.bounding_box()
        result = SimpleEvent(
            {
                self.x: closed(0, 2),
                self.y: SimpleInterval(0, 1).as_composite_set()
                | SimpleInterval(3, 4).as_composite_set(),
            }
        )
        self.assertEqual(bounding_box, result)
        self.assertEqual(event, event_before_bounding_box)

    def test_complex_event_bounding_box_with_references(self):
        event1 = SimpleEvent(
            {self.x: closed(0, 1) | closed(2, 3), self.y: closed(0, 1) | closed(2, 3)}
        ).as_composite_set()
        event2 = SimpleEvent(
            {self.x: closed(1, 2) | closed(3, 4), self.y: closed(1, 2) | closed(3, 4)}
        ).as_composite_set()
        event = event1 | event2
        event_before_bb = event.__deepcopy__()
        bb = event.bounding_box()
        self.assertEqual(event, event_before_bb)

    def test_setitem(self):
        event = SimpleEvent()
        event[self.a] = "a"
        self.assertEqual(event[self.a], SetElement("a", str_set).as_composite_set())
        event[self.a] = ("a", "b")
        self.assertEqual(
            event[self.a],
            SetElement("a", str_set).as_composite_set()
            | SetElement("b", str_set).as_composite_set(),
        )

        with self.assertRaises(ValueError):
            event[self.a] = 1

    def test_fill_missing_variables(self):
        e = SimpleEvent({self.x: closed(0, 1) | closed(3, 4)}).as_composite_set()
        y = Continuous("y")
        e.fill_missing_variables((y,))
        self.assertTrue(y in e.variables)

    def test_fill_missing_variables_pure(self):
        e = SimpleEvent({self.x: closed(0, 1) | closed(3, 4)}).as_composite_set()
        y = Continuous("y")
        e = e.fill_missing_variables_pure((y,))
        self.assertTrue(y in e.variables)

    def test_update_variables(self):
        e = (
            SimpleEvent({self.x: closed(0, 1), self.y: closed(3, 4)})
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
        empty_event = SimpleEvent(
            {self.x: SimpleInterval(0, 0), self.y: SimpleInterval(0, 0)}
        ).as_composite_set()
        event = SimpleEvent(
            {self.x: SimpleInterval(0, 1), self.y: SimpleInterval(0, 1)}
        ).as_composite_set()
        union = empty_event.union_with(event)
        union2 = event.union_with(empty_event)
        self.assertEqual(union, event)
        self.assertEqual(union2, event)

    def test_union_different_variables(self):
        simple_event1 = SimpleEvent({self.x: closed(0, 1)})
        simple_event2 = SimpleEvent({self.y: closed(3, 4)})
        event_1 = Event(simple_event1)
        event_2 = Event(simple_event2)

        event_1.fill_missing_variables(event_2.variables)
        event_2.fill_missing_variables(event_1.variables)

        union = event_1.union_with(event_2)

        exp_se1 = SimpleEvent(
            {self.x: closed(0, 1), self.y: open(float("-inf"), float("inf"))}
        )
        exp_se2 = SimpleEvent(
            {self.x: open(float("-inf"), float("inf")), self.y: closed(3, 4)}
        )
        exp_result = Event(exp_se1, exp_se2).make_disjoint()

        self.assertEqual(union, exp_result)

    def test_difference_with_empty_set(self):
        event = SimpleEvent(
            {self.x: SimpleInterval(0, 1), self.y: SimpleInterval(0, 1)}
        ).as_composite_set()
        empty_event = Event()
        diff = event.difference_with(empty_event)
        self.assertEqual(diff, event)


if __name__ == "__main__":
    unittest.main()
