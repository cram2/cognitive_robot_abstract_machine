import unittest
from enum import IntEnum

import numpy as np

from krrood.adapters.json_serializer import from_json, to_json
from random_events.interval import *
from random_events.product_algebra import SimpleEvent
from random_events.variable import *

str_set = {"a", "c", "b"}


class ContinuousTestCase(unittest.TestCase):
    x = Continuous("x")

    def test_creation(self):
        self.assertEqual(self.x.name, "x")
        self.assertEqual(self.x.domain, reals())

    def test_to_json(self):
        x_ = from_json(to_json(self.x))
        self.assertEqual(self.x, x_)


class IntegerTestCase(unittest.TestCase):

    def test_creation(self):
        x = Integer("x")
        self.assertEqual(x.name, "x")
        self.assertEqual(x.domain, reals())


class SymbolicTestCase(unittest.TestCase):

    def test_creation(self):
        a = SetElement.from_data("a", str_set)
        b = SetElement.from_data("b", str_set)
        c = SetElement.from_data("c", str_set)
        x = Symbolic(name="x", domain=Set.from_simple_sets(a, b, c))
        self.assertEqual(x.name, "x")
        self.assertEqual(x.domain, Set.from_simple_sets(a, b, c))

    def test_to_json(self):
        a = SetElement.from_data("a", str_set)
        b = SetElement.from_data("b", str_set)
        c = SetElement.from_data("c", str_set)
        x = Symbolic(name="x", domain=Set.from_simple_sets(a, b, c))
        x_ = from_json(to_json(x))
        self.assertEqual(x, x_)

    def test_empty_domain(self):
        def make_variable():
            domain_cls = IntEnum("Domain", {"A": 1, "B": 2})
            x = Symbolic(name="x", domain=Set.from_iterable(domain_cls))
            return x

        result = make_variable()
        self.assertEqual(len(result.domain.simple_sets), 2)


@dataclass(eq=False)
class InheritedContinuous(Continuous):
    mean: int

class InheritanceTestCase(unittest.TestCase):
    def test_conversion(self):
        v1 = InheritedContinuous(name="david", mean=2)
        event = SimpleEvent.from_data({v1: open_closed(-np.inf, 0)}).as_composite_set()
        event2 = event.complement()
        v2 = event2.variables[0]
        self.assertIsInstance(v2, InheritedContinuous)


class HashableTestClass:

    value: int

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)


class CustomObjectSetTestCase(unittest.TestCase):

    def test_variables(self):
        e1 = HashableTestClass(1)
        e2 = HashableTestClass(2)
        e3 = HashableTestClass(3)
        x = Symbolic(name="x", domain=Set.from_iterable({e1, e2, e3}))
        self.assertEqual(x.name, "x")
        se = SimpleEvent.from_data({x: e1}).as_composite_set()
        self.assertFalse(se.is_empty())


if __name__ == "__main__":
    unittest.main()
