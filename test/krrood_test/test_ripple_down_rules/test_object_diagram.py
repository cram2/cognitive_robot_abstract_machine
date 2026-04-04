import os
import unittest

from .datasets import load_zoo_dataset, Species
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.user_interface.object_diagram import generate_object_graph


class Address:
    def __init__(self, city):
        self.city = city


class Person:
    def __init__(self, name, address):
        self.name = name
        self.address = address


class ObjectDiagramTestCase(unittest.TestCase):
    """Test case for generating object diagrams of the ripple down rules package."""

    cases: list
    targets: list
    # cq: CaseQuery
    person: Person

    @classmethod
    def setUpClass(cls):
        cls.cases, cls.targets = load_zoo_dataset(
            cache_file=f"{os.path.dirname(__file__)}/test_results/zoo"
        )
        cls.cq = CaseQuery(
            cls.cases[0], "species", (Species,), True, _target=cls.targets[0]
        )
        cls.person = Person("Ahmed", Address("Cairo"))

    def test_generate_person_diagram(self):
        # Generate object diagram for the Person instance
        graph = generate_object_graph(self.person)
        graph.render(
            f"{os.path.dirname(__file__)}/test_results/person_object_diagram",
            view=False,
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.path.dirname(__file__)}/test_results/person_object_diagram.svg"
            )
        )

    def test_generate_case_query_diagram(self):
        # Generate object diagram for the CaseQuery instance
        graph = generate_object_graph(self.cq)
        graph.render(
            f"{os.path.dirname(__file__)}/test_results/case_query_object_diagram",
            view=False,
        )
        self.assertTrue(
            os.path.isfile(
                f"{os.path.dirname(__file__)}/test_results/case_query_object_diagram.svg"
            )
        )
