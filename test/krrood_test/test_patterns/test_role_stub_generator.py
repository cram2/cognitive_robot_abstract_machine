import inspect
import os
from dataclasses import dataclass, fields, is_dataclass
from typing import Dict, Any, Type

import pytest

from krrood.patterns.role_stub_generator import RoleStubGenerator
from test.krrood_test.dataset import (
    university_ontology_like_classes_without_descriptors,
)


def execute_stub(stub_content: str, name: str = "__stub__") -> Dict[str, Any]:
    """
    Executes the stub content and returns the namespace.
    """
    import sys
    import types

    # Create a real module object and put it in sys.modules
    # This is required for get_type_hints to work correctly with forward references.
    module = types.ModuleType(name)
    module.__dict__.update({"__name__": name})
    sys.modules[name] = module
    namespace = module.__dict__

    # Use a fake Symbol class to prevent stub classes from being picked up by SymbolGraph
    # and to avoid introspection issues during test cleanup/setup.
    class Symbol:
        pass

    namespace["Symbol"] = Symbol

    # Mock the Symbol class in its original modules so the stub's imports use our fake one.
    import unittest.mock

    with unittest.mock.patch("krrood.entity_query_language.predicate.Symbol", Symbol):
        with unittest.mock.patch("krrood.symbol_graph.symbol_graph.Symbol", Symbol):
            exec(stub_content, namespace)
    return namespace


@dataclass
class StubComparator:
    """
    Compares two stubs by executing them and comparing the resulting class objects.
    """

    generated_namespace: Dict[str, Any]
    expected_namespace: Dict[str, Any]

    def _get_dataclasses(self, namespace: Dict[str, Any]) -> Dict[str, Type]:
        return {
            name: obj
            for name, obj in namespace.items()
            if inspect.isclass(obj)
            and is_dataclass(obj)
            and obj.__module__ == namespace.get("__name__")
        }

    def compare_class_existence(self):
        """
        Verifies that all classes in the expected stub exist in the generated stub.
        """
        gen_classes = self._get_dataclasses(self.generated_namespace)
        exp_classes = self._get_dataclasses(self.expected_namespace)
        assert set(gen_classes.keys()) == set(
            exp_classes.keys()
        ), f"Missing classes: {set(exp_classes.keys()) - set(gen_classes.keys())}. Extra classes: {set(gen_classes.keys()) - set(exp_classes.keys())}"

    def compare_class_hierarchy(self):
        """
        Verifies that class bases match between stubs.
        """
        gen_classes = self._get_dataclasses(self.generated_namespace)
        exp_classes = self._get_dataclasses(self.expected_namespace)
        for name, exp_cls in exp_classes.items():
            gen_cls = gen_classes[name]
            assert [base.__name__ for base in gen_cls.__bases__] == [
                base.__name__ for base in exp_cls.__bases__
            ]

    def compare_field_details(self):
        """
        Verifies that all fields, their types, and arguments match.
        """
        gen_classes = self._get_dataclasses(self.generated_namespace)
        exp_classes = self._get_dataclasses(self.expected_namespace)
        for name, exp_cls in exp_classes.items():
            gen_cls = gen_classes[name]
            self._compare_fields(gen_cls, exp_cls)

    def _compare_fields(self, gen_cls: Type, exp_cls: Type):
        gen_fields = {f.name: f for f in fields(gen_cls)}
        exp_fields = {f.name: f for f in fields(exp_cls)}

        assert set(gen_fields.keys()) == set(
            exp_fields.keys()
        ), f"Fields of {gen_cls.__name__} mismatch"

        for name, exp_field in exp_fields.items():
            gen_field = gen_fields[name]
            for attr in ["type", "init", "default", "default_factory", "kw_only"]:
                assert str(getattr(gen_field, attr)) == str(
                    getattr(exp_field, attr)
                ), f"{attr} of {gen_cls.__name__}.{name} mismatch"

    def compare_dataclass_params(self):
        """
        Verifies that @dataclass decorator arguments match.
        """
        gen_classes = self._get_dataclasses(self.generated_namespace)
        exp_classes = self._get_dataclasses(self.expected_namespace)
        for name, exp_cls in exp_classes.items():
            gen_cls = gen_classes[name]
            gen_params = gen_cls.__dataclass_params__
            exp_params = exp_cls.__dataclass_params__

            assert (
                gen_params.eq == exp_params.eq
            ), f"eq param of {gen_cls.__name__} mismatch"
            assert (
                gen_params.unsafe_hash == exp_params.unsafe_hash
            ), f"unsafe_hash param of {gen_cls.__name__} mismatch"
            if hasattr(gen_params, "kw_only"):
                assert (
                    gen_params.kw_only == exp_params.kw_only
                ), f"kw_only param of {gen_cls.__name__} mismatch"


@pytest.fixture
def stub_comparator():
    """
    Fixture that provides a StubComparator initialized with generated and expected namespaces.
    """
    generator = RoleStubGenerator(university_ontology_like_classes_without_descriptors)
    generated_stub = generator.generate_stub()

    expected_stub_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "dataset",
        "university_ontology_like_classes_without_descriptors.pyi",
    )
    with open(expected_stub_path, "r") as f:
        expected_stub_content = f.read()

    gen_namespace = execute_stub(generated_stub, "generated_stub")
    exp_namespace = execute_stub(expected_stub_content, "expected_stub")

    yield StubComparator(gen_namespace, exp_namespace)

    import sys

    sys.modules.pop("generated_stub", None)
    sys.modules.pop("expected_stub", None)


def test_stub_generation_smoke(tmp_path):
    generator = RoleStubGenerator(university_ontology_like_classes_without_descriptors)
    stub = generator.generate_stub()
    assert "class Person(Symbol):" in stub
    assert "class RoleForPerson(Person):" in stub
    assert "class CEOAsFirstRole(RoleForPerson):" in stub
    assert "head_of: RecognizedGroup = field(init=False)" in stub
    assert "from __future__ import annotations" in stub

    with open("./" + "generated_stub.pyi", "w") as f:
        f.write(stub)

    with open(tmp_path / "generated_stub.pyi", "w") as f:
        f.write(stub)

    generated_stub_path = tmp_path / "generated_stub.pyi"
    assert generated_stub_path.exists()


def test_full_stub_comparison_class_existence(stub_comparator):
    """
    Tests that all classes defined in the expected stub exist in the generated stub.
    """
    stub_comparator.compare_class_existence()


def test_full_stub_comparison_class_hierarchy(stub_comparator):
    """
    Tests that the class hierarchy (base classes) matches between stubs.
    """
    stub_comparator.compare_class_hierarchy()


def test_full_stub_comparison_field_details(stub_comparator):
    """
    Tests that all fields, their types, and arguments match between stubs.
    """
    stub_comparator.compare_field_details()


def test_full_stub_comparison_dataclass_params(stub_comparator):
    """
    Tests that @dataclass decorator arguments match between stubs.
    """
    stub_comparator.compare_dataclass_params()
