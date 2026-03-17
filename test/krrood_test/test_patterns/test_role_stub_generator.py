import inspect
import os
from dataclasses import dataclass, fields, is_dataclass
from typing import Dict, Any, Type

import pytest

from krrood.patterns.role.role_stub_generator import RoleStubGenerator
from ..dataset.role_and_ontology import (
    university_ontology_like_classes_without_descriptors,
)


def execute_stub(stub_content: str, name: str = "__stub__", package: str | None = None):
    """
    Executes the stub content and returns the namespace.
    Supports relative imports by providing a proper module context.
    """
    import sys
    import types
    import importlib.machinery
    import unittest.mock

    # Create module
    module = types.ModuleType(name)

    module.__name__ = name
    module.__package__ = package if package else name.rpartition(".")[0]
    module.__file__ = f"<{name}>"

    # Create minimal spec so Python treats this like a real module
    module.__spec__ = importlib.machinery.ModuleSpec(
        name=name,
        loader=None,
        is_package=False,
    )

    # Register module
    sys.modules[name] = module

    namespace = module.__dict__

    # Fake Symbol class
    class Symbol:
        pass

    namespace["Symbol"] = Symbol

    # Patch Symbol in source modules
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
        "role_and_ontology",
        "_ground_truth_university_ontology_like_classes_without_descriptors.pyi",
    )
    with open(expected_stub_path, "r") as f:
        expected_stub_content = f.read()

    gen_namespace = execute_stub(
        generated_stub, "generated_stub", package="test.krrood_test.dataset.role_and_ontology"
    )
    exp_namespace = execute_stub(
        expected_stub_content, "expected_stub", package="test.krrood_test.dataset.role_and_ontology"
    )

    yield StubComparator(gen_namespace, exp_namespace)

    import sys

    sys.modules.pop("generated_stub", None)
    sys.modules.pop("expected_stub", None)


@pytest.mark.order("first")
def test_stub_generation_smoke():
    generator = RoleStubGenerator(university_ontology_like_classes_without_descriptors)
    stub = generator.generate_stub(write=True)
    assert generator.path.exists()


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
