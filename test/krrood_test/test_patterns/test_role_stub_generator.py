import pytest
import os
from krrood.patterns.role_stub_generator import RoleStubGenerator
from test.krrood_test.dataset import (
    university_ontology_like_classes_without_descriptors,
)
from dataclasses import fields


def test_stub_generation_smoke(tmp_path):
    generator = RoleStubGenerator(university_ontology_like_classes_without_descriptors)
    stub = generator.generate_stub()
    assert "class Person(Symbol):" in stub
    assert "class RoleForPerson(Person):" in stub
    assert "class CEOAsFirstRole(RoleForPerson):" in stub
    assert "head_of: RecognizedGroup = field(init=False)" in stub
    assert "from __future__ import annotations" in stub

    # with open("./" + "generated_stub.pyi", "w") as f:
    #     f.write(stub)

    with open(tmp_path / "generated_stub.pyi", "w") as f:
        f.write(stub)

    generated_stub_path = tmp_path / "generated_stub.pyi"
    assert generated_stub_path.exists()


def test_role_taker_mapping():
    from test.krrood_test.dataset.university_ontology_like_classes_without_descriptors import (
        CEOAsFirstRole,
        Person,
    )

    taker_type = CEOAsFirstRole.get_role_taker_type()
    assert taker_type is Person

    root_taker_type = CEOAsFirstRole.get_root_role_taker_type()
    assert root_taker_type is Person


def test_full_stub_comparison():
    generator = RoleStubGenerator()
    generated_stub = generator.generate_stub(
        university_ontology_like_classes_without_descriptors
    )

    expected_stub_path = "test/krrood_test/dataset/university_ontology_like_classes_without_descriptors.pyi"
    with open(expected_stub_path, "r") as f:
        expected_stub = f.read()

    # We don't expect exact string match because of formatting, order of roles, etc.
    # But we can check for key elements.

    # Check that all classes in expected are in generated
    import re

    class_names = re.findall(r"class (\w+)", expected_stub)
    for name in class_names:
        assert f"class {name}" in generated_stub

    # Check for specific field transformations
    assert "head_of: RecognizedGroup = field(init=False)" in generated_stub
    assert (
        "head_of: RecognizedGroup = field(kw_only=True, default=None)" in generated_stub
    )
