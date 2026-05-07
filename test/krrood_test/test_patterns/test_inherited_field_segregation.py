"""
Tests that a dataclass field defined on a grandparent mixin is delegated in the
RoleFor for the grandparent, not in the RoleFor for a taker that merely inherits
it transitively.

Dataset: inherited_field_takers.py
    FieldOrigin (defines shared_field)
        └─ IntermediateMixin  (inherits shared_field, no re-annotation)
               ├─ TakerA      (role taker – adds taker_a_field)
               └─ TakerB      (role taker – adds taker_b_field)

Expected mixin output:
    RoleForFieldOrigin      – contains `shared_field` property
    RoleForIntermediateMixin(RoleForFieldOrigin)  – no duplicate shared_field
    RoleForTakerA(RoleForIntermediateMixin)       – contains taker_a_field
    RoleForTakerB(RoleForIntermediateMixin)       – contains taker_b_field
"""

import pytest
import libcst as cst

from krrood.patterns.role.role_transformer import RoleTransformer, TransformationMode

TRANSFORMED = TransformationMode.TRANSFORMED.value
from test.krrood_test.dataset.role_and_ontology import inherited_field_takers


@pytest.fixture(scope="module")
def mixin_source():
    transformer = RoleTransformer(inherited_field_takers, file_name_prefix=TRANSFORMED)
    _, src = transformer.transform()[inherited_field_takers]
    return src


def _classes(source: str) -> dict[str, cst.ClassDef]:
    tree = cst.parse_module(source)
    return {stmt.name.value: stmt for stmt in tree.body if isinstance(stmt, cst.ClassDef)}


def _method_names(cls_def: cst.ClassDef) -> set[str]:
    return {stmt.name.value for stmt in cls_def.body.body if isinstance(stmt, cst.FunctionDef)}


def _base_names(cls_def: cst.ClassDef) -> list[str]:
    return [cst.parse_module("").code_for_node(b.value).strip() for b in cls_def.bases]


def test_rolefor_classes_generated(mixin_source):
    classes = _classes(mixin_source)
    assert "RoleForFieldOrigin" in classes
    assert "RoleForIntermediateMixin" in classes
    assert "RoleForTakerA" in classes
    assert "RoleForTakerB" in classes


def test_shared_field_only_in_grandparent_rolefor(mixin_source):
    """shared_field must be delegated in RoleForFieldOrigin, not in TakerA/B RoleFors."""
    classes = _classes(mixin_source)
    assert "shared_field" in _method_names(classes["RoleForFieldOrigin"])
    assert "shared_field" not in _method_names(classes["RoleForIntermediateMixin"])
    assert "shared_field" not in _method_names(classes["RoleForTakerA"])
    assert "shared_field" not in _method_names(classes["RoleForTakerB"])


def test_taker_fields_in_correct_rolefors(mixin_source):
    classes = _classes(mixin_source)
    assert "taker_a_field" in _method_names(classes["RoleForTakerA"])
    assert "taker_b_field" in _method_names(classes["RoleForTakerB"])
    assert "taker_a_field" not in _method_names(classes["RoleForFieldOrigin"])
    assert "taker_b_field" not in _method_names(classes["RoleForFieldOrigin"])


def test_inheritance_chain(mixin_source):
    """RoleForTakerA and RoleForTakerB each inherit from RoleForIntermediateMixin."""
    classes = _classes(mixin_source)
    assert "RoleForIntermediateMixin" in _base_names(classes["RoleForTakerA"])
    assert "RoleForIntermediateMixin" in _base_names(classes["RoleForTakerB"])
    assert "RoleForFieldOrigin" in _base_names(classes["RoleForIntermediateMixin"])


def test_shared_field_not_duplicated(mixin_source):
    classes = _classes(mixin_source)
    classes_with_shared_field = [
        name for name, cls_def in classes.items()
        if "shared_field" in _method_names(cls_def)
    ]
    assert classes_with_shared_field == ["RoleForFieldOrigin"]
