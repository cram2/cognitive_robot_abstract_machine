"""
Tests that RoleFor classes produced from a chain of non-taker mixin base classes
mirror the inheritance hierarchy instead of being all listed flat.

Dataset: chained_mixin_takers.py
    BaseA → ChildA → GrandchildA (role taker)

Expected mixin hierarchy:
    RoleForBaseA(ABC)
    RoleForChildA(RoleForBaseA, ABC)
    RoleForGrandchildA(RoleForChildA, ABC)   # NOT listing RoleForBaseA directly
"""

import pytest
import libcst as cst

from krrood.patterns.role.role_transformer import RoleTransformer, TransformationMode

TRANSFORMED = TransformationMode.TRANSFORMED.value
from test.krrood_test.dataset.role_and_ontology import chained_mixin_takers


@pytest.fixture(scope="module")
def mixin_source():
    transformer = RoleTransformer(chained_mixin_takers, file_name_prefix=TRANSFORMED)
    _, src = transformer.transform()[chained_mixin_takers]
    return src


def _classes(source: str) -> dict[str, cst.ClassDef]:
    tree = cst.parse_module(source)
    return {stmt.name.value: stmt for stmt in tree.body if isinstance(stmt, cst.ClassDef)}


def _base_names(cls_def: cst.ClassDef) -> list[str]:
    return [cst.parse_module("").code_for_node(b.value).strip() for b in cls_def.bases]


def _method_names(cls_def: cst.ClassDef) -> set[str]:
    return {stmt.name.value for stmt in cls_def.body.body if isinstance(stmt, cst.FunctionDef)}


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------


def test_all_rolefor_classes_generated(mixin_source):
    classes = _classes(mixin_source)
    assert "RoleForBaseA" in classes
    assert "RoleForChildA" in classes
    assert "RoleForGrandchildA" in classes


# ---------------------------------------------------------------------------
# Hierarchical bases
# ---------------------------------------------------------------------------


def test_base_rolefor_has_only_abc(mixin_source):
    """RoleForBaseA is the root: its only base should be ABC."""
    classes = _classes(mixin_source)
    bases = _base_names(classes["RoleForBaseA"])
    assert bases == ["ABC"], f"Expected ['ABC'], got {bases}"


def test_child_rolefor_inherits_base(mixin_source):
    """RoleForChildA must list RoleForBaseA as a base (not duplicate its methods)."""
    classes = _classes(mixin_source)
    bases = _base_names(classes["RoleForChildA"])
    assert "RoleForBaseA" in bases, f"RoleForChildA bases: {bases}"


def test_grandchild_rolefor_inherits_child_not_base_directly(mixin_source):
    """RoleForGrandchildA lists RoleForChildA but NOT RoleForBaseA (covered transitively)."""
    classes = _classes(mixin_source)
    bases = _base_names(classes["RoleForGrandchildA"])
    assert "RoleForChildA" in bases, f"RoleForGrandchildA bases: {bases}"
    assert "RoleForBaseA" not in bases, (
        f"RoleForBaseA should be transitively inherited, not listed directly; got: {bases}"
    )


def test_topological_order(mixin_source):
    """RoleForBaseA must appear before RoleForChildA, which must appear before RoleForGrandchildA."""
    base_pos = mixin_source.index("class RoleForBaseA")
    child_pos = mixin_source.index("class RoleForChildA")
    grand_pos = mixin_source.index("class RoleForGrandchildA")
    assert base_pos < child_pos < grand_pos, (
        f"Wrong order: RoleForBaseA at {base_pos}, RoleForChildA at {child_pos}, "
        f"RoleForGrandchildA at {grand_pos}"
    )


# ---------------------------------------------------------------------------
# Method placement — no duplication across the hierarchy
# ---------------------------------------------------------------------------


def test_base_method_only_in_base_rolefor(mixin_source):
    """base_method is defined on BaseA and must appear only in RoleForBaseA."""
    classes = _classes(mixin_source)
    assert "base_method" in _method_names(classes["RoleForBaseA"])
    assert "base_method" not in _method_names(classes["RoleForChildA"])
    assert "base_method" not in _method_names(classes["RoleForGrandchildA"])


def test_child_method_only_in_child_rolefor(mixin_source):
    """child_method is defined on ChildA and must appear only in RoleForChildA."""
    classes = _classes(mixin_source)
    assert "child_method" in _method_names(classes["RoleForChildA"])
    assert "child_method" not in _method_names(classes["RoleForBaseA"])
    assert "child_method" not in _method_names(classes["RoleForGrandchildA"])


def test_grandchild_method_only_in_grandchild_rolefor(mixin_source):
    """grandchild_method is defined on GrandchildA and must appear only in RoleForGrandchildA."""
    classes = _classes(mixin_source)
    assert "grandchild_method" in _method_names(classes["RoleForGrandchildA"])
    assert "grandchild_method" not in _method_names(classes["RoleForBaseA"])
    assert "grandchild_method" not in _method_names(classes["RoleForChildA"])


def test_grandchild_rolefor_has_abstract_role_taker(mixin_source):
    """RoleForGrandchildA must declare an abstract role_taker property."""
    classes = _classes(mixin_source)
    assert "role_taker" in _method_names(classes["RoleForGrandchildA"])


def test_each_segregated_rolefor_has_abstract_role_taker(mixin_source):
    """Both segregated RoleFor classes must declare an abstract role_taker property."""
    classes = _classes(mixin_source)
    assert "role_taker" in _method_names(classes["RoleForBaseA"])
    assert "role_taker" in _method_names(classes["RoleForChildA"])
