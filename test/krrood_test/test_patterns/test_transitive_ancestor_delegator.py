import pytest
import libcst as cst

from krrood.patterns.role.role_transformer import RoleTransformer, TransformationMode

TRANSFORMED = TransformationMode.TRANSFORMED.value
from test.krrood_test.dataset.role_and_ontology import (
    transitive_ancestor_base,
    transitive_ancestor_derived,
)


@pytest.fixture
def all_sources():
    """Return a dict mapping module -> (transformed, mixin) for the transitive ancestor scenario."""
    transformer = RoleTransformer(
        transitive_ancestor_derived, file_name_prefix=TRANSFORMED
    )
    return transformer.transform()


def _classes(source: str) -> dict[str, cst.ClassDef]:
    tree = cst.parse_module(source)
    return {
        stmt.name.value: stmt
        for stmt in tree.body
        if isinstance(stmt, cst.ClassDef)
    }


def _method_names(cls_def: cst.ClassDef) -> set[str]:
    return {
        stmt.name.value
        for stmt in cls_def.body.body
        if isinstance(stmt, cst.FunctionDef)
    }


def _base_names(cls_def: cst.ClassDef) -> list[str]:
    return [
        cst.parse_module("").code_for_node(b.value).strip()
        for b in cls_def.bases
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ancestor_delegator_in_base_module_mixin(all_sources):
    """DelegatorForAncestorBase must be in the base module's mixin, not the derived module's."""
    _, base_mixin = all_sources[transitive_ancestor_base]
    _, derived_mixin = all_sources[transitive_ancestor_derived]

    assert "class DelegatorForAncestorBase" in base_mixin, (
        "DelegatorForAncestorBase should be in the base module's mixin"
    )
    assert "class DelegatorForAncestorBase" not in derived_mixin, (
        "DelegatorForAncestorBase should NOT be in the derived module's mixin"
    )


def test_derived_delegator_in_derived_module_mixin(all_sources):
    """DelegatorForDerivedClass must be in the derived module's mixin."""
    _, derived_mixin = all_sources[transitive_ancestor_derived]
    assert "class DelegatorForDerivedClass" in derived_mixin


def test_shared_method_not_duplicated(all_sources):
    """shared_method appears exactly once across both mixin files."""
    _, base_mixin = all_sources[transitive_ancestor_base]
    _, derived_mixin = all_sources[transitive_ancestor_derived]

    assert base_mixin.count("def shared_method") == 1, (
        "shared_method should be defined exactly once in the base mixin"
    )
    assert derived_mixin.count("def shared_method") == 0, (
        "shared_method should not be duplicated in the derived mixin"
    )


def test_shared_field_in_base_mixin_only(all_sources):
    """shared_field property is in DelegatorForAncestorBase only."""
    _, base_mixin = all_sources[transitive_ancestor_base]
    _, derived_mixin = all_sources[transitive_ancestor_derived]

    base_classes = _classes(base_mixin)
    derived_classes = _classes(derived_mixin)

    assert "shared_field" in _method_names(base_classes["DelegatorForAncestorBase"])
    assert "shared_field" not in _method_names(derived_classes["DelegatorForDerivedClass"])


def test_derived_delegator_inherits_base_delegator(all_sources):
    """DelegatorForDerivedClass must inherit from DelegatorForAncestorBase."""
    _, derived_mixin = all_sources[transitive_ancestor_derived]
    derived_classes = _classes(derived_mixin)
    bases = _base_names(derived_classes["DelegatorForDerivedClass"])
    assert "DelegatorForAncestorBase" in bases, (
        f"DelegatorForDerivedClass must inherit DelegatorForAncestorBase; got bases: {bases}"
    )


def test_single_base_mixin_per_base(all_sources):
    """DelegatorForAncestorBase is defined exactly once across both mixin files."""
    _, base_mixin = all_sources[transitive_ancestor_base]
    _, derived_mixin = all_sources[transitive_ancestor_derived]
    total = base_mixin.count("class DelegatorForAncestorBase") + derived_mixin.count(
        "class DelegatorForAncestorBase"
    )
    assert total == 1, f"DelegatorForAncestorBase defined {total} times, expected 1"


def test_derived_only_method_stays_in_derived(all_sources):
    """derived_only_method is in DelegatorForDerivedClass, not in DelegatorForAncestorBase."""
    _, base_mixin = all_sources[transitive_ancestor_base]
    _, derived_mixin = all_sources[transitive_ancestor_derived]

    base_classes = _classes(base_mixin)
    derived_classes = _classes(derived_mixin)

    assert "derived_only_method" in _method_names(derived_classes["DelegatorForDerivedClass"])
    assert "derived_only_method" not in _method_names(base_classes["DelegatorForAncestorBase"])


def test_base_mixin_imported_by_derived_mixin(all_sources):
    """The derived mixin must import DelegatorForAncestorBase from the base module's mixin."""
    _, derived_mixin = all_sources[transitive_ancestor_derived]
    assert "from test.krrood_test.dataset.role_and_ontology.role_mixins.transitive_ancestor_base_role_mixins" in derived_mixin, (
        "Derived mixin must import DelegatorForAncestorBase from the base module's mixin"
    )
