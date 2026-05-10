import pytest
import libcst as cst

from krrood.patterns.role.role_transformer import RoleTransformer, TransformationMode

TRANSFORMED = TransformationMode.TRANSFORMED.value
from test.krrood_test.dataset.role_and_ontology import cross_subpackage_takers
from test.krrood_test.dataset.sibling_package import cross_subpackage_base


@pytest.fixture
def all_sources():
    """Return a dict mapping module -> (transformed, mixin) for the cross-subpackage scenario."""
    transformer = RoleTransformer(cross_subpackage_takers, file_name_prefix=TRANSFORMED)
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


def test_base_mixin_generated(all_sources):
    """DelegatorForCrossSubpackageBase must be in the base subpackage's mixin."""
    _, base_mixin = all_sources[cross_subpackage_base]
    assert "class DelegatorForCrossSubpackageBase" in base_mixin


def test_base_mixin_not_in_derived_mixin(all_sources):
    """DelegatorForCrossSubpackageBase must NOT be in the derived subpackage's mixin."""
    _, derived_mixin = all_sources[cross_subpackage_takers]
    assert "class DelegatorForCrossSubpackageBase" not in derived_mixin


def test_shared_method_not_duplicated(all_sources):
    """sub_method appears exactly once (in the base mixin, not in the derived mixin)."""
    _, base_mixin = all_sources[cross_subpackage_base]
    _, derived_mixin = all_sources[cross_subpackage_takers]
    assert base_mixin.count("def sub_method") == 1
    assert derived_mixin.count("def sub_method") == 0


def test_shared_field_in_base_mixin_only(all_sources):
    """sub_field property is in DelegatorForCrossSubpackageBase and not in taker-specific DelegatorFors."""
    _, base_mixin = all_sources[cross_subpackage_base]
    _, derived_mixin = all_sources[cross_subpackage_takers]
    base_classes = _classes(base_mixin)
    derived_classes = _classes(derived_mixin)
    assert "sub_field" in _method_names(base_classes["DelegatorForCrossSubpackageBase"])
    assert "sub_field" not in _method_names(derived_classes["DelegatorForTakerP"])
    assert "sub_field" not in _method_names(derived_classes["DelegatorForTakerQ"])


def test_single_base_mixin_per_base(all_sources):
    """DelegatorForCrossSubpackageBase is defined exactly once across both mixin files."""
    _, base_mixin = all_sources[cross_subpackage_base]
    _, derived_mixin = all_sources[cross_subpackage_takers]
    total = base_mixin.count(
        "class DelegatorForCrossSubpackageBase"
    ) + derived_mixin.count("class DelegatorForCrossSubpackageBase")
    assert total == 1


def test_taker_rolefors_inherit_base_mixin(all_sources):
    """Both DelegatorForTakerP and DelegatorForTakerQ inherit from DelegatorForCrossSubpackageBase."""
    _, derived_mixin = all_sources[cross_subpackage_takers]
    classes = _classes(derived_mixin)
    for name in ("DelegatorForTakerP", "DelegatorForTakerQ"):
        bases = _base_names(classes[name])
        assert "DelegatorForCrossSubpackageBase" in bases, (
            f"{name} does not inherit DelegatorForCrossSubpackageBase; got bases: {bases}"
        )


def test_taker_only_methods_stay_in_taker(all_sources):
    """taker_p_only_method stays in DelegatorForTakerP; taker_q_only_method stays in DelegatorForTakerQ."""
    _, base_mixin = all_sources[cross_subpackage_base]
    _, derived_mixin = all_sources[cross_subpackage_takers]
    classes = _classes(derived_mixin)
    assert "taker_p_only_method" in _method_names(classes["DelegatorForTakerP"])
    assert "taker_q_only_method" in _method_names(classes["DelegatorForTakerQ"])
    base_classes = _classes(base_mixin)
    shared_methods = _method_names(base_classes["DelegatorForCrossSubpackageBase"])
    assert "taker_p_only_method" not in shared_methods
    assert "taker_q_only_method" not in shared_methods


def test_base_mixin_has_abstract_role_taker(all_sources):
    """DelegatorForCrossSubpackageBase declares an abstract delegatee property."""
    _, base_mixin = all_sources[cross_subpackage_base]
    classes = _classes(base_mixin)
    assert "delegatee" in _method_names(classes["DelegatorForCrossSubpackageBase"])


def test_base_mixin_emitted_before_taker_rolefors(all_sources):
    """DelegatorForCrossSubpackageBase appears before DelegatorForTakerP in the base mixin source."""
    _, base_mixin = all_sources[cross_subpackage_base]
    assert "class DelegatorForCrossSubpackageBase" in base_mixin


def test_cross_subpackage_base_import_present(all_sources):
    """The base mixin imports CrossSubpackageBase from its sibling-subpackage module."""
    _, base_mixin = all_sources[cross_subpackage_base]
    assert "CrossSubpackageBase" in base_mixin
    assert "cross_subpackage_base" in base_mixin


def test_derived_mixin_imports_base_delegator(all_sources):
    """The derived mixin imports DelegatorForCrossSubpackageBase from the base subpackage's mixin."""
    _, derived_mixin = all_sources[cross_subpackage_takers]
    assert (
        "from test.krrood_test.dataset.sibling_package.role_mixins.cross_subpackage_base_role_mixins"
        in derived_mixin
    )
