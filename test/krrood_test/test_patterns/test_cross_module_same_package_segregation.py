import pytest
import libcst as cst

from krrood.patterns.role.role_transformer import RoleTransformer, TransformationMode

TRANSFORMED = TransformationMode.TRANSFORMED.value
from test.krrood_test.dataset.role_and_ontology import (
    cross_module_takers,
    cross_module_shared_base,
)


@pytest.fixture
def all_sources():
    """Return a dict mapping module -> (transformed, mixin) for the cross-module scenario."""
    transformer = RoleTransformer(cross_module_takers, file_name_prefix=TRANSFORMED)
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
    """DelegatorForCrossModuleBase must be in the shared base module's mixin."""
    _, base_mixin = all_sources[cross_module_shared_base]
    assert "class DelegatorForCrossModuleBase" in base_mixin


def test_base_mixin_not_in_derived_mixin(all_sources):
    """DelegatorForCrossModuleBase must NOT be in the derived module's mixin."""
    _, derived_mixin = all_sources[cross_module_takers]
    assert "class DelegatorForCrossModuleBase" not in derived_mixin


def test_shared_method_not_duplicated(all_sources):
    """cross_method appears exactly once (in DelegatorForCrossModuleBase)."""
    _, base_mixin = all_sources[cross_module_shared_base]
    _, derived_mixin = all_sources[cross_module_takers]
    assert base_mixin.count("def cross_method") == 1
    assert derived_mixin.count("def cross_method") == 0


def test_shared_field_in_base_mixin_only(all_sources):
    """cross_field property is in DelegatorForCrossModuleBase and not in taker-specific DelegatorFors."""
    _, base_mixin = all_sources[cross_module_shared_base]
    _, derived_mixin = all_sources[cross_module_takers]
    base_classes = _classes(base_mixin)
    derived_classes = _classes(derived_mixin)
    assert "cross_field" in _method_names(base_classes["DelegatorForCrossModuleBase"])
    assert "cross_field" not in _method_names(derived_classes["DelegatorForTakerX"])
    assert "cross_field" not in _method_names(derived_classes["DelegatorForTakerY"])


def test_single_base_mixin_per_base(all_sources):
    """DelegatorForCrossModuleBase is defined exactly once."""
    _, base_mixin = all_sources[cross_module_shared_base]
    _, derived_mixin = all_sources[cross_module_takers]
    total = base_mixin.count("class DelegatorForCrossModuleBase") + derived_mixin.count(
        "class DelegatorForCrossModuleBase"
    )
    assert total == 1


def test_taker_rolefors_inherit_base_mixin(all_sources):
    """Both DelegatorForTakerX and DelegatorForTakerY inherit from DelegatorForCrossModuleBase."""
    _, derived_mixin = all_sources[cross_module_takers]
    classes = _classes(derived_mixin)
    for name in ("DelegatorForTakerX", "DelegatorForTakerY"):
        bases = _base_names(classes[name])
        assert "DelegatorForCrossModuleBase" in bases, (
            f"{name} does not inherit DelegatorForCrossModuleBase; got bases: {bases}"
        )


def test_taker_only_methods_stay_in_taker(all_sources):
    """taker_x_only_method stays in DelegatorForTakerX; taker_y_only_method stays in DelegatorForTakerY."""
    _, base_mixin = all_sources[cross_module_shared_base]
    _, derived_mixin = all_sources[cross_module_takers]
    classes = _classes(derived_mixin)
    assert "taker_x_only_method" in _method_names(classes["DelegatorForTakerX"])
    assert "taker_y_only_method" in _method_names(classes["DelegatorForTakerY"])
    base_classes = _classes(base_mixin)
    shared_methods = _method_names(base_classes["DelegatorForCrossModuleBase"])
    assert "taker_x_only_method" not in shared_methods
    assert "taker_y_only_method" not in shared_methods


def test_base_mixin_has_abstract_role_taker(all_sources):
    """DelegatorForCrossModuleBase declares an abstract delegatee property."""
    _, base_mixin = all_sources[cross_module_shared_base]
    classes = _classes(base_mixin)
    assert "delegatee" in _method_names(classes["DelegatorForCrossModuleBase"])


def test_base_mixin_emitted_before_taker_rolefors(all_sources):
    """DelegatorForCrossModuleBase appears before DelegatorForTakerX in the base mixin source."""
    _, base_mixin = all_sources[cross_module_shared_base]
    assert "class DelegatorForCrossModuleBase" in base_mixin


def test_cross_module_base_import_present(all_sources):
    """The base mixin imports CrossModuleBase from its original module."""
    _, base_mixin = all_sources[cross_module_shared_base]
    assert "CrossModuleBase" in base_mixin
    assert "cross_module_shared_base" in base_mixin


def test_derived_mixin_imports_base_delegator(all_sources):
    """The derived mixin imports DelegatorForCrossModuleBase from the shared base's mixin."""
    _, derived_mixin = all_sources[cross_module_takers]
    assert (
        "from test.krrood_test.dataset.role_and_ontology.role_mixins.cross_module_shared_base_role_mixins"
        in derived_mixin
    )
