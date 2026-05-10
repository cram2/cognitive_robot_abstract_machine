import pytest

from krrood.patterns.role.role_transformer import RoleTransformer, TransformationMode

TRANSFORMED = TransformationMode.TRANSFORMED.value
from .helpers import get_module_comparators, get_ground_truth_module_source, get_comparator_for_modules
from ..dataset.role_and_ontology import (
    university_ontology_like_classes_without_descriptors,
    reproduction_module,
    generic_typevar_takers,
    subclass_safe_generic_takers,
    independent_typevar_takers,
    two_role_taker_narrowing,
    unsubscripted_intermediate_taker,
)

import libcst as cst
from krrood.patterns.role.role_transformer import RoleModuleTransformer
from libcst.codemod import CodemodContext

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def module_transformer():
    return RoleTransformer(
        university_ontology_like_classes_without_descriptors,
        file_name_prefix=TRANSFORMED,
    )


@pytest.fixture
def module_comparators(module_transformer):
    return get_module_comparators(
        module_transformer.transform()
    )  # no cleanup needed — no sys.modules pollution


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.order("first")
def test_transformation_smoke(module_transformer):
    _ = module_transformer.transform(write=True)
    assert module_transformer.path.exists()


def test_class_existence(module_comparators):
    """Tests that all classes defined in the ground truth module exist in the generated module."""
    for comparator in module_comparators:
        comparator.compare_class_existence()


def test_class_hierarchy(module_comparators):
    """Tests that the class hierarchy (base classes) matches between modules."""
    for comparator in module_comparators:
        comparator.compare_class_hierarchy()


def test_field_details(module_comparators):
    """Tests that all fields, their types, and defaults match between modules."""
    for comparator in module_comparators:
        comparator.compare_field_details()


def test_dataclass_params(module_comparators):
    """Tests that @dataclass decorator arguments match between modules."""
    for comparator in module_comparators:
        comparator.compare_dataclass_params()


def test_field_order(module_comparators):
    """Tests that fields appear in the same order between modules."""
    for comparator in module_comparators:
        comparator.compare_field_order()


def test_method_details(module_comparators):
    """Tests that all methods, properties, their parameters, and return types match between modules."""
    for comparator in module_comparators:
        comparator.compare_method_details()


def test_imports(module_comparators):
    """Tests that all import statements match between modules."""
    for comparator in module_comparators:
        comparator.compare_imports()


def test_missing_imports_in_mixins():
    """
    Tests that missing imports in role mixins are resolved.
    In reproduction_module, Taker inherits from BaseTaker.
    BaseTaker.get_external() returns ExternalType.
    TakerRoleAttributes should include get_external() and import ExternalType.
    """
    transformer = RoleTransformer(reproduction_module, file_name_prefix=TRANSFORMED)
    results = transformer.transform()

    # reproduction_module should be in results
    assert reproduction_module in results
    transformed_source, mixin_source = results[reproduction_module]

    # Check mixin_source for ExternalType import
    assert (
        "from test.krrood_test.dataset.role_and_ontology.external_types import ExternalType"
        in mixin_source
    )

    # Check for generic type handling: List[ExternalType] should NOT have full path
    # and ExternalType should be imported (covered above)
    print(f"DEBUG: mixin_source:\n{mixin_source}")
    assert (
        "List[test.krrood_test.dataset.role_and_ontology.external_types.ExternalType]"
        not in mixin_source
    )
    assert "list[ExternalType]" in mixin_source or "List[ExternalType]" in mixin_source


def test_transformation_idempotency():
    """
    Tests that rerunning the transformation does not duplicate base classes.
    """
    transformer = RoleTransformer(reproduction_module, file_name_prefix=TRANSFORMED)
    results = transformer.transform()
    transformed_source, _ = results[reproduction_module]

    # Now simulate rerunning on the transformed source
    tree = cst.parse_module(transformed_source)
    context = CodemodContext()
    mod_transformer = RoleModuleTransformer(
        context=context,
        class_diagram=transformer.class_diagram,
        module=reproduction_module,
        taker_modules=transformer.taker_modules,
        file_name_prefix=TRANSFORMED,
    )

    # We need to make sure mod_transformer uses the same logic as transform()
    mod_transformer.transform_module(tree)
    retransformed_source = mod_transformer.transformed_module.code

    # Check for duplicates in retransformed_source:
    # HasRoles should appear exactly once in the base list and once in the import.
    assert retransformed_source.count("HasRoles") == 2


def test_typing_alias_imported_from_base_method():
    """
    Regression: a typing alias (e.g. Dict) used only in a base taker's method
    return type must still be imported in the generated mixin.

    Root cause: get_origin(Dict[str, str]) returns the builtin 'dict', which
    _handle_generic_type registers as 'dict -> builtins'.  The uppercase alias
    name 'Dict' is never added to name_to_module_map, so _add_typing_imports
    silently skips it and the generated mixin is missing the import.

    Reproduction: BaseTaker.to_dict() -> Dict[str, str] (Dict imported in
    base_taker.py only).  reproduction_module.py does NOT import Dict.

    BaseTaker is a transitive same-package ancestor of Taker; its DelegatorFor
    mixin is generated in the base_taker module's mixin, not reproduction_module's.
    """
    from test.krrood_test.dataset.role_and_ontology import base_taker as base_taker_module

    transformer = RoleTransformer(reproduction_module, file_name_prefix=TRANSFORMED)
    all_sources = transformer.transform()
    _, base_mixin = all_sources[base_taker_module]

    # The delegation method must appear in the base module's mixin
    assert "def to_dict" in base_mixin, "to_dict delegation missing from base module mixin"
    # Dict must be imported at the top level (it is a typing alias → top-level import)
    assert (
        "import Dict" in base_mixin
    ), f"'Dict' not imported in base module mixin.\nMixin source:\n{base_mixin}"


def test_no_init_or_post_init_in_role_for():
    """
    Tests that __init__ and __post_init__ are NOT present in the generated DelegatorFor class.
    """
    # We add them to Taker for this test
    from test.krrood_test.dataset.role_and_ontology.reproduction_module import Taker

    # Save original methods if any
    orig_init = getattr(Taker, "__init__", None)
    orig_post_init = getattr(Taker, "__post_init__", None)

    try:

        def mock_init(self, some_arg):
            pass

        def mock_post_init(self):
            pass

        Taker.__init__ = mock_init
        Taker.__post_init__ = mock_post_init

        transformer = RoleTransformer(reproduction_module, file_name_prefix=TRANSFORMED)
        results = transformer.transform()

        assert reproduction_module in results
        _, mixin_source = results[reproduction_module]

        # DelegatorForTaker should be generated for Taker
        assert "class DelegatorForTaker" in mixin_source

        # Ensure __init__ and __post_init__ are not present as methods in DelegatorForTaker
        assert "def __init__" not in mixin_source
        assert "def __post_init__" not in mixin_source
    finally:
        # Restore (or remove if they didn't exist)
        if orig_init:
            Taker.__init__ = orig_init
        else:
            if hasattr(Taker, "__init__"):
                del Taker.__init__

        if orig_post_init:
            Taker.__post_init__ = orig_post_init
        else:
            if hasattr(Taker, "__post_init__"):
                del Taker.__post_init__


# ---------------------------------------------------------------------------
# TypeVar preservation and re-declaration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def generic_typevar_mixin_source():
    transformer = RoleTransformer(generic_typevar_takers, file_name_prefix=TRANSFORMED)
    _, mixin_source = transformer.transform()[generic_typevar_takers]
    return mixin_source


@pytest.fixture
def generic_typevar_mixin_comparator(generic_typevar_mixin_source):
    expected = get_ground_truth_module_source(generic_typevar_takers, is_mixin=True)
    return get_comparator_for_modules(generic_typevar_mixin_source, expected)


def test_typevar_preserved_in_base_rolefor(generic_typevar_mixin_source):
    """entity in DelegatorForGenericBaseMixin uses TBase, not BaseEntity."""
    assert "TBase" in generic_typevar_mixin_source
    assert "def entity(self) -> TBase" in generic_typevar_mixin_source


def test_typevar_narrowing_redeclared(generic_typevar_mixin_source):
    """DelegatorForNarrowedTypeVarTaker re-declares entity with the narrowed TConcreteEntity."""
    assert "TConcreteEntity" in generic_typevar_mixin_source
    assert "def entity(self) -> TConcreteEntity" in generic_typevar_mixin_source


def test_concrete_type_substitution_redeclared(generic_typevar_mixin_source):
    """DelegatorForConcreteTypeTaker re-declares entity with the concrete ConcreteEntity."""
    assert "ConcreteEntity" in generic_typevar_mixin_source
    assert "def entity(self) -> ConcreteEntity" in generic_typevar_mixin_source


def test_unspecialized_subclass_no_entity_redeclaration(generic_typevar_mixin_source):
    """DelegatorForUnspecializedSubTaker inherits entity from base without re-declaring it."""
    assert "class DelegatorForUnspecializedSubTaker" in generic_typevar_mixin_source
    unspecialized_section = generic_typevar_mixin_source.split("class DelegatorForUnspecializedSubTaker")[1]
    assert "def entity" not in unspecialized_section


def test_generic_typevar_mixin_class_hierarchy(generic_typevar_mixin_comparator):
    """All DelegatorFor classes have the correct base classes."""
    generic_typevar_mixin_comparator.compare_class_hierarchy()


def test_generic_typevar_mixin_class_existence(generic_typevar_mixin_comparator):
    """All expected DelegatorFor classes are generated."""
    generic_typevar_mixin_comparator.compare_class_existence()


def test_generic_typevar_mixin_method_details(generic_typevar_mixin_comparator):
    """All methods and properties have correct signatures and return types."""
    generic_typevar_mixin_comparator.compare_method_details()


def test_generic_typevar_mixin_imports(generic_typevar_mixin_comparator):
    """Generated mixin imports the correct TypeVars and classes."""
    generic_typevar_mixin_comparator.compare_imports()


# ---------------------------------------------------------------------------
# SubClassSafeGeneric TypeVar narrowing tests
# ---------------------------------------------------------------------------


@pytest.fixture
def subclass_safe_generic_mixin_source():
    transformer = RoleTransformer(subclass_safe_generic_takers, file_name_prefix=TRANSFORMED)
    _, mixin_source = transformer.transform()[subclass_safe_generic_takers]
    return mixin_source


@pytest.fixture
def subclass_safe_generic_mixin_comparator(subclass_safe_generic_mixin_source):
    expected = get_ground_truth_module_source(subclass_safe_generic_takers, is_mixin=True)
    return get_comparator_for_modules(subclass_safe_generic_mixin_source, expected)


def test_subclasssafegeneric_base_rolefor_uses_base_typevar(subclass_safe_generic_mixin_source):
    """item in DelegatorForItemHolder uses TItem (the SubClassSafeGeneric TypeVar), not a concrete type."""
    assert "def item(self) -> TItem" in subclass_safe_generic_mixin_source


def test_subclasssafegeneric_typevar_narrowing_redeclared(subclass_safe_generic_mixin_source):
    """DelegatorForSpecificItemTaker re-declares item with TSpecificItem despite the SubClassSafeGeneric
    TypeVar aliasing that causes __parameters__ to expose T instead of the annotation-level TypeVar."""
    section = subclass_safe_generic_mixin_source.split("class DelegatorForSpecificItemTaker")[1]
    assert "def item(self) -> TSpecificItem" in section


def test_subclasssafegeneric_mixin_class_hierarchy(subclass_safe_generic_mixin_comparator):
    """DelegatorForSpecificItemTaker inherits from DelegatorForItemHolder."""
    subclass_safe_generic_mixin_comparator.compare_class_hierarchy()


def test_subclasssafegeneric_mixin_method_details(subclass_safe_generic_mixin_comparator):
    """All methods and properties have correct signatures and return types."""
    subclass_safe_generic_mixin_comparator.compare_method_details()


def test_subclasssafegeneric_mixin_imports(subclass_safe_generic_mixin_comparator):
    """Generated mixin imports TItem and TSpecificItem."""
    subclass_safe_generic_mixin_comparator.compare_imports()


# ---------------------------------------------------------------------------
# Independent TypeVar tests
# Regression: get_generic_type_param incorrectly resolved transitive generic
# bases returning the wrong TypeVar (e.g. THasRootBody instead of TBody).
# ---------------------------------------------------------------------------


@pytest.fixture
def independent_typevar_mixin_source():
    transformer = RoleTransformer(independent_typevar_takers, file_name_prefix=TRANSFORMED)
    _, mixin_source = transformer.transform()[independent_typevar_takers]
    return mixin_source


@pytest.fixture
def independent_typevar_mixin_comparator(independent_typevar_mixin_source):
    expected = get_ground_truth_module_source(independent_typevar_takers, is_mixin=True)
    return get_comparator_for_modules(independent_typevar_mixin_source, expected)


def test_independent_typevar_root_not_overwritten_by_content_typevar(independent_typevar_mixin_source):
    """root must not be redeclared in DelegatorForMultiTaker — TSpecificRoot is already established
    in DelegatorForNarrowedRootHolder and inherited through the chain.

    Regression: transitive get_generic_type_param returned the wrong TypeVar
    from an unrelated independent generic (TContent2) for the root property.
    """
    multi_taker_section = independent_typevar_mixin_source.split("class DelegatorForMultiTaker")[1]
    assert "def root" not in multi_taker_section
    assert "def root(self) -> TContent2" not in multi_taker_section


def test_independent_typevar_content_uses_narrowed_typevar(independent_typevar_mixin_source):
    """content in DelegatorForMultiTaker uses TContent2 (narrowed from TContent)."""
    multi_taker_section = independent_typevar_mixin_source.split("class DelegatorForMultiTaker")[1]
    assert "def content(self) -> TContent2" in multi_taker_section


def test_independent_typevar_mixin_class_existence(independent_typevar_mixin_comparator):
    """All expected DelegatorFor classes are generated."""
    independent_typevar_mixin_comparator.compare_class_existence()


def test_independent_typevar_mixin_class_hierarchy(independent_typevar_mixin_comparator):
    """All DelegatorFor classes have the correct base classes."""
    independent_typevar_mixin_comparator.compare_class_hierarchy()


def test_independent_typevar_mixin_method_details(independent_typevar_mixin_comparator):
    """All methods and properties have correct signatures and return types."""
    independent_typevar_mixin_comparator.compare_method_details()


def test_independent_typevar_mixin_imports(independent_typevar_mixin_comparator):
    """Generated mixin imports the correct TypeVars and classes."""
    independent_typevar_mixin_comparator.compare_imports()


# ---------------------------------------------------------------------------
# Two role taker narrowing tests
# Regression: when BOTH the defining base and the narrowing subclass are role
# takers, the field ends up in taker_fields and was skipped before any type-
# narrowing check, so the re-declaration was never generated.
# ---------------------------------------------------------------------------


@pytest.fixture
def two_role_taker_narrowing_mixin_source():
    transformer = RoleTransformer(two_role_taker_narrowing, file_name_prefix=TRANSFORMED)
    _, mixin_source = transformer.transform()[two_role_taker_narrowing]
    return mixin_source


@pytest.fixture
def two_role_taker_narrowing_comparator(two_role_taker_narrowing_mixin_source):
    expected = get_ground_truth_module_source(two_role_taker_narrowing, is_mixin=True)
    return get_comparator_for_modules(two_role_taker_narrowing_mixin_source, expected)


def test_two_role_taker_narrowing_entity_redeclared_in_derived(two_role_taker_narrowing_mixin_source):
    """entity in DelegatorForDerivedHolder must be redeclared as TSpecificEntity, not TBaseEntity.

    Regression: when both BaseHolder and DerivedHolder are role takers, entity was in
    taker_fields and skipped before the narrowing check, so no re-declaration was generated.
    """
    derived_section = two_role_taker_narrowing_mixin_source.split("class DelegatorForDerivedHolder")[1]
    assert "def entity(self) -> TSpecificEntity" in derived_section
    assert "def entity(self) -> TBaseEntity" not in derived_section


def test_two_role_taker_narrowing_base_uses_base_typevar(two_role_taker_narrowing_mixin_source):
    """entity in DelegatorForBaseHolder uses TBaseEntity (not narrowed)."""
    base_section = two_role_taker_narrowing_mixin_source.split("class DelegatorForBaseHolder")[1]
    base_section = base_section.split("class DelegatorForDerivedHolder")[0]
    assert "def entity(self) -> TBaseEntity" in base_section


def test_two_role_taker_narrowing_class_existence(two_role_taker_narrowing_comparator):
    """All expected DelegatorFor classes are generated."""
    two_role_taker_narrowing_comparator.compare_class_existence()


def test_two_role_taker_narrowing_class_hierarchy(two_role_taker_narrowing_comparator):
    """DelegatorForDerivedHolder extends DelegatorForBaseHolder."""
    two_role_taker_narrowing_comparator.compare_class_hierarchy()


def test_two_role_taker_narrowing_method_details(two_role_taker_narrowing_comparator):
    """All methods and properties have correct signatures and return types."""
    two_role_taker_narrowing_comparator.compare_method_details()


def test_two_role_taker_narrowing_imports(two_role_taker_narrowing_comparator):
    """Generated mixin imports TBaseEntity and TSpecificEntity."""
    two_role_taker_narrowing_comparator.compare_imports()


# ---------------------------------------------------------------------------
# Unsubscripted intermediate taker tests
# Regression: when a role taker (Shelf) inherits a concrete intermediate
# (CargoCrate(Box[Cargo])) without subscript, from_specialization(Shelf, Box)
# returned an empty substitution because get_generic_type_param skips
# unsubscripted bases.  As a result, nearest_covered_type stayed as TBoxItem
# and the derived role taker (Rack) got a spurious ``item -> Cargo``
# re-declaration even though Shelf already covered it.
# ---------------------------------------------------------------------------


@pytest.fixture
def unsubscripted_intermediate_mixin_source():
    transformer = RoleTransformer(unsubscripted_intermediate_taker, file_name_prefix=TRANSFORMED)
    _, mixin_source = transformer.transform()[unsubscripted_intermediate_taker]
    return mixin_source


@pytest.fixture
def unsubscripted_intermediate_mixin_comparator(unsubscripted_intermediate_mixin_source):
    expected = get_ground_truth_module_source(unsubscripted_intermediate_taker, is_mixin=True)
    return get_comparator_for_modules(unsubscripted_intermediate_mixin_source, expected)


def test_unsubscripted_intermediate_item_not_redeclared_in_rack(unsubscripted_intermediate_mixin_source):
    """item must not appear in DelegatorForRack — it is already covered by DelegatorForShelf (via CargoCrate).

    Regression: from_specialization(Shelf, Box) returned empty because Shelf inherits CargoCrate
    without a subscript, so get_generic_type_param skipped CargoCrate.  nearest_covered_type
    stayed as TBoxItem instead of Cargo, causing a spurious ``item -> Cargo`` re-declaration.
    """
    rack_section = unsubscripted_intermediate_mixin_source.split("class DelegatorForRack")[1]
    assert "def item" not in rack_section


def test_unsubscripted_intermediate_item_narrowed_in_cargo_crate(unsubscripted_intermediate_mixin_source):
    """item -> Cargo is generated for DelegatorForCargoCrate (the concrete intermediate)."""
    cargo_crate_section = unsubscripted_intermediate_mixin_source.split("class DelegatorForCargoCrate")[1]
    cargo_crate_section = cargo_crate_section.split("class DelegatorForShelf")[0]
    assert "def item(self) -> Cargo" in cargo_crate_section


def test_unsubscripted_intermediate_slot_narrowed_in_rack(unsubscripted_intermediate_mixin_source):
    """slot -> TRackSlot is generated for DelegatorForRack (Rack narrows TShelfContent to TRackSlot)."""
    rack_section = unsubscripted_intermediate_mixin_source.split("class DelegatorForRack")[1]
    assert "def slot(self) -> TRackSlot" in rack_section


def test_unsubscripted_intermediate_mixin_class_existence(unsubscripted_intermediate_mixin_comparator):
    """All expected DelegatorFor classes are generated."""
    unsubscripted_intermediate_mixin_comparator.compare_class_existence()


def test_unsubscripted_intermediate_mixin_class_hierarchy(unsubscripted_intermediate_mixin_comparator):
    """DelegatorForRack extends DelegatorForShelf which extends DelegatorForCargoCrate which extends DelegatorForBox."""
    unsubscripted_intermediate_mixin_comparator.compare_class_hierarchy()


def test_unsubscripted_intermediate_mixin_method_details(unsubscripted_intermediate_mixin_comparator):
    """All methods and properties have correct signatures and return types."""
    unsubscripted_intermediate_mixin_comparator.compare_method_details()


def test_unsubscripted_intermediate_mixin_imports(unsubscripted_intermediate_mixin_comparator):
    """Generated mixin imports TBoxItem, Cargo, TShelfContent, TRack, TRackSlot."""
    unsubscripted_intermediate_mixin_comparator.compare_imports()
