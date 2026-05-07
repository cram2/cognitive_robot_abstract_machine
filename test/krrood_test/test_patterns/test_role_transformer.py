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
    """
    transformer = RoleTransformer(reproduction_module, file_name_prefix=TRANSFORMED)
    _, mixin_source = transformer.transform()[reproduction_module]

    # The delegation method must appear in the mixin
    assert "def to_dict" in mixin_source, "to_dict delegation missing from mixin"
    # Dict must be imported at the top level (it is a typing alias → top-level import)
    assert (
        "import Dict" in mixin_source
    ), f"'Dict' not imported in mixin.\nMixin source:\n{mixin_source}"


def test_no_init_or_post_init_in_role_for():
    """
    Tests that __init__ and __post_init__ are NOT present in the generated RoleFor class.
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

        # RoleForTaker should be generated for Taker
        assert "class RoleForTaker" in mixin_source

        # Ensure __init__ and __post_init__ are not present as methods in RoleForTaker
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
    """entity in RoleForGenericBaseMixin uses TBase, not BaseEntity."""
    assert "TBase" in generic_typevar_mixin_source
    assert "def entity(self) -> TBase" in generic_typevar_mixin_source


def test_typevar_narrowing_redeclared(generic_typevar_mixin_source):
    """RoleForNarrowedTypeVarTaker re-declares entity with the narrowed TConcreteEntity."""
    assert "TConcreteEntity" in generic_typevar_mixin_source
    assert "def entity(self) -> TConcreteEntity" in generic_typevar_mixin_source


def test_concrete_type_substitution_redeclared(generic_typevar_mixin_source):
    """RoleForConcreteTypeTaker re-declares entity with the concrete ConcreteEntity."""
    assert "ConcreteEntity" in generic_typevar_mixin_source
    assert "def entity(self) -> ConcreteEntity" in generic_typevar_mixin_source


def test_unspecialized_subclass_no_entity_redeclaration(generic_typevar_mixin_source):
    """RoleForUnspecializedSubTaker inherits entity from base without re-declaring it."""
    assert "class RoleForUnspecializedSubTaker" in generic_typevar_mixin_source
    unspecialized_section = generic_typevar_mixin_source.split("class RoleForUnspecializedSubTaker")[1]
    assert "def entity" not in unspecialized_section


def test_generic_typevar_mixin_class_hierarchy(generic_typevar_mixin_comparator):
    """All RoleFor classes have the correct base classes."""
    generic_typevar_mixin_comparator.compare_class_hierarchy()


def test_generic_typevar_mixin_class_existence(generic_typevar_mixin_comparator):
    """All expected RoleFor classes are generated."""
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
    """item in RoleForItemHolder uses TItem (the SubClassSafeGeneric TypeVar), not a concrete type."""
    assert "def item(self) -> TItem" in subclass_safe_generic_mixin_source


def test_subclasssafegeneric_typevar_narrowing_redeclared(subclass_safe_generic_mixin_source):
    """RoleForSpecificItemTaker re-declares item with TSpecificItem despite the SubClassSafeGeneric
    TypeVar aliasing that causes __parameters__ to expose T instead of the annotation-level TypeVar."""
    section = subclass_safe_generic_mixin_source.split("class RoleForSpecificItemTaker")[1]
    assert "def item(self) -> TSpecificItem" in section


def test_subclasssafegeneric_mixin_class_hierarchy(subclass_safe_generic_mixin_comparator):
    """RoleForSpecificItemTaker inherits from RoleForItemHolder."""
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
    """root must not be redeclared in RoleForMultiTaker — TSpecificRoot is already established
    in RoleForNarrowedRootHolder and inherited through the chain.

    Regression: transitive get_generic_type_param returned the wrong TypeVar
    from an unrelated independent generic (TContent2) for the root property.
    """
    multi_taker_section = independent_typevar_mixin_source.split("class RoleForMultiTaker")[1]
    assert "def root" not in multi_taker_section
    assert "def root(self) -> TContent2" not in multi_taker_section


def test_independent_typevar_content_uses_narrowed_typevar(independent_typevar_mixin_source):
    """content in RoleForMultiTaker uses TContent2 (narrowed from TContent)."""
    multi_taker_section = independent_typevar_mixin_source.split("class RoleForMultiTaker")[1]
    assert "def content(self) -> TContent2" in multi_taker_section


def test_independent_typevar_mixin_class_existence(independent_typevar_mixin_comparator):
    """All expected RoleFor classes are generated."""
    independent_typevar_mixin_comparator.compare_class_existence()


def test_independent_typevar_mixin_class_hierarchy(independent_typevar_mixin_comparator):
    """All RoleFor classes have the correct base classes."""
    independent_typevar_mixin_comparator.compare_class_hierarchy()


def test_independent_typevar_mixin_method_details(independent_typevar_mixin_comparator):
    """All methods and properties have correct signatures and return types."""
    independent_typevar_mixin_comparator.compare_method_details()


def test_independent_typevar_mixin_imports(independent_typevar_mixin_comparator):
    """Generated mixin imports the correct TypeVars and classes."""
    independent_typevar_mixin_comparator.compare_imports()
