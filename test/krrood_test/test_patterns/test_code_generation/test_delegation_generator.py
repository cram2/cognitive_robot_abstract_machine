import sys
import pytest
import libcst

from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.utils import classes_of_module
from krrood.patterns.code_generation import DelegationGenerator, LibCSTNodeFactory, TypeNormaliser, ImportNameResolver
from krrood.patterns.role.role import Role

# import the test dataset module
sys.path.insert(0, "test")
from krrood_test.dataset.role_and_ontology import university_ontology_like_classes_without_descriptors as univ_module


@pytest.fixture
def class_diagram():
    classes = classes_of_module(univ_module)
    return ClassDiagram(classes)


@pytest.fixture
def delegation_generator(class_diagram):
    source_module = univ_module
    resolver = ImportNameResolver(
        source_module=source_module,
        companion_modules=[],
        class_diagram=class_diagram,
    )
    normaliser = TypeNormaliser(resolver=resolver, class_diagram=class_diagram)
    factory = LibCSTNodeFactory()
    role_takers = set(class_diagram.role_takers)
    excluded_names = frozenset({"__init__", "__post_init__", "__new__"})

    return DelegationGenerator(
        delegatee_attribute_name="role_taker",
        node_factory=factory,
        type_normaliser=normaliser,
        already_covered_bases=role_takers,
        excluded_method_names=excluded_names,
        name_resolver=resolver,
    )


def test_delegation_groups_produced_for_taker(delegation_generator, class_diagram):
    """Delegation groups are non-empty for a known role taker."""
    taker = next(wc for wc in class_diagram.wrapped_classes if wc.clazz in class_diagram.role_takers)
    groups = delegation_generator.collect_delegation_groups(
        taker, univ_module.__name__
    )
    all_items = {k: v for k, v in groups.items() if v}
    assert all_items, "Expected non-empty delegation groups for a role taker"


def test_delegation_nodes_are_function_defs(delegation_generator, class_diagram):
    """All generated delegation nodes are libcst.FunctionDef instances."""
    taker = next(wc for wc in class_diagram.wrapped_classes if wc.clazz in class_diagram.role_takers)
    groups = delegation_generator.collect_delegation_groups(
        taker, univ_module.__name__
    )
    for group in groups.values():
        for nodes in group.values():
            for node in nodes:
                assert isinstance(node, libcst.FunctionDef)


def test_delegatee_attribute_appears_in_getter(delegation_generator, class_diagram):
    """Property getters reference the delegatee attribute name."""
    taker = next(wc for wc in class_diagram.wrapped_classes if wc.clazz in class_diagram.role_takers and wc.fields)
    groups = delegation_generator.collect_delegation_groups(
        taker, univ_module.__name__
    )
    module = libcst.Module([])
    found = False
    for group in groups.values():
        for nodes in group.values():
            for node in nodes:
                if node.decorators and "property" in module.code_for_node(node.decorators[0].decorator):
                    body_code = module.code_for_node(node.body)
                    if "role_taker" in body_code:
                        found = True
    assert found, "Expected delegation getter to reference 'role_taker'"
