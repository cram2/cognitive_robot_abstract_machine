import libcst
import pytest
from krrood.patterns.code_generation.libcst_node_factory import LibCSTNodeFactory


def test_make_property_getter_node_has_property_decorator():
    node = LibCSTNodeFactory.make_property_getter_node("name", "str", "self.delegatee.name")
    assert any(
        libcst.Module([]).code_for_node(d.decorator) == "property"
        for d in node.decorators
    )


def test_make_property_getter_node_abstract_has_abstractmethod():
    node = LibCSTNodeFactory.make_property_getter_node("name", "str", "...")
    decorator_names = [libcst.Module([]).code_for_node(d.decorator) for d in node.decorators]
    assert "abstractmethod" in decorator_names


def test_make_property_setter_node_has_setter_decorator():
    node = LibCSTNodeFactory.make_property_setter_node("name", "str", "self.delegatee.name = value")
    assert any(
        "setter" in libcst.Module([]).code_for_node(d.decorator)
        for d in node.decorators
    )


def test_make_dataclass_has_decorator():
    node = LibCSTNodeFactory.make_dataclass("MyClass")
    decorator_code = libcst.Module([]).code_for_node(node.decorators[0].decorator)
    assert "dataclass" in decorator_code


def test_make_dataclass_with_bases():
    node = LibCSTNodeFactory.make_dataclass("MyClass", bases=["Base1", "Base2"])
    base_names = [libcst.Module([]).code_for_node(b.value) for b in node.bases]
    assert "Base1" in base_names
    assert "Base2" in base_names


def test_get_name_from_base_node_plain():
    node = libcst.parse_expression("SomeBase")
    assert LibCSTNodeFactory.get_name_from_base_node(node) == "SomeBase"


def test_get_name_from_base_node_subscript():
    node = libcst.parse_expression("Generic[T]")
    assert LibCSTNodeFactory.get_name_from_base_node(node) == "Generic"


def test_to_cst_expression_string():
    expr = LibCSTNodeFactory.to_cst_expression("MyClass")
    assert isinstance(expr, (libcst.Name, libcst.Attribute))


def test_make_annotation():
    ann = LibCSTNodeFactory.make_annotation("Optional[str]")
    assert isinstance(ann, libcst.Annotation)


def test_get_node_with_new_body():
    original = LibCSTNodeFactory.make_dataclass("Foo")
    new_body = [libcst.parse_statement("x: int")]
    updated = LibCSTNodeFactory.get_node_with_new_body(original, new_body)
    assert updated.name.value == "Foo"
    assert len(updated.body.body) == 1
