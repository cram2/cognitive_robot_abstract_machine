from __future__ import annotations
import inspect
from typing_extensions import Self, Any

from krrood.patterns.code_generation.delegation_generator import DelegationGenerator
from krrood.patterns.role.meta_data import MethodType
from krrood.patterns.code_generation.libcst_node_factory import LibCSTNodeFactory
import libcst as cst
from textwrap import dedent


class Taker:
    def normal_method(self):
        pass

    @classmethod
    def class_method(cls) -> Any:
        pass

    @classmethod
    def factory_method(cls) -> Self:
        return cls()


def test_method_type_detection():
    # Setup
    from unittest.mock import MagicMock

    node_factory = LibCSTNodeFactory()
    type_normaliser = MagicMock()
    generator = DelegationGenerator(
        node_factory=node_factory,
        delegatee_attribute_name="delegatee",
        type_normaliser=type_normaliser,
    )

    # Test cases
    cases = [
        ("normal_method", MethodType.NORMAL),
        ("class_method", MethodType.CLASS_METHOD),
        ("factory_method", MethodType.FACTORY_METHOD),
    ]

    for method_name, expected_type in cases:
        method_obj = getattr(Taker, method_name)
        source = dedent(inspect.getsource(method_obj))
        method_node = cst.parse_module(source).body[0]
        assert isinstance(method_node, cst.FunctionDef)

        detected_type = generator._get_method_type(method_obj, method_node)
        assert (
            detected_type == expected_type
        ), f"Failed for {method_name}: expected {expected_type}, got {detected_type}"


if __name__ == "__main__":
    test_method_type_detection()
    print("Test passed!")
