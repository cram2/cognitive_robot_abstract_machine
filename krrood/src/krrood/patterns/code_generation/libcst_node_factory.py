"""
Factory for constructing libcst nodes for code generation.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import libcst


@dataclasses.dataclass
class LibCSTNodeFactory:
    """
    Builds libcst nodes for code generation pipelines.
    All methods are pure: they depend only on their arguments.
    """

    @classmethod
    def make_dataclass(
        cls,
        name: str,
        bases: list[type | str] | None = None,
        body: list[libcst.BaseStatement] | None = None,
    ) -> libcst.ClassDef:
        """
        Build a ``@dataclass``-decorated ClassDef node.

        :param name: Name of the dataclass.
        :param bases: Base classes of the dataclass.
        :param body: Body statements of the dataclass.
        :return: A libcst ClassDef node for the given dataclass.
        """
        return libcst.ClassDef(
            name=libcst.Name(name),
            bases=[libcst.Arg(value=cls.to_cst_expression(b)) for b in (bases or [])],
            body=libcst.IndentedBlock(
                body=body if body else [libcst.parse_statement("...")]
            ),
            decorators=[cls.make_dataclass_decorator()],
        )

    @classmethod
    def to_cst_expression(
        cls, has_name: type | Callable | str
    ) -> libcst.BaseExpression:
        """
        Convert a name or named object to a CST expression node.

        :param has_name: A string, class, or callable whose name should be converted.
        :return: A libcst expression node for the given name.
        """
        if isinstance(has_name, str):
            name = has_name
        elif hasattr(has_name, "__name__"):
            name = has_name.__name__
        else:
            name = str(has_name)
        name = name.replace("typing.", "").replace("typing_extensions.", "")
        try:
            return libcst.parse_expression(name)
        except Exception:
            return libcst.Name(name)

    @classmethod
    def make_dataclass_decorator(cls) -> libcst.Decorator:
        """Build the ``@dataclass(eq=False)`` decorator node."""
        return libcst.Decorator(
            decorator=libcst.parse_expression("dataclass(eq=False)")
        )

    @classmethod
    def make_property_getter_node(
        cls, name: str, type_: str, return_statement: str
    ) -> libcst.FunctionDef:
        """
        Build a property getter FunctionDef node.

        :param name: The name of the property.
        :param type_: The return type annotation string.
        :param return_statement: The expression to return, or ``...`` for abstract.
        :return: A libcst FunctionDef node representing the property getter.
        """
        decorators = [cls.make_decorator("property")]
        if return_statement == "...":
            decorators.append(cls.make_decorator("abstractmethod"))
            body = [libcst.SimpleStatementLine([libcst.Expr(libcst.Ellipsis())])]
        else:
            body = [libcst.parse_statement(f"return {return_statement}")]

        return libcst.FunctionDef(
            decorators=decorators,
            name=libcst.Name(name),
            params=cls.make_function_parameters({"self": None}),
            returns=cls.make_annotation(type_) if type_ else None,
            body=libcst.IndentedBlock(body),
        )

    @classmethod
    def make_property_setter_node(
        cls, name: str, type_: str, statement: str
    ) -> libcst.FunctionDef:
        """
        Build a property setter FunctionDef node.

        :param name: The name of the property.
        :param type_: The value type annotation string.
        :param statement: The statement executed by the setter body.
        :return: A libcst FunctionDef node representing the property setter.
        """
        return libcst.FunctionDef(
            decorators=[cls.make_decorator(f"{name}.setter")],
            name=libcst.Name(name),
            params=cls.make_function_parameters({"self": None, "value": type_}),
            body=libcst.IndentedBlock([libcst.parse_statement(statement)]),
        )

    @classmethod
    def make_property_getter_and_setter_nodes(
        cls, name: str, type_: str, getter_return_statement: str, setter_statement: str
    ) -> list[libcst.FunctionDef]:
        """
        Build both getter and setter property nodes for a field.

        :param name: The name of the field.
        :param type_: The type annotation string.
        :param getter_return_statement: The expression returned by the getter.
        :param setter_statement: The statement executed by the setter.
        :return: A list containing the getter and setter FunctionDef nodes.
        """
        getter_node = cls.make_property_getter_node(
            name, type_, getter_return_statement
        )
        setter_node = cls.make_property_setter_node(name, type_, setter_statement)
        return [getter_node, setter_node]

    @classmethod
    def make_decorator(cls, decorator_name: str) -> libcst.Decorator:
        """
        Build a Decorator node for the given name.

        :param decorator_name: The decorator expression string.
        :return: A libcst Decorator node.
        """
        return libcst.Decorator(decorator=libcst.parse_expression(decorator_name))

    @classmethod
    def make_function_parameters(
        cls, parameters: dict[str, str | None]
    ) -> libcst.Parameters:
        """
        Build a Parameters node from a mapping of names to type annotations.

        :param parameters: Mapping of parameter names to their type annotation strings.
        :return: A libcst Parameters node.
        """
        parameters = parameters or {}
        return libcst.Parameters(
            params=[
                libcst.Param(
                    name=libcst.Name(param),
                    annotation=(
                        cls.make_annotation(annotation)
                        if annotation is not None
                        else None
                    ),
                )
                for param, annotation in parameters.items()
            ]
        )

    @classmethod
    def make_annotation(cls, value: str) -> libcst.Annotation:
        """
        Build an Annotation node from a type string.

        :param value: The type annotation as a string.
        :return: A libcst Annotation node.
        """
        return libcst.Annotation(libcst.parse_expression(value))

    @classmethod
    def make_return_statement_body(cls, statement: str) -> libcst.IndentedBlock:
        """
        Build an IndentedBlock containing a single return statement.

        :param statement: The expression to return.
        :return: A libcst IndentedBlock with a return statement.
        """
        return libcst.IndentedBlock(
            body=[
                libcst.SimpleStatementLine(
                    body=[libcst.Return(value=libcst.parse_expression(statement))]
                )
            ]
        )

    @classmethod
    def get_node_with_new_body(
        cls, node: libcst.ClassDef, new_body: list[libcst.BaseStatement]
    ) -> libcst.ClassDef:
        """
        Return a copy of the node with its body replaced.

        :param node: The node to update.
        :param new_body: The replacement body statements.
        :return: A new node identical to the original but with the new body.
        """
        return node.with_changes(body=node.body.with_changes(body=new_body))

    @classmethod
    def get_renamed_node(cls, node: libcst.CSTNode, new_name: str) -> libcst.CSTNode:
        """
        Return a copy of the node with its name replaced.

        :param node: The node to rename.
        :param new_name: The replacement name string.
        :return: A new node identical to the original but with the new name.
        """
        return node.with_changes(name=libcst.Name(new_name))

    @classmethod
    def make_argument(cls, value: str) -> libcst.Arg:
        """
        Build an Arg node from a string expression.

        :param value: The argument expression string.
        :return: A libcst Arg node.
        """
        return libcst.Arg(value=libcst.parse_expression(value))

    @classmethod
    def get_name_from_base_node(cls, base_node: libcst.BaseExpression) -> str:
        """
        Extract the class name from a base class CST node.

        :param base_node: The base node to extract the name from.
        :return: The class name as a string.
        """
        if isinstance(base_node, libcst.Name):
            return base_node.value
        if isinstance(base_node, libcst.Subscript):
            if isinstance(base_node.value, libcst.Name):
                return base_node.value.value
        elif isinstance(base_node, libcst.Attribute):
            return base_node.attr.value
        raise ValueError(f"Unexpected base node type: {base_node}")

    @classmethod
    def _get_field_name_if_statement_is_field_definition(
        cls, item: libcst.BaseStatement
    ) -> str | None:
        """Return the field name if the statement is an annotated assignment, otherwise None."""
        if (
            isinstance(item, libcst.SimpleStatementLine)
            and len(item.body) == 1
            and isinstance(ann_assign := item.body[0], libcst.AnnAssign)
            and isinstance(field_name := ann_assign.target, libcst.Name)
        ):
            return field_name.value
        return None

    @classmethod
    def _get_decorator_name(cls, decorator_node: libcst.BaseExpression) -> str | None:
        """Return the simple name from a decorator expression node, or None."""
        if isinstance(decorator_node, libcst.Name):
            return decorator_node.value
        if isinstance(decorator_node, libcst.Call) and isinstance(
            decorator_node.func, libcst.Name
        ):
            return decorator_node.func.value
        return None
