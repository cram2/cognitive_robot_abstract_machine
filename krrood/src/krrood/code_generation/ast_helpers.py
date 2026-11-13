"""
Builders for the :mod:`ast` nodes used when generating Python source, so code
generators read as *what* they emit rather than *how* the AST is spelled. Each name
starts with ``ast_`` to say it builds or renders an AST node.
"""

from __future__ import annotations

import ast
import enum
import operator

from typing_extensions import Any, Callable


def ast_load_name(identifier: str) -> ast.Name:
    """
    Build a load-context name reference.

    :param identifier: The identifier to reference.
    :return: An :class:`ast.Name` in load context.
    """
    return ast.Name(id=identifier, ctx=ast.Load())


def ast_attribute_access(value: ast.expr, attribute_name: str) -> ast.Attribute:
    """
    Build a load-context attribute access ``value.attribute_name``.

    :param value: The expression the attribute is read from.
    :param attribute_name: The attribute to read.
    :return: An :class:`ast.Attribute` in load context.
    """
    return ast.Attribute(value=value, attr=attribute_name, ctx=ast.Load())


def ast_constant(value: Any) -> ast.Constant:
    """
    Build a literal constant node.

    :param value: The literal value to embed.
    :return: An :class:`ast.Constant` wrapping *value*.
    """
    return ast.Constant(value=value)


def ast_enum_member(member: enum.Enum) -> ast.Attribute:
    """
    Build the attribute access that reconstructs an enum member (``EnumType.MEMBER``).

    :param member: The enum member to reference.
    :return: An :class:`ast.Attribute` accessing the member on its enum class.
    """
    return ast_attribute_access(ast_load_name(type(member).__name__), member.name)


def ast_call(
    function_name: str,
    arguments: list[ast.expr],
    keyword_arguments: dict[str, ast.expr] | None = None,
) -> ast.Call:
    """
    Build a call of a named function.

    :param function_name: The name of the function to call.
    :param arguments: The positional arguments.
    :param keyword_arguments: The keyword arguments, keyed by parameter name.
    :return: An :class:`ast.Call`.
    """
    return ast.Call(
        func=ast_load_name(function_name),
        args=list(arguments),
        keywords=[
            ast.keyword(arg=name, value=value)
            for name, value in (keyword_arguments or {}).items()
        ],
    )


def ast_compare(
    left: ast.expr, operator_class: type[ast.cmpop], right: ast.expr
) -> ast.Compare:
    """
    Build a single binary comparison ``left <op> right``.

    :param left: The left-hand operand.
    :param operator_class: The comparison-operator class (e.g. :class:`ast.Eq`).
    :param right: The right-hand operand.
    :return: An :class:`ast.Compare` with one operator and one comparator.
    """
    return ast.Compare(left=left, ops=[operator_class()], comparators=[right])


def ast_comparison_operator_class(
    operation: Callable[[Any, Any], Any],
) -> type[ast.cmpop] | None:
    """
    Return the comparison-operator class that renders a comparison function.

    :param operation: A comparison function from :mod:`operator`, such as
        :func:`operator.eq`.
    :return: The matching class, such as :class:`ast.Eq`, or ``None`` for any other
        function.
    """
    match operation:
        case operator.eq:
            return ast.Eq
        case operator.ne:
            return ast.NotEq
        case operator.lt:
            return ast.Lt
        case operator.le:
            return ast.LtE
        case operator.gt:
            return ast.Gt
        case operator.ge:
            return ast.GtE
        case _:
            return None


def ast_with_block(context: ast.expr, body: list[ast.stmt]) -> ast.With:
    """
    Build a ``with context:`` statement wrapping *body*.

    :param context: The context-manager expression.
    :param body: The statements nested under the ``with``.
    :return: An :class:`ast.With` with a single unnamed item.
    """
    return ast.With(
        items=[ast.withitem(context_expr=context, optional_vars=None)],
        body=list(body),
    )


def ast_unparse_expression(expression: ast.expr) -> str:
    """
    Render a single expression to source, filling in required node locations.

    :param expression: The expression to render.
    :return: The unparsed source string.
    """
    return ast.unparse(ast.fix_missing_locations(expression))


def ast_unparse_statements(statements: list[ast.stmt]) -> str:
    """
    Render a list of statements to source, filling in required node locations.

    :param statements: The statements to render as a module body.
    :return: The unparsed source string.
    """
    module = ast.fix_missing_locations(
        ast.Module(body=list(statements), type_ignores=[])
    )
    return ast.unparse(module)
