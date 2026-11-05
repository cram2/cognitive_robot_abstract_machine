"""
Finds dataclasses whose fields, and methods whose parameters, can be passed
positionally.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path

from typing_extensions import Iterator, override

from krrood.development_tools.agents.checker import CodeRuleChecker, CodeRuleViolation
from krrood.development_tools.agents.imported_module import (
    ImportedModule,
    InstalledCode,
)
from krrood.utils import recursive_subclasses

# %% the source vocabulary


class DataclassSyntax(StrEnum):
    """
    The names a dataclass declaration is written with.
    """

    DECORATOR = "dataclass"
    """
    The decorator, imported or reached through :mod:`dataclasses`.
    """

    KEYWORD_ONLY = "kw_only"
    """
    The decorator argument making every field keyword-only.
    """


class ImplicitArgumentDecorator(StrEnum):
    """
    The decorators deciding whether a method receives an implicit first argument.
    """

    STATIC_METHOD = "staticmethod"
    """
    Makes a method receive no instance and no class.
    """


@dataclass(frozen=True)
class NameDispatch:
    """
    An installed base class that calls the methods of its subclasses whose names start
    with a prefix, passing their arguments positionally.
    """

    base: type
    """
    The base class.
    """

    method_prefix: str
    """
    The start of the names of the methods it calls.
    """


class NameDispatchingBase(NameDispatch, Enum):
    """
    The installed base classes dispatching to methods by name.
    """

    NODE_VISITOR = (ast.NodeVisitor, "visit_")
    """
    Calls ``visit_<node class name>`` for each syntax tree node it visits.
    """


class PositionallyCalledAccessor(StrEnum):
    """
    The property accessors Python calls with a positional argument.
    """

    SETTER = "setter"
    DELETER = "deleter"


# %% walking the classes of a file


@dataclass(frozen=True, kw_only=True)
class ClassDefinition:
    """
    A class definition in a source file.
    """

    node: ast.ClassDef
    """
    The definition's syntax tree.
    """

    scope: tuple[str, ...] | None
    """
    The names of the class and the classes enclosing it; none for a class inside a
    function, which cannot be looked up.
    """


@dataclass(frozen=True, kw_only=True)
class SourceClasses:
    """
    The class definitions of one source file.
    """

    path: Path
    """
    The source file.
    """

    def definitions(self) -> Iterator[ClassDefinition]:
        """
        :return: Every class definition in the file, at any depth.
        """
        yield from self.definitions_below(
            parent=ast.parse(self.path.read_text()), scope=()
        )

    def definitions_below(
        self, *, parent: ast.AST, scope: tuple[str, ...] | None
    ) -> Iterator[ClassDefinition]:
        """
        :param scope: The names of the classes enclosing *parent*; none inside a
            function.
        :return: The class definitions below *parent*.
        """
        for node in ast.iter_child_nodes(parent):
            if isinstance(node, ast.ClassDef):
                class_scope = None if scope is None else (*scope, node.name)
                yield ClassDefinition(node=node, scope=class_scope)
                yield from self.definitions_below(parent=node, scope=class_scope)
                continue
            inner_scope = (
                None
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                else scope
            )
            yield from self.definitions_below(parent=node, scope=inner_scope)


# %% dataclasses


@dataclass(frozen=True, kw_only=True)
class PositionalFieldsChecker(CodeRuleChecker):
    """
    Finds the dataclasses not declared with ``kw_only=True``; a dataclass an
    :class:`~enum.Enum` mixes in is exempt, as the enum passes member values
    positionally.
    """

    @override
    def violations(self, *, path: Path) -> Iterator[CodeRuleViolation]:
        imported = ImportedModule(path=path)
        for definition in SourceClasses(path=path).definitions():
            decorator = self.dataclass_decorator(node=definition.node)
            if decorator is None or self.is_keyword_only(decorator=decorator):
                continue
            if self.is_enum_mixin(cls=imported.class_named(scope=definition.scope)):
                continue
            yield CodeRuleViolation(
                path=path,
                first_line=decorator.lineno,
                last_line=decorator.end_lineno,
                description=f"dataclass {definition.node.name} takes its fields "
                f"positionally; declare it with `@{DataclassSyntax.DECORATOR}"
                f"({DataclassSyntax.KEYWORD_ONLY}=True)`",
            )

    @staticmethod
    def dataclass_decorator(*, node: ast.ClassDef) -> ast.expr | None:
        """
        :return: The decorator making *node* a dataclass; none when it is not one.
        """
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Attribute):
                name = target.attr
            elif isinstance(target, ast.Name):
                name = target.id
            else:
                continue
            if name == DataclassSyntax.DECORATOR:
                return decorator
        return None

    @staticmethod
    def is_keyword_only(*, decorator: ast.expr) -> bool:
        """
        :return: Whether *decorator* is called with ``kw_only=True``.
        """
        if not isinstance(decorator, ast.Call):
            return False
        return any(
            keyword.arg == DataclassSyntax.KEYWORD_ONLY
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in decorator.keywords
        )

    @staticmethod
    def is_enum_mixin(*, cls: type | None) -> bool:
        """
        :return: Whether an :class:`~enum.Enum` subclasses *cls*; false when *cls* could
            not be imported.
        """
        if cls is None:
            return False
        return any(issubclass(subclass, Enum) for subclass in recursive_subclasses(cls))


# %% methods


@dataclass(frozen=True, kw_only=True)
class PositionalParametersChecker(CodeRuleChecker):
    """
    Finds the methods taking a parameter besides ``self`` or ``cls`` that can be passed
    positionally.

    Dunder methods, property setters and deleters, and overrides of a method an
    installed base class defines are exempt, since their callers pass positionally.
    """

    installed_code: InstalledCode = field(default_factory=InstalledCode)
    """
    The code whose base classes call the methods overriding theirs.
    """

    @override
    def violations(self, *, path: Path) -> Iterator[CodeRuleViolation]:
        imported = ImportedModule(path=path)
        for definition in SourceClasses(path=path).definitions():
            for method in definition.node.body:
                if not isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if not self.takes_positional_parameters(method=method):
                    continue
                if self.may_be_called_positionally(
                    definition=definition,
                    method=method,
                    cls=imported.class_named(scope=definition.scope),
                ):
                    continue
                yield CodeRuleViolation(
                    path=path,
                    first_line=method.lineno,
                    last_line=method.lineno,
                    description=f"method {definition.node.name}.{method.name} takes "
                    "parameters positionally; put a `*` before them",
                )

    @staticmethod
    def takes_positional_parameters(
        *, method: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> bool:
        """
        :return: Whether a caller can pass a named parameter of *method* positionally,
            besides the instance or class the method receives.
        """
        implicit_arguments = (
            0
            if any(
                isinstance(decorator, ast.Name)
                and decorator.id == ImplicitArgumentDecorator.STATIC_METHOD
                for decorator in method.decorator_list
            )
            else 1
        )
        positional = [*method.args.posonlyargs, *method.args.args]
        return len(positional) > implicit_arguments

    def may_be_called_positionally(
        self,
        *,
        definition: ClassDefinition,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        cls: type | None,
    ) -> bool:
        """
        :param definition: The class defining *method*.
        :param cls: The class *definition* defines; none when it cannot be imported.
        :return: Whether Python or an installed base class may call *method* with
            positional arguments. For a class with bases that cannot be imported, this
            cannot be ruled out.
        """
        if method.name.startswith("__") and method.name.endswith("__"):
            return True
        if any(
            isinstance(decorator, ast.Attribute)
            and decorator.attr in list(PositionallyCalledAccessor)
            for decorator in method.decorator_list
        ):
            return True
        if not definition.node.bases:
            return False
        if cls is None:
            return True
        return self.installed_code.defines_inherited_method(
            cls=cls, method_name=method.name
        ) or any(
            issubclass(cls, dispatch.base)
            and method.name.startswith(dispatch.method_prefix)
            for dispatch in NameDispatchingBase
        )
