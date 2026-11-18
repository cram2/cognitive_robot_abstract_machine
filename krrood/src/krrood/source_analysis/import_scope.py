"""
The names a source file's imports bind, built by running the imports.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
import types
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.util import resolve_name
from pathlib import Path

from typing_extensions import Any, Iterator, Literal, Mapping, overload

from krrood import logger
from krrood.exceptions import SourceDataNotProvided
from krrood.utils import get_import_path_from_path


@overload
def get_scope_from_imports(
    file_path: str | None = None,
    tree: ast.AST | None = None,
    package_name: str | None = None,
    source: str | None = None,
    *,
    defers_type_checking_imports: Literal[False] = False,
) -> dict[str, Any]:
    """
    The imports of the file or tree, all run.
    """


@overload
def get_scope_from_imports(
    file_path: str | None = None,
    tree: ast.AST | None = None,
    package_name: str | None = None,
    source: str | None = None,
    *,
    defers_type_checking_imports: Literal[True],
) -> DeferredImportScope:
    """
    The imports of the file or tree, those under ``TYPE_CHECKING`` deferred.
    """


def get_scope_from_imports(
    file_path: str | None = None,
    tree: ast.AST | None = None,
    package_name: str | None = None,
    source: str | None = None,
    *,
    defers_type_checking_imports: bool = False,
) -> dict[str, Any] | DeferredImportScope:
    """
    Create a scope dictionary from imports in a Python file or an AST tree.

    An import in an ``if TYPE_CHECKING:`` block whose module cannot be imported
    contributes no names; elsewhere, a plain ``import`` of such a module raises.

    :param file_path: The path to the Python file to extract imports from.
    :param tree: An AST tree to extract imports from. If provided, file_path is ignored.
    :param package_name: The name of the package to use for relative imports.
    :param source: The source code to extract imports from. If provided, file_path and
        tree are ignored.
    :param defers_type_checking_imports: Whether an import in an ``if TYPE_CHECKING:``
        block waits until a name it binds is first looked up, so a module nothing asks
        for is never loaded.
    :return: A dictionary representing the scope with imported modules and their
        attributes, or the :class:`DeferredImportScope` when imports are deferred.
    """
    if tree is None and file_path is None and source is None:
        raise SourceDataNotProvided(file_path=file_path, tree=tree, source_code=source)

    # Ensure we have source and a parsed AST
    if file_path and source is None:
        with open(file_path, "r") as f:
            source = f.read()
    parsed_tree = tree or (
        ast.parse(source)
        if file_path is None
        else ast.parse(source, filename=file_path)
    )

    if defers_type_checking_imports:
        deferred_scope = DeferredImportScope()
        bind_imports(
            parsed_tree=parsed_tree,
            file_path=file_path,
            package_name=package_name,
            scope=deferred_scope.bound_names,
            deferred_scope=deferred_scope,
        )
        return deferred_scope
    scope: dict[str, Any] = {}
    bind_imports(
        parsed_tree=parsed_tree,
        file_path=file_path,
        package_name=package_name,
        scope=scope,
    )
    return scope


def bind_imports(
    *,
    parsed_tree: ast.AST,
    file_path: str | None,
    package_name: str | None,
    scope: dict[str, Any],
    deferred_scope: DeferredImportScope | None = None,
) -> None:
    """
    Run the module-level imports of a tree and bind the names they import in a scope.

    :param parsed_tree: The tree whose imports run.
    :param file_path: The file the tree was parsed from.
    :param package_name: The package that relative imports resolve against.
    :param scope: The scope the imported names are bound in.
    :param deferred_scope: Where the imports in ``if TYPE_CHECKING:`` blocks wait for a
        name they bind to be looked up. They run with the others when ``None``.
    """
    imports_that_never_run = set(type_checking_imports(parsed_tree))

    for node in module_level_imports(parsed_tree):
        if deferred_scope is not None and node in imports_that_never_run:
            deferred_scope.deferred_imports.append(
                DeferredImport(
                    node=node, file_path=file_path, package_name=package_name
                )
            )
        elif isinstance(node, ast.Import):
            _handle_import_node(
                node=node,
                scope=scope,
                file_path=file_path,
                package_name=package_name,
                skips_unimportable_module=node in imports_that_never_run,
            )
        elif isinstance(node, ast.ImportFrom):
            package_name = _handle_import_from_node(
                node=node,
                scope=scope,
                file_path=file_path,
                package_name=package_name,
            )


@dataclass(kw_only=True)
class DeferredImport:
    """
    An import statement held back until a name it binds is looked up.
    """

    node: ast.Import | ast.ImportFrom
    """
    The import statement.
    """

    file_path: str | None
    """
    The file the statement was found in.
    """

    package_name: str | None
    """
    The package that a relative import resolves against.
    """

    def binds(self, *, name: str) -> bool:
        """
        :return: Whether the statement binds *name*; a star import may bind any.
        """
        return any(
            alias.name == "*" or (alias.asname or alias.name) == name
            for alias in self.node.names
        )


@dataclass(kw_only=True)
class DeferredImportScope(Mapping[str, Any]):
    """
    A scope whose deferred import statements run when a name they bind is first looked
    up.
    """

    bound_names: dict[str, Any] = field(default_factory=dict)
    """
    The names the statements run so far have bound.
    """

    deferred_imports: list[DeferredImport] = field(default_factory=list)
    """
    The statements that have not run yet.
    """

    def run_deferred_imports_binding(self, *, name: str) -> None:
        """
        Run the deferred statements that bind *name*, each at most once.
        """
        binding = [
            deferred for deferred in self.deferred_imports if deferred.binds(name=name)
        ]
        for deferred in binding:
            self.deferred_imports.remove(deferred)
            if isinstance(deferred.node, ast.Import):
                _handle_import_node(
                    node=deferred.node,
                    scope=self.bound_names,
                    file_path=deferred.file_path,
                    package_name=deferred.package_name,
                    skips_unimportable_module=True,
                )
            else:
                _handle_import_from_node(
                    node=deferred.node,
                    scope=self.bound_names,
                    file_path=deferred.file_path,
                    package_name=deferred.package_name,
                )

    def absorb(self, *, other: DeferredImportScope) -> None:
        """
        Take over the names *other* has bound and the statements it still defers.
        """
        self.bound_names.update(other.bound_names)
        self.deferred_imports.extend(
            deferred
            for deferred in other.deferred_imports
            if deferred not in self.deferred_imports
        )

    def __getitem__(self, name: str) -> Any:
        """
        :return: The object *name* is bound to, after running the deferred statements
            that bind it.
        :raises KeyError: If no statement binds it.
        """
        if name not in self.bound_names:
            self.run_deferred_imports_binding(name=name)
        return self.bound_names[name]

    def __contains__(self, name: object) -> bool:
        """
        :return: Whether *name* is bound, after running the deferred statements that
            bind it.
        """
        if isinstance(name, str) and name not in self.bound_names:
            self.run_deferred_imports_binding(name=name)
        return name in self.bound_names

    def __iter__(self) -> Iterator[str]:
        """
        :return: The names bound so far.
        """
        return iter(self.bound_names)

    def __len__(self) -> int:
        """
        :return: The number of names bound so far.
        """
        return len(self.bound_names)


def module_level_imports(node: ast.AST) -> Iterator[ast.Import | ast.ImportFrom]:
    """
    Yield the import statements that bind names in the module namespace.

    Imports inside functions, lambdas and class bodies are skipped, since the names they
    bind are local to that body.

    :param node: The node whose statements are searched.
    :return: The module-level import statements, including those nested in compound
        statements such as ``if TYPE_CHECKING:`` or ``try``.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            yield child
        elif not isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            yield from module_level_imports(child)


def type_checking_imports(node: ast.AST) -> Iterator[ast.Import | ast.ImportFrom]:
    """
    Yield the module-level import statements that only run while type checking.

    :param node: The node whose statements are searched.
    :return: The module-level import statements inside ``if TYPE_CHECKING:`` blocks,
        whether the condition names ``TYPE_CHECKING`` directly or as a module attribute.
    """
    for child in ast.iter_child_nodes(node):
        if (
            isinstance(child, ast.If)
            and ast.unparse(child.test).rpartition(".")[2] == "TYPE_CHECKING"
        ):
            yield from module_level_imports(
                ast.Module(body=child.body, type_ignores=[])
            )
        elif not isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            yield from type_checking_imports(child)


def get_and_import_module(
    module_name: str, package_name: str | None
) -> types.ModuleType:
    """
    Attempt to import a module with an optional package context and return the module or
    raise.

    :param module_name: The name of the module to import.
    :param package_name: The package name to use for relative imports, or None for
        absolute imports.
    :return: The imported module.
    :raises ModuleNotFoundError: If the module cannot be found.
    """
    module = get_module_object(module_name, package_name)
    if module is not None:
        return module

    try:
        return importlib.import_module(module_name, package=package_name)
    except ModuleNotFoundError as e:
        if not package_name:
            raise e
        if module_name.startswith(".") and package_name:
            full_name = resolve_name(module_name, package_name)
        else:
            full_name = f"{package_name}.{module_name}"
        if full_name in sys.modules:
            return sys.modules[full_name]
        return importlib.import_module(full_name)


def get_module_object(
    module_name: str, package_name: str | None = None
) -> types.ModuleType | None:
    """
    :param module_name: The name of the module to import.
    :param package_name: The name of the package containing the module.
    :return: The imported module object.
    """
    try:
        full_name = module_name
        if module_name.startswith(".") and package_name:
            full_name = resolve_name(module_name, package_name)

        if full_name in sys.modules:
            return sys.modules[full_name]
    except Exception:
        return None


def _resolve_relative_import(
    file_path: str | None,
    node: ast.ImportFrom,
    module_name: str | None,
    package_name: str | None,
) -> tuple[str | None, str | None]:
    """
    Resolve relative import context and possibly adjust module and package names based
    on file location.

    :param file_path: The path to the file containing the import statement.
    :param node: The import from node to process.
    :param module_name: The module name to resolve.
    :param package_name: The package name to use for relative imports.
    :return: a tuple of (resolved_module_name, resolved_package_name).
    """
    resolved_module = module_name
    resolved_package = package_name

    if node.level > 0 and file_path is not None:
        try:
            resolved_package = get_import_path_from_path(
                Path(os.path.join(file_path, *[".."] * node.level)).resolve()
            )
        except Exception:
            # Keep original package if resolution fails
            pass

        if resolved_package is not None and module_name is not None:
            try:
                module_rel_path = Path(
                    os.path.join(file_path, *[".."] * node.level, module_name)
                ).resolve()
                idx = str(module_rel_path).rfind(resolved_package)
                if idx != -1:
                    resolved_module = str(module_rel_path)[idx:].replace(
                        os.path.sep, "."
                    )
            except Exception:
                # Fall back to original module name
                pass

    return resolved_module, resolved_package


def _handle_import_node(
    node: ast.Import,
    scope: dict[str, Any],
    file_path: str | None,
    package_name: str | None,
    skips_unimportable_module: bool,
) -> None:
    """
    Process a standard import node and update the provided scope mapping.

    :param node: The import node to process.
    :param scope: The scope mapping to update.
    :param file_path: The path to the file containing the import statement.
    :param package_name: The package name to use for relative imports.
    :param skips_unimportable_module: Whether a module that cannot be imported
        contributes no name instead of raising.
    """
    for alias in node.names:
        asname = alias.asname or alias.name
        try:
            scope[asname] = get_and_import_module(alias.name, package_name)
        except ImportError as error:
            if not skips_unimportable_module:
                raise
            _log_unresolvable_import_once(alias.name, asname, file_path, str(error))


@lru_cache(maxsize=None)
def _log_unresolvable_import_once(
    resolved_module_name: str | None,
    name: str,
    file_path: str | None,
    error_message: str,
) -> None:
    """
    Log, at most once per ``(resolved_module_name, name, file_path)``, that a name could
    not be imported while extracting a file's imports.

    The identical failure repeats for every class that needs a name from a module still
    initializing in a circular import, so the repeats collapse to one log line.

    :param resolved_module_name: The module the failed import targeted.
    :param name: The name that could not be bound from that module.
    :param file_path: The path of the file whose imports were being extracted.
    :param error_message: The message of the error that was raised.
    """
    logger.debug(
        f"Could not import {resolved_module_name}: {error_message} while extracting imports from {file_path}"
    )


def _handle_import_from_node(
    node: ast.ImportFrom,
    scope: dict[str, Any],
    file_path: str | None,
    package_name: str | None,
) -> str | None:
    """
    Process a from-import node and update the provided scope mapping.

    A statement whose module cannot be imported contributes no names, so one that cannot
    be bound does not cost the caller the others. A module a generator is about to write
    is absent while it runs.

    :param node: The from-import node to process.
    :param scope: The scope mapping to update.
    :param file_path: The path to the file containing the import statement.
    :param package_name: The package name to use for relative imports.
    :return: the (possibly) updated package_name to mimic original behavior.
    """
    module_name = node.module

    # Resolve relative imports (may update package_name and module_name)
    resolved_module_name, resolved_package_name = _resolve_relative_import(
        file_path=file_path,
        node=node,
        module_name=module_name,
        package_name=package_name,
    )

    # Mimic original behavior: allow package_name to be overwritten for subsequent iterations
    package_name = resolved_package_name

    try:
        module = None
        if resolved_module_name is not None:
            module = get_and_import_module(resolved_module_name, package_name)

        if module is None and resolved_package_name and resolved_module_name:
            # Fallback already attempted in _import_module_safely; keep for parity
            module = get_and_import_module(
                f"{resolved_package_name}.{resolved_module_name}", None
            )
    except ImportError as error:
        for alias in node.names:
            _log_unresolvable_import_once(
                resolved_module_name, alias.name, file_path, str(error)
            )
        return package_name

    for alias in node.names:
        name = alias.name
        asname = alias.asname or name
        try:
            if name == "*":
                scope.update(module.__dict__)
            else:
                scope[asname] = getattr(module, name)
        except AttributeError as e:
            _log_unresolvable_import_once(resolved_module_name, name, file_path, str(e))

    return package_name
