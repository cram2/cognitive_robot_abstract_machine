"""
Orchestrator that collects and emits import statements for generated modules.
"""

from __future__ import annotations

import dataclasses
from types import ModuleType

import libcst
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import AddImportsVisitor

from krrood.patterns.code_generation.import_name_resolver import ImportNameResolver
from krrood.patterns.code_generation.libcst_node_factory import LibCSTNodeFactory

_TYPING_MODULES: frozenset[str] = frozenset({"typing", "typing_extensions"})
_NON_IMPORTABLE_MODULES: frozenset[str] = frozenset(
    {"typing", "typing_extensions", "builtins"}
)
_EXCLUDED_IMPORT_NAMES: frozenset[str] = frozenset(
    {"dataclass", "field", "ABC", "abstractmethod", "TYPE_CHECKING"}
)


class NameCollector(libcst.CSTVisitor):
    """Collects all Name node values encountered during a CST traversal."""

    def __init__(self):
        self.names: set[str] = set()

    def visit_Name(self, node: libcst.Name) -> None:
        self.names.add(node.value)


class RuntimeNameCollector(libcst.CSTVisitor):
    """Collects names that appear inside decorator expressions."""

    def __init__(self):
        self.names: set[str] = set()

    def visit_Decorator(self, node: libcst.Decorator) -> None:
        """Record all names found inside a decorator expression."""
        collector = NameCollector()
        node.visit(collector)
        self.names.update(collector.names)


@dataclasses.dataclass
class GeneratedModuleImportOrchestrator:
    """
    Orchestrates the collection and emission of import statements for generated modules.
    """

    generated_context: CodemodContext
    original_context: CodemodContext
    resolver: ImportNameResolver
    source_module: ModuleType

    def require_import(self, module: str, names: str | list[str]) -> None:
        """Record an import that must appear in the generated module.

        :param module: The module to import from.
        :param names: The name or list of names to import.
        """
        if module in ["builtins", self.source_module.__name__]:
            return
        if isinstance(names, str):
            names = [names]
        for name in names:
            AddImportsVisitor.add_needed_import(
                self.generated_context,
                module=module,
                obj=name,
            )

    def require_original_import(
        self, module: str, obj: str | list[str] | None = None
    ) -> None:
        """Record an import that must appear in the transformed original module.

        :param module: The module to import from.
        :param obj: The name or names to import from the module.
        """
        if module in ["builtins", self.source_module.__name__]:
            return
        if obj is None:
            AddImportsVisitor.add_needed_import(self.original_context, module)
        elif isinstance(obj, str):
            AddImportsVisitor.add_needed_import(self.original_context, module, obj)
        else:
            for name in obj:
                AddImportsVisitor.add_needed_import(self.original_context, module, name)

    def build_generated_module(
        self,
        updated_module_node: libcst.Module,
        generated_classes: list[libcst.ClassDef],
        factory: LibCSTNodeFactory,
    ) -> libcst.Module:
        """Build the complete generated module AST with all imports.

        :param updated_module_node: The transformed source module node (used for header/footer).
        :param generated_classes: The generated class nodes to include.
        :param factory: The node factory for creating CST nodes.
        :return: A complete Module node ready to emit as source code.
        """
        used_names = self._collect_used_names(generated_classes)
        self._add_required_imports(used_names)
        runtime_names = self._collect_runtime_names(generated_classes)
        top_level_names, type_checking_names = self._classify_import_names(
            used_names, runtime_names
        )

        self._add_typing_imports(used_names)
        self._add_runtime_imports(top_level_names, generated_classes)

        module_body = [self._create_future_annotations_import()]

        type_checking_block = self._create_type_checking_block(
            type_checking_names, generated_classes, factory
        )
        if type_checking_block:
            module_body.append(type_checking_block)

        module_body.extend(generated_classes)

        return libcst.Module(
            body=module_body,
            header=updated_module_node.header,
            footer=updated_module_node.footer,
        )

    def _classify_import_names(
        self, used_names: set[str], runtime_names: set[str]
    ) -> tuple[set[str], set[str]]:
        """Partition used names into top-level and TYPE_CHECKING groups.

        :param used_names: All names referenced inside the generated classes.
        :param runtime_names: Names that must be available at runtime (e.g. decorator names).
        :return: A tuple of (top_level_names, type_checking_names).
        """
        return runtime_names, used_names - runtime_names

    def _collect_used_names(
        self, generated_classes: list[libcst.ClassDef]
    ) -> set[str]:
        """Return all identifier names referenced inside the given generated classes."""
        used_names: set[str] = set()
        for class_def in generated_classes:
            collector = NameCollector()
            class_def.visit(collector)
            used_names.update(collector.names)
        return used_names

    def _collect_runtime_names(self, generated_classes: list[libcst.ClassDef]) -> set[str]:
        """Return all names used inside decorator expressions in the given classes."""
        collector = RuntimeNameCollector()
        for class_def in generated_classes:
            class_def.visit(collector)
        return collector.names

    def _add_required_imports(self, used_names: set[str] | None = None) -> None:
        """Record the standard imports that every generated module needs."""
        dataclass_names = ["dataclass"]
        if used_names and "field" in used_names:
            dataclass_names.append("field")
        self.require_import("dataclasses", dataclass_names)
        self.require_import("abc", ["ABC", "abstractmethod"])
        self.require_import("typing_extensions", ["TYPE_CHECKING"])

    def _add_typing_imports(self, used_names: set[str]) -> None:
        """Record imports for names whose source module is typing or typing_extensions."""
        for name in used_names:
            module = self.resolver.resolve(name)
            if module in _TYPING_MODULES:
                self.require_import(module, name)

    def _add_runtime_imports(
        self, names: set[str], generated_classes: list[libcst.ClassDef]
    ) -> None:
        """Record top-level imports for names that must be available at runtime."""
        defined_names = {cd.name.value for cd in generated_classes}
        for name in names:
            if name in _EXCLUDED_IMPORT_NAMES or name in defined_names:
                continue
            module_name = self.resolver.resolve(name)
            if module_name:
                self.require_import(module_name, name)

    def _create_future_annotations_import(self) -> libcst.SimpleStatementLine:
        """Return a ``from __future__ import annotations`` statement node."""
        return libcst.SimpleStatementLine(
            body=[
                libcst.ImportFrom(
                    module=libcst.Name("__future__"),
                    names=[libcst.ImportAlias(name=libcst.Name("annotations"))],
                )
            ]
        )

    def _build_import_map(
        self, used_names: set[str], generated_classes: list[libcst.ClassDef]
    ) -> dict[str, set[str]]:
        """Return a mapping of module name to the set of names to import."""
        defined_names = {cd.name.value for cd in generated_classes}
        import_map: dict[str, set[str]] = {}
        for name in used_names:
            if name in _EXCLUDED_IMPORT_NAMES or name in defined_names:
                continue
            module_name = self.resolver.resolve(name)
            if module_name and module_name not in _NON_IMPORTABLE_MODULES:
                import_map.setdefault(module_name, set()).add(name)
        return import_map

    def _create_type_checking_block(
        self,
        used_names: set[str],
        generated_classes: list[libcst.ClassDef],
        factory: LibCSTNodeFactory,
    ) -> libcst.If | None:
        """Build an ``if TYPE_CHECKING:`` block containing all non-runtime imports."""
        import_map = self._build_import_map(used_names, generated_classes)
        if not import_map:
            return None

        type_checking_body = [
            self._create_import_from_node(module_name, names, factory)
            for module_name, names in sorted(import_map.items())
        ]

        return libcst.If(
            test=libcst.Name("TYPE_CHECKING"),
            body=libcst.IndentedBlock(body=type_checking_body),
        )

    def _create_import_from_node(
        self, module_name: str, names: set[str], factory: LibCSTNodeFactory
    ) -> libcst.SimpleStatementLine:
        """Return a ``from <module> import <names>`` CST node."""
        return libcst.SimpleStatementLine(
            body=[
                libcst.ImportFrom(
                    module=factory.to_cst_expression(module_name) if module_name else None,
                    names=[
                        libcst.ImportAlias(name=libcst.Name(n)) for n in sorted(names)
                    ],
                    relative=[],
                )
            ]
        )
