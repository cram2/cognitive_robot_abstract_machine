"""
Role transformer: converts role-pattern modules into mixin-based equivalents.
"""

from __future__ import annotations

import dataclasses
import enum
import sys
from abc import ABC
from pathlib import Path
from types import ModuleType
from typing import (
    Any,
    Callable,
    _GenericAlias,
)

import libcst
from libcst.codemod import ContextAwareTransformer, CodemodContext
from libcst.codemod.visitors import AddImportsVisitor
from typing_extensions import Dict, Type, get_args, get_origin

from krrood import logger
from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.exceptions import ClassIsUnMappedInClassDiagram
from krrood.class_diagrams.utils import (
    classes_of_module,
    topological_sort_by_inheritance,
    same_package,
)
from krrood.patterns.code_generation import (
    LibCSTNodeFactory,
    TypeNormaliser,
    ImportNameResolver,
    GeneratedModuleImportOrchestrator,
    GeneratedCodeFileWriter,
    DelegationGenerator,
)
from krrood.patterns.property_delegator import PropertyDelegator
from krrood.patterns.role.exceptions import RoleTransformerError
from krrood.patterns.role.meta_data import RoleType
from krrood.patterns.role.role import Role, HasRoles

DELEGATEE_ATTR = "delegatee"
"""
Attribute name used to access the delegatee instance via the generated mixin.
"""
ROLE_MIXINS_FOLDER = "role_mixins"
"""
Folder name for generated role mixins.
"""
ROLE_MIXINS_SUFFIX = "_role_mixins"
"""
Suffix for generated role mixin file names.
"""

_ALWAYS_EXCLUDED_METHODS: frozenset[str] = frozenset(
    {"__init__", "__post_init__", "__new__"}
)


# __new__ is defined on Symbol (a Role base) as a staticmethod; inspect.getmembers
# unwraps it to a plain function, so it must be excluded explicitly here.


class TransformationMode(str, enum.Enum):
    """Enumeration of transformation mode identifiers used as file-name prefixes."""

    GROUND_TRUTH = "_ground_truth_"
    TRANSFORMED = "transformed_"


def _compute_all_delegatees(
    class_diagram: ClassDiagram,
    pd_only_delegatees: set[type],
) -> set[type]:
    """Return the set of all delegatee types including transitive same-package ancestors.

    Direct role takers and pd_only delegatees are the starting set.  For each of
    those, same-package ancestors discovered through the MRO are transitively
    added so that every ancestor module gets a chance to generate the
    ``DelegatorFor`` mixin for its own classes before a derived module would
    otherwise claim ownership.
    """
    from krrood.class_diagrams.utils import same_package

    direct = set(class_diagram.role_takers) | pd_only_delegatees
    result = set(direct)
    for delegatee in direct:
        for ancestor in delegatee.__mro__:
            if ancestor is object or ancestor is delegatee:
                continue
            if not same_package(ancestor.__module__, delegatee.__module__):
                continue
            result.add(ancestor)
    return result


def _mixin_module_dotted_name(module_dotted_name: str) -> str:
    """Return the fully-qualified module name for the generated mixin module."""
    parts = module_dotted_name.split(".")
    package = ".".join(parts[:-1])
    leaf = parts[-1]
    return f"{package}.{ROLE_MIXINS_FOLDER}.{leaf}{ROLE_MIXINS_SUFFIX}"


def _is_from_property_delegator_class(name: str, clazz: type) -> bool:
    """Return True if *name* is inherited from the PropertyDelegator hierarchy without being overridden.

    :param name: The attribute name to look up.
    :param clazz: The class whose MRO is searched.
    :return: True if the first defining class in the MRO is a PropertyDelegator subclass.
    """
    for klass in clazz.__mro__:
        if name in vars(klass):
            return issubclass(klass, PropertyDelegator)
    return False


def _is_role_base_node(base_node: libcst.BaseExpression) -> bool:
    """Return True if the base node represents the Role class (for HasRoles injection checks)."""
    name = LibCSTNodeFactory.get_name_from_base_node(base_node)
    return name == "Role"


def _sort_modules_by_dependency(
    modules: list[ModuleType], class_diagram: ClassDiagram
) -> list[ModuleType]:
    """Return modules in topological order so that modules containing base classes come first.

    If role taker T (in module B) inherits from class C (in module A), module A is placed
    before module B. This ensures base-class RoleFor nodes are generated before they are
    referenced by derived-class transformers.
    """
    module_set = set(modules)
    deps: dict[ModuleType, set[ModuleType]] = {m: set() for m in modules}
    for taker in class_diagram.role_takers:
        taker_module = sys.modules.get(taker.__module__)
        if taker_module not in module_set:
            continue
        for ancestor in taker.__mro__[1:]:
            if ancestor is object:
                continue
            ancestor_module = sys.modules.get(ancestor.__module__)
            if (
                ancestor_module is not None
                and ancestor_module in module_set
                and ancestor_module is not taker_module
            ):
                deps[taker_module].add(ancestor_module)

    result: list[ModuleType] = []
    visited: set[ModuleType] = set()

    def visit(m: ModuleType) -> None:
        if m in visited:
            return
        visited.add(m)
        for dep in deps.get(m, set()):
            visit(dep)
        result.append(m)

    for m in modules:
        visit(m)
    return result


@dataclasses.dataclass
class RoleTransformer:
    """
    Transforms role-pattern modules into mixin-based equivalents and generates
    the corresponding RoleFor mixin classes for each role taker.
    """

    module: ModuleType
    taker_modules: list[ModuleType] = dataclasses.field(default_factory=list)
    class_diagram: ClassDiagram = dataclasses.field(init=False)
    pd_only_delegatees: set[type] = dataclasses.field(init=False, default_factory=set)
    path: Path | None = None
    file_name_prefix: str = ""

    def __post_init__(self):
        """Set up the transformer for the given module."""
        if self.path is None:
            self.path = self.get_generated_file_path(self.module)
        self._refresh_diagram()

    def _refresh_diagram(self) -> None:
        """Sync the class diagram and taker modules list with the current module state."""
        self.class_diagram, self.taker_modules, self.pd_only_delegatees = (
            self._build_role_diagram(self.module, self.taker_modules)
        )

    @classmethod
    def _build_role_diagram(
        cls,
        module: ModuleType,
        taker_modules: list[ModuleType],
    ) -> tuple[ClassDiagram, list[ModuleType], set[type]]:
        """Build a ClassDiagram for the module, auto-discovering role taker modules.

        :param module: The primary module containing role classes.
        :param taker_modules: The initial list of known taker modules.
        :return: A tuple of (ClassDiagram, updated taker_modules, pd_only_delegatees) where
            pd_only_delegatees is the set of delegatee types used exclusively by non-Role
            PropertyDelegator subclasses (these get DelegatorFor mixins but not HasRoles).
        """
        classes = classes_of_module(module)
        role_classes = [clazz for clazz in classes if issubclass(clazz, Role)]
        pd_only_classes = [
            clazz
            for clazz in classes
            if issubclass(clazz, PropertyDelegator) and not issubclass(clazz, Role)
        ]
        updated_taker_modules = list(taker_modules)

        def add_delegatee_class(delegatee_class: Type):
            classes.append(delegatee_class)
            delegatee_module = sys.modules[delegatee_class.__module__]
            if delegatee_module not in updated_taker_modules:
                updated_taker_modules.append(delegatee_module)

        for clazz in role_classes:
            role_taker_type = clazz.get_role_taker_type()
            if role_taker_type not in classes:
                add_delegatee_class(role_taker_type)

        pd_only_delegatees: set[type] = set()
        for clazz in pd_only_classes:
            delegatee_type = clazz.get_delegatee_type()
            if delegatee_type not in classes:
                add_delegatee_class(delegatee_type)
            pd_only_delegatees.add(delegatee_type)

        # Discover transitive same-package ancestors of all delegatee classes so
        # their modules also get a RoleModuleTransformer and can generate the
        # DelegatorFor mixins for their own classes.
        delegatee_classes = [
            c for c in classes if c not in role_classes and c not in pd_only_classes
        ]
        for delegatee in list(delegatee_classes):
            for ancestor in delegatee.__mro__:
                if ancestor is object or ancestor is delegatee:
                    continue
                if not same_package(ancestor.__module__, delegatee.__module__):
                    continue
                if ancestor not in classes:
                    add_delegatee_class(ancestor)

        return ClassDiagram(classes), updated_taker_modules, pd_only_delegatees

    def transform(self, write: bool = False) -> dict[ModuleType, tuple[str, str]]:
        """Transform the module and its taker modules, generating mixins for each role taker.

        :param write: When True, writes the generated files to the file system and formats them.
        :return: A dictionary mapping each transformed module to a tuple of its transformed
                 module content and its mixin module content.
        """
        import importlib

        all_modules = list(self.taker_modules)
        if self.module not in all_modules:
            all_modules.append(self.module)

        all_modules = _sort_modules_by_dependency(all_modules, self.class_diagram)
        for module in all_modules:
            importlib.reload(module)
        self._refresh_diagram()

        all_modules = _sort_modules_by_dependency(all_modules, self.class_diagram)

        global_base_class_ownership: dict[type, ModuleType] = {}
        all_modules_sources = {}
        for module in all_modules:
            with open(self.get_module_file_path(module), "r") as f:
                source = f.read()

            context = CodemodContext()

            transformer = RoleModuleTransformer(
                context=context,
                class_diagram=self.class_diagram,
                module=module,
                taker_modules=self.taker_modules,
                file_name_prefix=self.file_name_prefix,
                global_base_class_ownership=global_base_class_ownership,
                pd_only_delegatees=self.pd_only_delegatees,
            )
            tree = libcst.parse_module(source)

            mixin_result = transformer.transform_module(tree)
            mixin_result = AddImportsVisitor(context).transform_module(mixin_result)

            transformed_original = transformer.transformed_module
            transformed_original = AddImportsVisitor(
                transformer.original_context
            ).transform_module(transformed_original)

            all_modules_sources[module] = (transformed_original.code, mixin_result.code)

        if write:
            writer = GeneratedCodeFileWriter()
            writer.write(all_modules_sources, self.get_generated_file_path)

        return all_modules_sources

    def __hash__(self):
        return hash((self.__class__, self.module))

    def __eq__(self, other):
        return hash(self) == hash(other)

    @staticmethod
    def get_module_file_path(module: ModuleType) -> Path:
        """Return the file path of the given module.

        :param module: The module whose file path is needed.
        :return: Path to the module file.
        """
        return Path(sys.modules[module.__name__].__file__)

    @staticmethod
    def _normalize_file_prefix(prefix: str) -> str:
        """Return the prefix with a trailing underscore, adding one if absent.

        :param prefix: The raw file name prefix string.
        :return: The normalised prefix string.
        """
        if prefix and not prefix.endswith("_"):
            return f"{prefix}_"
        return prefix

    def get_generated_file_path(
        self, module: ModuleType, is_mixin: bool = False
    ) -> Path:
        """Return the path where the generated file for the module should be written.

        :param module: The module for which to compute the generated path.
        :param is_mixin: Whether the path is for the mixin file rather than the transformed original.
        :return: Path to the generated file.
        """
        parent_directory = Path(self.get_module_file_path(module)).parent
        module_name = module.__name__.split(".")[-1]
        if is_mixin:
            role_mixins_folder = parent_directory / ROLE_MIXINS_FOLDER
            filename = f"{module_name}{ROLE_MIXINS_SUFFIX}.py"
            return role_mixins_folder / filename
        else:
            prefix = self._normalize_file_prefix(self.file_name_prefix)
            filename = f"{prefix}{module_name}.py"
            return parent_directory / filename


class RoleModuleTransformer(ContextAwareTransformer):
    """
    Applies role pattern transformations to a Python module AST and generates
    the corresponding mixin module.
    """

    def __init__(
        self,
        context: CodemodContext,
        class_diagram: ClassDiagram,
        module: ModuleType,
        taker_modules: list[ModuleType],
        file_name_prefix: str = "",
        global_base_class_ownership: dict[type, ModuleType] | None = None,
        pd_only_delegatees: set[type] | None = None,
    ):
        """Initialise the transformer with the class diagram and module context.

        :param context: The codemod context for import tracking.
        :param class_diagram: The class diagram describing all relevant classes.
        :param module: The module being transformed.
        :param taker_modules: All modules that contain role taker classes.
        :param file_name_prefix: Prefix applied to generated file names.
        :param global_base_class_ownership: Shared dict mapping base class to the module that owns its DelegatorFor node.
            When provided, prevents duplicate DelegatorFor generation across multiple module transformers.
        :param pd_only_delegatees: Delegatee types used by non-Role PropertyDelegator subclasses.
            These receive DelegatorFor mixins but not HasRoles injection.
        """
        super().__init__(context)
        self.class_diagram = class_diagram
        self.source_module = module
        self.taker_modules = taker_modules
        self.file_name_prefix = file_name_prefix
        self.role_for: dict[WrappedClass, libcst.ClassDef] = {}
        self._base_class_role_for_nodes: dict[type, libcst.ClassDef] = {}
        self._global_base_class_ownership: dict[type, ModuleType] = (
            global_base_class_ownership
            if global_base_class_ownership is not None
            else {}
        )
        self._cross_module_rolefor_bases: dict[type, str] = {}
        self._pd_only_delegatees: set[type] = pd_only_delegatees or set()
        self._all_delegatees: set[type] = _compute_all_delegatees(
            class_diagram, self._pd_only_delegatees
        )
        self.transformed_module: libcst.Module | None = None
        self.original_context = CodemodContext()
        self.current_class: type | None = None
        self._factory = LibCSTNodeFactory()
        self._resolver = ImportNameResolver(
            source_module=module,
            companion_modules=list(taker_modules),
            class_diagram=class_diagram,
        )
        self._normaliser = TypeNormaliser(
            resolver=self._resolver,
            class_diagram=class_diagram,
            class_name_getter=self._role_class_name_getter(),
        )
        self._import_orchestrator = GeneratedModuleImportOrchestrator(
            generated_context=context,
            original_context=self.original_context,
            resolver=self._resolver,
            source_module=module,
        )
        self._delegation_generator = DelegationGenerator(
            delegatee_attribute_name=DELEGATEE_ATTR,
            node_factory=self._factory,
            type_normaliser=self._normaliser,
            already_covered_bases=self._all_delegatees,
            excluded_method_names=_ALWAYS_EXCLUDED_METHODS,
            excluded_member_predicate=_is_from_property_delegator_class,
            is_excluded_defining_class=lambda klass: issubclass(
                klass, PropertyDelegator
            ),
            name_resolver=self._resolver,
        )

    def _role_class_name_getter(self) -> Callable[[type], str]:
        """Return a class name getter that uses the TypeVar naming convention for role classes."""

        def getter(clazz: type) -> str:
            if issubclass(clazz, Role):
                type_var_name = f"T{clazz.__name__}"
                class_module = sys.modules.get(clazz.__module__)
                if class_module is not None and type_var_name in class_module.__dict__:
                    return type_var_name
            return clazz.__name__

        return getter

    def _get_role_taker_type_name(self, clazz: type) -> str:
        """Return the TypeVar name for a class if one exists in its module, else the plain name.

        :param clazz: The class whose type name is needed.
        :return: The TypeVar name or the plain class name.
        """
        type_var_name = f"T{clazz.__name__}"
        class_module = sys.modules.get(clazz.__module__)
        if class_module is not None and type_var_name in class_module.__dict__:
            return type_var_name
        return clazz.__name__

    def require_original_import(
        self, module: str, obj: str | list[str] | None = None
    ) -> None:
        """Record an import that must appear in the transformed original module.

        :param module: The module to import from.
        :param obj: The name or names to import from the module.
        """
        self._import_orchestrator.require_original_import(module, obj)

    def leave_ClassDef(
        self, original_node: libcst.ClassDef, updated_node: libcst.ClassDef
    ) -> libcst.ClassDef | libcst.FlattenSentinel[libcst.BaseCompoundStatement]:
        """Handle class-level transformations for role takers and role classes.

        :param original_node: The original class node before any transformations.
        :param updated_node: The class node after child transformations.
        :return: The transformed class node or a flattened sentinel with multiple nodes.
        """
        wrapped_class = self._find_wrapped_class(updated_node.name.value)
        if wrapped_class is None:
            return updated_node

        result_nodes = [updated_node]
        if wrapped_class.clazz in self._all_delegatees:
            result_nodes = self._handle_taker_transformation(
                updated_node, wrapped_class
            )
            updated_node = result_nodes[0]

        role_type = RoleType.get_role_type(wrapped_class)
        if role_type != RoleType.NOT_A_ROLE:
            updated_node = self._handle_role_transformation(updated_node, wrapped_class)
            result_nodes[0] = updated_node

        if len(result_nodes) > 1:
            return libcst.FlattenSentinel(result_nodes)
        return result_nodes[0]

    def _find_wrapped_class(self, class_name: str) -> WrappedClass | None:
        """Return the WrappedClass with the given name, or None if not found."""
        for wrapped_class in self.class_diagram.wrapped_classes:
            if wrapped_class.clazz.__name__ == class_name:
                return wrapped_class
        return None

    def _handle_taker_transformation(
        self, node: libcst.ClassDef, wrapped_class: WrappedClass
    ) -> list[libcst.ClassDef]:
        """Apply role-taker transformation, tracking current_class for import resolution."""
        self.current_class = wrapped_class.clazz
        result = self._transform_role_taker(node, wrapped_class)
        self.current_class = None
        return result

    def _handle_role_transformation(
        self, node: libcst.ClassDef, wrapped_class: WrappedClass
    ) -> libcst.ClassDef:
        """Apply role transformation, tracking current_class for import resolution."""
        self.current_class = wrapped_class.clazz
        result = self._transform_role(node, wrapped_class)
        self.current_class = None
        return result

    def leave_ImportFrom(
        self, original_node: libcst.ImportFrom, updated_node: libcst.ImportFrom
    ) -> libcst.ImportFrom:
        """Rewrite import statements: resolve relative imports and prefix transformed module names.

        :param original_node: The original import node.
        :param updated_node: The import node after child transformations.
        :return: The rewritten import node.
        """
        updated_node = self._resolve_relative_import(updated_node)
        return self._rewrite_prefixed_module_name(updated_node)

    def _resolve_relative_import(self, node: libcst.ImportFrom) -> libcst.ImportFrom:
        """Resolve a relative import to an absolute import path."""
        if len(node.relative) == 0:
            return node
        current_module_parts = self.source_module.__name__.split(".")
        is_package = hasattr(self.source_module, "__path__")
        package_parts = (
            current_module_parts if is_package else current_module_parts[:-1]
        )

        levels_up = len(node.relative) - 1
        if levels_up > 0:
            package_parts = package_parts[:-levels_up]

        base_module = ".".join(package_parts)
        module_name = self._get_module_name_str(node.module)

        if module_name:
            absolute_module = (
                f"{base_module}.{module_name}" if base_module else module_name
            )
        else:
            absolute_module = base_module

        return node.with_changes(
            relative=[],
            module=(
                LibCSTNodeFactory.to_cst_expression(absolute_module)
                if absolute_module
                else None
            ),
        )

    def _rewrite_prefixed_module_name(
        self, node: libcst.ImportFrom
    ) -> libcst.ImportFrom:
        """Rewrite the last module segment to include the file name prefix."""
        module_name = self._get_module_name_str(node.module)
        new_module_node = node.module

        if module_name:
            last_part = module_name.split(".")[-1]
            all_target_modules = [self.source_module] + self.taker_modules
            all_target_module_names = {
                m.__name__.split(".")[-1] for m in all_target_modules
            }
            if last_part in all_target_module_names:
                prefix = RoleTransformer._normalize_file_prefix(self.file_name_prefix)
                new_last_part = f"{prefix}{last_part}"
                new_module_node = self._update_last_module_part(
                    node.module, new_last_part
                )

        return node.with_changes(module=new_module_node)

    def _get_module_name_str(self, node: libcst.BaseExpression | None) -> str | None:
        """Extract the dotted module name string from a CST expression node."""
        if node is None:
            return None
        if isinstance(node, libcst.Name):
            return node.value
        if isinstance(node, libcst.Attribute):
            base = self._get_module_name_str(node.value)
            if base:
                return f"{base}.{node.attr.value}"
        return None

    def _update_last_module_part(
        self, node: libcst.BaseExpression, new_name: str
    ) -> libcst.BaseExpression:
        """Replace the last segment of a dotted module expression with new_name."""
        if isinstance(node, libcst.Name):
            return libcst.Name(new_name)
        if isinstance(node, libcst.Attribute):
            return node.with_changes(attr=libcst.Name(new_name))
        return node

    def leave_Module(
        self, original_node: libcst.Module, updated_node: libcst.Module
    ) -> libcst.Module:
        """Capture the transformed original module and produce the mixin module AST.

        :param original_node: The module node before any transformations.
        :param updated_node: The module node after all child transformations.
        :return: The generated mixin module AST.
        """
        self.transformed_module = updated_node
        return self._generate_mixin_module_ast(updated_node)

    def _generate_mixin_module_ast(self, updated_node: libcst.Module) -> libcst.Module:
        """Build the complete mixin module AST from the transformed node and collected mixins.

        :param updated_node: The module node after all class transformations.
        :return: A new Module node containing only the mixin classes and their imports.
        """
        sorted_base_types = topological_sort_by_inheritance(
            list(self._base_class_role_for_nodes.keys())
        )
        base_class_nodes = [
            self._base_class_role_for_nodes[k] for k in sorted_base_types
        ]
        all_mixin_classes = base_class_nodes + list(self.role_for.values())
        return self._import_orchestrator.build_generated_module(
            updated_node, all_mixin_classes, self._factory
        )

    def _resolve_name_to_module(self, name: str) -> str | None:
        """Return the source module for the given identifier name.

        :param name: The identifier to resolve.
        :return: The fully-qualified module name, or None if unresolvable.
        """
        return self._resolver.resolve(name, self.current_class)

    def _transform_role_taker(
        self, role_taker_node: libcst.ClassDef, wrapped_class: WrappedClass
    ) -> list[libcst.ClassDef]:
        """Transform a role taker class by adding HasRoles as a base if required."""
        self.make_role_for_node(role_taker_node, wrapped_class)

        if self._should_add_has_roles(role_taker_node, wrapped_class):
            role_taker_class_bases = list(role_taker_node.bases)
            if not any(
                LibCSTNodeFactory.get_name_from_base_node(base.value)
                == HasRoles.__name__
                for base in role_taker_class_bases
            ):
                role_taker_class_bases.append(
                    LibCSTNodeFactory.make_argument(HasRoles.__name__)
                )
            role_taker_node = role_taker_node.with_changes(bases=role_taker_class_bases)
            role_taker_node = self._ensure_HasRoles_init_is_called_in_explicit_init_of_role_taker_class(
                role_taker_node
            )
            self.require_original_import("krrood.patterns.role", [HasRoles.__name__])

        return [role_taker_node]

    def _ensure_HasRoles_init_is_called_in_explicit_init_of_role_taker_class(
        self, role_taker_node: libcst.ClassDef
    ) -> libcst.ClassDef:
        """Handle explicit __init__ method in a class definition.

        Ensures that HasRoles.__init__ is called in the __init__ method of the class.

        :param role_taker_node: The class definition node to handle.
        :return: The modified class definition node with HasRoles.__init__ call added if necessary.
        """
        decorator_kwargs = self._get_keyword_arguments_of_decorator_of_class_node(
            role_taker_node, "dataclass"
        )
        if (
            not decorator_kwargs
            or "init" not in decorator_kwargs
            or decorator_kwargs["init"].value != "False"
        ):
            return role_taker_node
        original_body = role_taker_node.body
        init_function_index, init_function = next(
            (
                (i, node)
                for i, node in enumerate(original_body.body)
                if isinstance(node, libcst.FunctionDef)
                and node.name.value == "__init__"
            ),
            (None, None),
        )
        if not init_function:
            return role_taker_node

        init_line_exists = any(
            isinstance(stmt, libcst.SimpleStatementLine)
            and len(stmt.body) == 1
            and isinstance(stmt.body[0], libcst.Expr)
            and isinstance(stmt.body[0].value, libcst.Call)
            and libcst.Module([]).code_for_node(stmt.body[0].value.func)
            == f"{HasRoles.__name__}.__init__"
            for stmt in init_function.body.body
        )

        if init_line_exists:
            return role_taker_node

        new_body = libcst.IndentedBlock(
            list(init_function.body.body)
            + [libcst.parse_statement(f"{HasRoles.__name__}.__init__(self)")]
        )
        init_function = init_function.with_changes(body=new_body)
        new_body = libcst.IndentedBlock(
            list(original_body.body[:init_function_index])
            + [init_function]
            + list(original_body.body[init_function_index + 1 :])
        )
        role_taker_node = role_taker_node.with_changes(body=new_body)

        return role_taker_node

    def _should_add_has_roles(
        self, node: libcst.ClassDef, wrapped_class: WrappedClass
    ) -> bool:
        """Return True if HasRoles should be added to this role taker's bases.

        :param node: The role taker class node.
        :param wrapped_class: The wrapped class of the role taker.
        :return: True only for root Role-pattern takers that do not already inherit HasRoles.
            Non-Role PropertyDelegator delegatees never receive HasRoles.
            Transitive same-package ancestors that are not direct role takers also never
            receive HasRoles — they only need DelegatorFor mixins, not HasRoles injection.
        """
        if wrapped_class.clazz in self._pd_only_delegatees:
            return False
        if wrapped_class.clazz not in self.class_diagram.role_takers:
            return False
        return not (
            any(_is_role_base_node(base.value) for base in node.bases)
            or self.bases_of_class_that_are_role_takers(wrapped_class)
        )

    def bases_of_class_that_are_role_takers(
        self, wrapped_class: WrappedClass
    ) -> dict[str, type]:
        """Return all direct base classes of the wrapped class that are also role takers.

        :param wrapped_class: Wrapped class of the role taker.
        :return: Dictionary of base class names to base class types for role takers.
        """
        return {
            base.__name__: base
            for base in wrapped_class.clazz.__bases__
            if base in self.class_diagram.role_takers
        }

    def make_role_for_node(
        self,
        node: libcst.ClassDef,
        wrapped_class: WrappedClass,
    ) -> None:
        """Create and store a DelegatorFor<Delegatee> class for the given delegatee.

        :param node: The delegatee class node to transform.
        :param wrapped_class: The wrapped class of the delegatee.
        """
        role_for_name = self.get_delegator_for_name(wrapped_class.clazz)
        role_for_node = LibCSTNodeFactory.get_renamed_node(node, role_for_name)

        all_taker_fields = self._collect_base_taker_field_names(wrapped_class)
        body_by_class = self._group_all_body_items(wrapped_class, all_taker_fields)
        segregated_base_types = self._populate_base_rolefor_nodes(body_by_class)

        role_for_bases = self.make_role_for_bases(
            role_for_node, wrapped_class, segregated_base_types
        )
        role_for_node = role_for_node.with_changes(bases=role_for_bases)

        flattened_body = self._assemble_role_for_body(
            wrapped_class, body_by_class.get(None, {})
        )
        self.role_for[wrapped_class] = LibCSTNodeFactory.get_node_with_new_body(
            role_for_node, flattened_body
        )
        self._global_base_class_ownership[wrapped_class.clazz] = self.source_module

    def _assemble_role_for_body(
        self,
        wrapped_class: WrappedClass,
        taker_direct_items: dict[str, list],
    ) -> list[libcst.FunctionDef]:
        """Return the body nodes for a DelegatorFor class.

        :param wrapped_class: The delegatee whose DelegatorFor body is being built.
        :param taker_direct_items: Members defined directly on the delegatee (not from a base class).
        :return: Flattened list of FunctionDef nodes for the class body.
        """
        taker_type_name = self._get_role_taker_type_name(wrapped_class.clazz)
        body_items: dict[str, list] = {
            DELEGATEE_ATTR: [
                LibCSTNodeFactory.make_property_getter_node(
                    DELEGATEE_ATTR, taker_type_name, "..."
                )
            ]
        }
        body_items.update(taker_direct_items)
        return [node for nodes in body_items.values() for node in nodes]

    def _collect_base_taker_field_names(self, wrapped_class: WrappedClass) -> list[str]:
        """Return all field names from base taker classes of the given wrapped class."""
        all_taker_fields = []
        for base_name, taker_type in self.bases_of_class_that_are_role_takers(
            wrapped_class
        ).items():
            wrapped_taker = self.class_diagram.get_wrapped_class(taker_type)
            all_taker_fields.extend([f.name for f in wrapped_taker.fields])
        return all_taker_fields

    def _group_all_body_items(
        self,
        wrapped_class: WrappedClass,
        all_taker_fields: list[str],
    ) -> dict[type | None, dict[str, list[libcst.FunctionDef]]]:
        """Return delegation nodes grouped by the class in the MRO that defines each item.

        :param wrapped_class: Wrapped class of the role taker.
        :param all_taker_fields: Field names already covered by base-taker RoleFor nodes.
        :return: Nested dict mapping defining class (or None for taker-direct) to name to nodes.
        """
        additional_skip_bases: set[type] | None = None
        if issubclass(wrapped_class.clazz, Role):
            additional_skip_bases = {wrapped_class.clazz.get_role_taker_type()}
        return self._delegation_generator.collect_delegation_groups(
            wrapped_class,
            self.source_module.__name__,
            already_delegated_field_names=all_taker_fields,
            additional_skip_bases=additional_skip_bases,
        )

    def _populate_base_rolefor_nodes(
        self,
        body_by_class: dict[type | None, dict[str, list]],
    ) -> list[type]:
        """Ensure a RoleFor<Base> node exists for every non-None key in body_by_class.

        When a base class was already generated by an earlier module transformer (tracked via
        ``_global_base_class_ownership``), registers a cross-module import instead of re-generating.

        :param body_by_class: Grouped body items produced by ``_group_all_body_items``.
        :return: The segregated base class types in topological order (ancestors first).
        """
        base_classes = [k for k in body_by_class if k is not None]
        if not base_classes:
            return []
        sorted_bases = topological_sort_by_inheritance(base_classes)
        for base_class in sorted_bases:
            if (
                base_class in self._base_class_role_for_nodes
                or base_class in self._cross_module_rolefor_bases
            ):
                continue
            owner_module = self._global_base_class_ownership.get(base_class)
            if owner_module is not None and owner_module is not self.source_module:
                mixin_module = _mixin_module_dotted_name(owner_module.__name__)
                role_for_name = self.get_delegator_for_name(base_class)
                self._cross_module_rolefor_bases[base_class] = mixin_module
                self.require_import(mixin_module, role_for_name)
                self._resolver.name_to_module_map[role_for_name] = mixin_module
            elif owner_module is None:
                self._base_class_role_for_nodes[base_class] = (
                    self._make_base_rolefor_node(base_class, body_by_class[base_class])
                )
                self._global_base_class_ownership[base_class] = self.source_module
        return sorted_bases

    def _make_base_rolefor_node(
        self,
        base_class: type,
        body_items: dict[str, list[libcst.FunctionDef]],
    ) -> libcst.ClassDef:
        """Generate a ``@dataclass(eq=False) class DelegatorFor<Base>(...)`` node.

        :param base_class: The base class to generate a DelegatorFor node for.
        :param body_items: Mapping of member name to list of FunctionDef nodes.
        :return: The generated ClassDef node.
        """
        self._resolver.name_to_module_map[base_class.__name__] = base_class.__module__
        delegatee_node = LibCSTNodeFactory.make_property_getter_node(
            DELEGATEE_ATTR, base_class.__name__, "..."
        )
        body_nodes: list[libcst.FunctionDef] = [delegatee_node]
        for nodes in body_items.values():
            body_nodes.extend(nodes)

        rolefor_bases = self._resolve_rolefor_bases_for(base_class)
        bases = rolefor_bases + [ABC.__name__]
        return LibCSTNodeFactory.make_dataclass(
            self.get_delegator_for_name(base_class), bases=bases, body=body_nodes
        )

    def _resolve_rolefor_bases_for(self, base_class: type) -> list[str]:
        """Return RoleFor class names for the nearest ancestors of base_class that already have RoleFor nodes.

        Checks locally generated nodes, cross-module imported nodes, and the global ownership
        registry (which tracks direct role-taker RoleFor nodes from any module transformer).

        :param base_class: The class whose parent hierarchy is searched.
        :return: List of RoleFor class name strings.
        """
        rolefor_bases: list[str] = []
        seen: set[type] = set()
        for parent in base_class.__bases__:
            for ancestor in parent.__mro__:
                if ancestor is object:
                    break
                in_local = ancestor in self._base_class_role_for_nodes
                in_cross = ancestor in self._cross_module_rolefor_bases
                owner_module = self._global_base_class_ownership.get(ancestor)
                if (
                    in_local or in_cross or owner_module is not None
                ) and ancestor not in seen:
                    if owner_module is not None and not in_local and not in_cross:
                        role_for_name = self.get_delegator_for_name(ancestor)
                        if owner_module is not self.source_module:
                            mixin_module = _mixin_module_dotted_name(
                                owner_module.__name__
                            )
                            self._cross_module_rolefor_bases[ancestor] = mixin_module
                            self.require_import(mixin_module, role_for_name)
                            self._resolver.name_to_module_map[role_for_name] = (
                                mixin_module
                            )
                    rolefor_bases.append(self.get_delegator_for_name(ancestor))
                    seen.add(ancestor)
                    break
        return rolefor_bases

    def make_role_for_bases(
        self,
        node: libcst.ClassDef,
        wrapped_class: WrappedClass,
        segregated_base_types: list[type] | None = None,
    ) -> list[libcst.Arg]:
        """Generate the base class arguments for a RoleFor class.

        :param node: The original taker class node.
        :param wrapped_class: The taker wrapped class.
        :param segregated_base_types: Same-module base classes that received their own RoleFor mixin.
        :return: List of Arg nodes representing the base classes.
        """
        role_for_bases = []
        bases_that_are_takers = self.bases_of_class_that_are_role_takers(wrapped_class)
        all_segregated = list(segregated_base_types or [])
        uncovered_segregated = [
            st
            for st in all_segregated
            if not any(
                other is not st and issubclass(other, st) for other in all_segregated
            )
        ]

        for base in node.bases:
            base_name = LibCSTNodeFactory.get_name_from_base_node(base.value)
            if base_name in bases_that_are_takers:
                taker_type = bases_that_are_takers[base_name]
                if any(
                    issubclass(seg, taker_type) and seg is not taker_type
                    for seg in uncovered_segregated
                ):
                    continue  # A more-derived segregated base already covers this taker
                role_for_name = self.get_delegator_for_name(taker_type)
                role_for_bases.append(LibCSTNodeFactory.make_argument(role_for_name))
                owner_module = self._global_base_class_ownership.get(taker_type)
                if owner_module is not None and owner_module is not self.source_module:
                    mixin_module = _mixin_module_dotted_name(owner_module.__name__)
                    if taker_type not in self._cross_module_rolefor_bases:
                        self._cross_module_rolefor_bases[taker_type] = mixin_module
                        self.require_import(mixin_module, role_for_name)
                        self._resolver.name_to_module_map[role_for_name] = mixin_module
            elif self._get_owned_delegatee_for_name(base_name) is not None:
                delegatee_type = self._get_owned_delegatee_for_name(base_name)
                role_for_name = self.get_delegator_for_name(delegatee_type)
                role_for_bases.append(LibCSTNodeFactory.make_argument(role_for_name))
                owner_module = self._global_base_class_ownership.get(delegatee_type)
                if owner_module is not None and owner_module is not self.source_module:
                    mixin_module = _mixin_module_dotted_name(owner_module.__name__)
                    if delegatee_type not in self._cross_module_rolefor_bases:
                        self._cross_module_rolefor_bases[delegatee_type] = mixin_module
                        self.require_import(mixin_module, role_for_name)
                        self._resolver.name_to_module_map[role_for_name] = mixin_module
            elif _is_role_base_node(base.value) and issubclass(
                wrapped_class.clazz, Role
            ):
                taker_type = wrapped_class.clazz.get_role_taker_type()
                role_for_bases.append(
                    LibCSTNodeFactory.make_argument(
                        self.get_delegator_for_name(taker_type)
                    )
                )

        for base_type in uncovered_segregated:
            role_for_bases.append(
                LibCSTNodeFactory.make_argument(self.get_delegator_for_name(base_type))
            )

        role_for_bases.append(LibCSTNodeFactory.make_argument(ABC.__name__))
        return role_for_bases

    def _transform_role(
        self, node: libcst.ClassDef, wrapped_class: WrappedClass
    ) -> libcst.ClassDef:
        """Adjust a delegator class by adding the corresponding DelegatorFor base.

        :param node: The original class node.
        :param wrapped_class: The wrapped class information.
        :return: The updated class node with the DelegatorFor base added.
        """
        role_type = RoleType.get_role_type(wrapped_class)
        logger.debug(
            "Transforming role %s, type: %s", wrapped_class.clazz.__name__, role_type
        )
        if role_type == RoleType.NOT_A_ROLE:
            return node
        taker_type = wrapped_class.clazz.get_delegatee_type()

        new_bases = []
        for base in node.bases:
            new_bases.append(base)
            base_name = LibCSTNodeFactory.get_name_from_base_node(base.value)
            logger.debug("  Checking base %s", base_name)

            if self._is_delegator_base_node(base):
                role_for_name = self.get_delegator_for_name(taker_type)
                if not any(
                    LibCSTNodeFactory.get_name_from_base_node(b.value) == role_for_name
                    for b in node.bases
                ):
                    logger.debug("  Adding %s", role_for_name)
                    new_bases.append(LibCSTNodeFactory.make_argument(role_for_name))

                mixin_module_name = _mixin_module_dotted_name(
                    sys.modules[taker_type.__module__].__name__
                )
                self.require_original_import(mixin_module_name, [role_for_name])

        return node.with_changes(bases=new_bases)

    def _is_delegator_base_node(self, base: libcst.Arg) -> bool:
        """Return True if the base argument represents a PropertyDelegator class in the diagram.

        :param base: A base class argument from a ClassDef node.
        :return: True if the base is Role, PropertyDelegator, or a PropertyDelegator subclass.
        """
        base_name = LibCSTNodeFactory.get_name_from_base_node(base.value)
        if base_name in (Role.__name__, PropertyDelegator.__name__):
            return True
        return any(
            wrapped.clazz.__name__ == base_name
            and issubclass(wrapped.clazz, PropertyDelegator)
            for wrapped in self.class_diagram.wrapped_classes
        )

    @classmethod
    def get_delegator_for_name(cls, delegatee_class: type) -> str:
        """Return the name of the DelegatorFor class for the given delegatee class."""
        return f"DelegatorFor{delegatee_class.__name__}"

    def _get_owned_delegatee_for_name(self, name: str) -> type | None:
        """Return the delegatee type whose ``__name__`` matches *name*, looking first
        in ``_global_base_class_ownership`` and then in ``_all_delegatees``."""
        for delegatee_type in self._global_base_class_ownership:
            if delegatee_type.__name__ == name:
                return delegatee_type
        for delegatee_type in self._all_delegatees:
            if delegatee_type.__name__ == name:
                return delegatee_type
        return None

    def require_import(self, module: str, names: str | list[str]):
        """Record an import that must appear in the generated mixin module.

        :param module: The module to import from.
        :param names: The name or list of names to import.
        """
        self._import_orchestrator.require_import(module, names)

    def _get_keyword_arguments_of_decorator_of_class_node(
        self,
        class_node: libcst.ClassDef,
        decorator_name: str,
    ) -> Dict[str, libcst.BaseExpression]:
        """Return the keyword arguments of the decorator of a class node.

        :param class_node: The class node to get the decorator of.
        :param decorator_name: The name of the decorator to get the arguments of.
        :return: A dictionary of keyword arguments and their values.
        """
        decorator_node = next(
            (
                d
                for d in class_node.decorators
                if self._get_decorator_name(d.decorator) == decorator_name
            ),
            None,
        )

        if decorator_node is None:
            return {}

        if isinstance(decorator_node.decorator, libcst.Call):
            return {
                arg.keyword.value: arg.value
                for arg in decorator_node.decorator.args
                if arg.keyword is not None
            }

        return {}

    def _get_decorator_name(self, decorator_node: libcst.BaseExpression) -> str | None:
        """Return the simple name from a decorator expression node, or None."""
        if isinstance(decorator_node, libcst.Name):
            return decorator_node.value
        if isinstance(decorator_node, libcst.Call) and isinstance(
            decorator_node.func, libcst.Name
        ):
            return decorator_node.func.value
        return None

    def __hash__(self):
        return hash((self.__class__, self.source_module))
