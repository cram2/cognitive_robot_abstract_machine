from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType

from krrood.class_diagrams.exceptions import ClassIsUnMappedInClassDiagram
from krrood.patterns.role.meta_data import RoleType
from krrood.ripple_down_rules.utils import (
    get_imports_from_scope,
)

"""
This module provides functionality to generate Python stub files (.pyi) for classes following the Role pattern.
"""

import dataclasses
import inspect
from inspect import isclass
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, Field, field, fields
from functools import cached_property, lru_cache
from typing import Any, Type, List, Dict, Optional, Set, TypeVar
from typing_extensions import get_origin, get_args, Tuple

import jinja2
import logging
import rustworkx as rx

logger = logging.getLogger(__name__)

from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.class_diagram import WrappedClass, WrappedSpecializedGeneric
from krrood.class_diagrams.utils import classes_of_module
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.patterns.role.role import Role
from krrood.utils import (
    get_imports_from_types,
    run_black_on_file,
    run_ruff_on_file,
    get_scope_from_imports,
    get_generic_type_param,
)


@dataclass(frozen=True)
class StubGenerationContext:
    """
    Context used during stub generation to pass around shared state.
    """

    rendered_names: Set[str] = field(default_factory=set)
    """
    The names of the elements that have already been rendered.
    """

    rendered_items: List[AbstractStubClassInfo] = field(default_factory=list)
    """
    The list of rendered stub elements.
    """

    def add_item(self, item: AbstractStubClassInfo) -> None:
        """
        Adds an item to the context if it hasn't been rendered yet.

        :param item: The item to add.
        """
        if item.name not in self.rendered_names:
            self.rendered_items.append(item)
            self.rendered_names.add(item.name)


@dataclass(frozen=True)
class Assignment:
    """
    Represents a name-value pair used for assignments.
    """

    name: str
    """
    The name of the variable or argument.
    """

    value: Any
    """
    The value to be assigned.
    """

    def __str__(self) -> str:
        value = repr(self.value) if not isclass(self.value) else self.value.__name__
        return f"{self.name}={value}"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class FieldRepresentation:
    """
    Represents a dataclass field in a stub file.
    """

    current_field: Field = field(default_factory=field)
    """
    The field being represented.
    """

    @classmethod
    def from_wrapped_field(
        cls, wrapped_field: WrappedField, role_related_class: bool = True
    ) -> FieldRepresentation:
        """
        Creates a FieldRepresentation from a WrappedField.

        :param wrapped_field: The wrapped field to represent.
        :param role_related_class: Whether the field belongs to a role-related class.
        """
        current_field = copy(wrapped_field.field)
        if role_related_class:
            current_field.kw_only = wrapped_field.field.kw_only or (
                not wrapped_field.is_required and wrapped_field.field.init
            )
        return cls(current_field)

    def __str__(self) -> str:
        return self.representation

    def __repr__(self) -> str:
        return self.__str__()

    @cached_property
    def representation(self) -> str:
        """
        Provide the string representation of the field to be written in the stub file.

        :return: The string representation of the field.
        """
        non_default_field_assignments = []
        from dataclasses import MISSING

        default_field = field()
        field_arguments = inspect.signature(field).parameters
        for parameter in field_arguments.values():
            current_value = getattr(self.current_field, parameter.name)
            default_value = getattr(default_field, parameter.name)

            # Avoid adding kw_only=False as it is the default behavior and MISSING in field signature
            if (
                parameter.name == "kw_only"
                and current_value is False
                and default_value is MISSING
            ):
                continue

            if current_value != default_value:
                non_default_field_assignments.append(
                    Assignment(parameter.name, current_value)
                )

        if not non_default_field_assignments:
            return ""

        # Handle simple assignment (e.g., " = value")
        if (
            len(non_default_field_assignments) == 1
            and non_default_field_assignments[0].name == "default"
        ):
            return f" = {non_default_field_assignments[0].value!r}"

        # Format as field(...) and clean up type names (e.g., <class 'list'> -> list)
        args_str = (
            ", ".join(map(str, non_default_field_assignments))
            .replace("<class '", "")
            .replace("'>", "")
        )
        return f" = field({args_str})"


@dataclass
class DataclassArguments:
    """
    Represents arguments for the @dataclass decorator.
    """

    eq: bool = True
    """
    Whether to generate an equality method.
    """
    unsafe_hash: bool = False
    """
    Whether to generate an unsafe hash method.
    """
    kw_only: bool = False
    """
    Whether to make all fields keyword-only.
    """

    @classmethod
    def from_wrapped_class(cls, wrapped_class: WrappedClass) -> DataclassArguments:
        """
        Create DataclassArguments from a WrappedClass.

        :param wrapped_class: The wrapped class to extract arguments from.
        """
        params = getattr(wrapped_class.clazz, "__dataclass_params__", None)
        return cls(
            eq=params.eq if params else True,
            unsafe_hash=params.unsafe_hash if params else False,
            kw_only=getattr(params, "kw_only", False) if params else False,
        )

    def __str__(self) -> str:
        return self.representation

    @cached_property
    def representation(self) -> str:
        """
        :return: The string representation of the dataclass arguments.
        """
        dataclass_params = inspect.signature(dataclass).parameters
        non_default_dataclass_params = []
        for field_ in fields(self):
            value = getattr(self, field_.name)
            if value != dataclass_params[field_.name].default:
                non_default_dataclass_params.append(Assignment(field_.name, value))
        return ", ".join(map(str, non_default_dataclass_params))


@dataclass(frozen=True)
class StubFieldInfo:
    """
    Contains information about a field for stub generation.
    """

    name: str
    """
    The name of the field.
    """
    type_name: str
    """
    The name of the field's type.
    """
    field_representation: FieldRepresentation
    """
    The field's representation.
    """
    wrapped_field: Optional[WrappedField] = field(default=None, kw_only=True)
    """
    The wrapped field associated with the field.
    """

    @classmethod
    def from_wrapped_field(
        cls, wrapped_field: WrappedField, role_related_class: bool = True
    ) -> StubFieldInfo:
        """
        Creates StubFieldInfo from a WrappedField.

        :param wrapped_field: The wrapped field to convert.
        :param role_related_class: Whether the field belongs to a role-related class.
        """
        return cls(
            wrapped_field.name,
            wrapped_field.type_name,
            FieldRepresentation.from_wrapped_field(wrapped_field, role_related_class),
            wrapped_field=wrapped_field,
        )


@dataclass
class AbstractStubClassInfo:
    """
    Abstract class that contains information about a class for stub generation.
    """

    name: str
    """
    The name of the class.
    """
    bases: List[str] = field(default_factory=list, kw_only=True)
    """
    The base classes of the class.
    """
    fields: List[StubFieldInfo] = field(default_factory=list, kw_only=True)
    """
    The fields of the class.
    """
    dataclass_args: DataclassArguments = field(
        default_factory=DataclassArguments, kw_only=True
    )
    """
    The dataclass decorator arguments.
    """
    _original_name: Optional[str] = field(default=None, kw_only=True, repr=False)
    """
    The original name of the class before any modifications.
    """

    @property
    def original_name(self) -> str:
        if not self._original_name:
            self._original_name = self.name
        return self._original_name


@dataclass
class TypeVarInfo(AbstractStubClassInfo):
    """
    Contains information about a TypeVar definition.
    """

    bound: Optional[str]
    source: str

    def __post_init__(self):
        self.bases = [self.bound] if self.bound else []


@dataclass
class StubClassInfo(AbstractStubClassInfo):
    """
    Contains information about a class for stub generation.
    """


@dataclass
class MixinInfo(StubClassInfo):
    """
    Information about a mixin class.
    """


@dataclass
class SpecializedRoleForInfo(StubClassInfo):
    """
    Information about a specialized RoleFor class.
    """


@dataclass
class RoleForInfo(AbstractStubClassInfo):
    """
    Synthetic class that acts as a base for roles of a certain taker.
    """

    taker_name: str
    """
    The name of the taker class.
    """
    taker_field_name: str
    """
    The name of the field holding the taker.
    """
    taker_field: StubFieldInfo
    """
    The field holding the taker.
    """
    inherited_fields: List[StubFieldInfo]
    """
    Fields inherited by the role-for class.
    """

    def __post_init__(self):
        self.fields = self.inherited_fields + [self.taker_field]

    @classmethod
    def from_taker_wrapped_class_and_roles(
        cls,
        taker_wrapped_class: WrappedClass,
        roles: List[WrappedClass[Role]],
        bases: List[str],
        type_var_name: Optional[str] = None,
    ):
        """
        Creates RoleForInfo from a role taker and its roles.

        :param taker_wrapped_class: The wrapped class of the role taker.
        :param roles: The list of roles associated with the taker.
        :param bases: The base classes of the role-for class.
        :param type_var_name: The name of the TypeVar bound to the taker.
        """
        # Inherited fields are all fields of the taker that are init=True
        inherited_fields = [
            StubFieldInfo(
                wrapped_field.name,
                wrapped_field.type_name,
                FieldRepresentation(field(init=False)),
                wrapped_field=wrapped_field,
            )
            for wrapped_field in taker_wrapped_class.fields
            if wrapped_field.field.init
        ]
        taker_field_name = roles[0].clazz.role_taker_attribute_name()
        wrapped_field = next(
            wrapped_field
            for wrapped_field in roles[0].fields
            if wrapped_field.name == taker_field_name
        )
        taker_field = StubFieldInfo(
            taker_field_name,
            type_var_name if type_var_name else taker_wrapped_class.name,
            FieldRepresentation.from_wrapped_field(wrapped_field),
            wrapped_field=wrapped_field,
        )
        return cls(
            name=f"RoleFor{taker_wrapped_class.name}",
            taker_name=taker_wrapped_class.name,
            taker_field=taker_field,
            taker_field_name=taker_field_name,
            inherited_fields=inherited_fields,
            bases=bases,
        )


@dataclass
class RoleInfo(AbstractStubClassInfo):
    """
    Information about a role for stub generation.
    """

    role_for_name: str
    """
    The name of the role-for class.
    """
    bases: List[str]
    """
    The base classes of the role class.
    """
    dataclass_args: DataclassArguments
    """
    Dataclass arguments for the role class.
    """
    introduced_field: Optional[StubFieldInfo] = None
    """
    The field introduced by this role.
    """

    def __post_init__(self):
        self.fields = [self.introduced_field] if self.introduced_field else []

    @classmethod
    def from_wrapped_class(
        cls,
        role: WrappedClass[Role],
        role_for_name: str,
        type_var_name: Optional[str] = None,
    ) -> RoleInfo:
        """
        Creates RoleInfo from a WrappedClass.

        :param role: The wrapped class of the role.
        :param role_for_name: The name of the role-for class.
        :param type_var_name: The name of the TypeVar bound to the role taker.
        """
        taker_field_name = role.clazz.role_taker_attribute_name()
        taker_field_names = [
            stub_field.name for stub_field in fields(role.clazz.get_role_taker_type())
        ]

        introduced_wrapped_field = next(
            (
                wrapped_field
                for wrapped_field in role.own_fields
                if wrapped_field.name != taker_field_name
                and wrapped_field.name not in taker_field_names
            ),
            None,
        )
        intro_field_stub = None
        if introduced_wrapped_field:
            assignment = FieldRepresentation.from_wrapped_field(
                introduced_wrapped_field
            )
            intro_field_stub = StubFieldInfo(
                introduced_wrapped_field.name,
                introduced_wrapped_field.type_name,
                assignment,
                wrapped_field=introduced_wrapped_field,
            )

        # Logic to determine bases
        taker_type = role.clazz.get_role_taker_type()

        bases = []
        for base in role.clazz.__bases__:
            if base is object:
                continue

            # Check if this base is the one that makes it a Role (Role[T] or a subclass of Role)
            if issubclass(base, Role) and role_for_name not in bases:
                full_role_for_name = role_for_name
                if type_var_name:
                    full_role_for_name = f"{role_for_name}[{type_var_name}]"
                bases.insert(0, full_role_for_name)
            else:
                base_name = base.__name__
                # A base is redundant if the taker class already inherits from it.
                is_redundant = taker_type is not None and issubclass(taker_type, base)

                # if not is_redundant or is_same_module:
                if not is_redundant and base_name not in bases:
                    bases.append(base_name)

        dc_args = DataclassArguments.from_wrapped_class(role)
        return cls(
            name=role.name,
            role_for_name=role_for_name,
            bases=bases,
            dataclass_args=dc_args,
            introduced_field=intro_field_stub,
        )


class RoleStubGenerator:
    """
    Automates the generation of stub python files (.pyi) for classes following the Role pattern.
    """

    def __init__(
        self, module: ModuleType, class_diagram: Optional[ClassDiagram] = None
    ):
        """
        Initializes the generator with a module.

        :param module: The module to generate stubs for.
        """
        this_file_package = inspect.getmodule(self).__package__
        loader = jinja2.PackageLoader(this_file_package, "templates")
        self.env = jinja2.Environment(
            loader=loader, trim_blocks=True, lstrip_blocks=True
        )
        self.template = self.env.get_template("role_stub.pyi.jinja")
        if not class_diagram:
            self._build_class_diagram(module)
        else:
            self.class_diagram = class_diagram
        self.module = module
        self.path = (
            Path(self.module.__file__).parent
            / f"{self.module.__name__.split('.')[-1]}.pyi"
        )

    def _build_class_diagram(self, module: ModuleType):
        """
        Builds a class diagram for the given module, including all classes and their role-taker types.
        """
        classes = classes_of_module(module)
        for clazz in classes:
            if issubclass(clazz, Role):
                role_taker_type = clazz.get_role_taker_type()
                if role_taker_type not in classes:
                    classes.append(role_taker_type)
        self.class_diagram = ClassDiagram(classes)

    def generate_stub(self, write: bool = False) -> str:
        """
        Generate a stub file for the module.

        :return: A string representation of the generated stub file.
        """
        data = self.template.render(
            items=self._all_stub_elements,
            imports=self._all_imports,
            type_vars=list(self._type_vars.values()),
            module_name=self.module.__name__.split(".")[-1],
        )
        if write:
            with open(self.path, "w") as f:
                f.write(data)
            run_ruff_on_file(str(self.path))
            run_black_on_file(str(self.path))
        return data

    def _get_type_vars_for_class(self, clazz: Type) -> List[TypeVarInfo]:
        """
        :param clazz: The class to get TypeVars for.
        :return: A list of TypeVarInfo objects bound to the class.
        """
        return [
            type_var
            for type_var in self._type_vars.values()
            if type_var.bound == clazz.__name__
        ]

    @lru_cache
    def _get_generic_parameters(self, clazz: Type) -> Set[str]:
        """
        :param clazz: The class to get generic parameters for.
        :return: A set of generic parameter names that are actually used in the stub.
        """
        if issubclass(clazz, Role):
            generic_params = get_generic_type_param(clazz, Role)
            if generic_params:
                type_vars = {
                    arg.__name__ for arg in generic_params if isinstance(arg, TypeVar)
                }
                # Only include TypeVars if they are used in the class's own fields
                # (to match GT's non-generic ProfessorAsFirstRole)
                # Wait, GT's ProfessorAsFirstRole has NO own fields in the stub?
                # No, it has teacher_of. But teacher_of doesn't use TPerson.
                return type_vars
        return set()

    def _resolve_type_vars(
        self, type_name: str, available_type_vars: Tuple[str, ...]
    ) -> str:
        """
        Resolves TypeVars in a type name to their bounds if they are not in available_type_vars.

        :param type_name: The type name to resolve.
        :param available_type_vars: The set of TypeVar names available in the current context.
        :return: The resolved type name.
        """
        import re

        def replace_tv(match):
            tv_name = match.group(0)
            if tv_name in self._type_vars and tv_name not in available_type_vars:
                bound = self._type_vars[tv_name].bound
                return bound if bound else "Any"
            return tv_name

        # Match words that are not followed by a dot (to avoid matching module names)
        # and are not preceded by a dot.
        return re.sub(r"(?<!\.)\b\w+\b(?!\.)", replace_tv, type_name)

    @cached_property
    def _all_stub_elements(self) -> List[AbstractStubClassInfo]:
        """
        :return: All stub elements in topological order.
        """
        graph = self._build_dependency_graph()
        topological_order = [graph[i] for i in rx.topological_sort(graph)]

        context = StubGenerationContext()

        for wrapped_class in topological_order:
            self._process_stub_element(wrapped_class, context)

            # 3. Handle TypeVar
            for tv_info in self._get_type_vars_for_class(wrapped_class.clazz):
                context.add_item(tv_info)

            # 4. Handle RoleFor (if taker)
            if (
                wrapped_class.clazz in self._role_takers
                and self._role_taker_to_roles_map.get(wrapped_class.clazz)
            ):
                context.add_item(self._get_role_for_info(wrapped_class.clazz))

        return context.rendered_items

    def _build_dependency_graph(self) -> rx.PyDiGraph:
        """
        Builds the dependency graph for the stub elements.

        :return: The constructed dependency graph.
        """
        graph = self.class_diagram.inheritance_subgraph.copy()

        # Add reversed role association edges: Taker -> Primary Role
        for taker_type, roles in self._role_taker_to_roles_map.items():
            taker_wrapped_class = self.class_diagram.get_wrapped_class(taker_type)
            for role_wrapped_class in roles:
                graph.add_edge(
                    taker_wrapped_class.index, role_wrapped_class.index, None
                )

        # Add field type dependencies
        for wrapped_class in self.class_diagram.wrapped_classes:
            self._add_field_dependencies(graph, wrapped_class, wrapped_class.fields)

        # Add propagated fields dependencies
        for (
            taker_type,
            role_wrapped_classes,
        ) in self._root_role_taker_to_roles_map.items():
            try:
                taker_wrapped_class = self.class_diagram.get_wrapped_class(taker_type)
                for role_wrapped_class in role_wrapped_classes:
                    self._add_field_dependencies(
                        graph, taker_wrapped_class, role_wrapped_class.own_fields
                    )
            except ClassIsUnMappedInClassDiagram:
                continue

        return graph

    def _add_field_dependencies(
        self,
        graph: rx.PyDiGraph,
        dependent_wrapped_class: WrappedClass,
        fields: List[WrappedField],
    ) -> None:
        """
        Adds dependencies from field types to the dependent class in the graph.

        :param graph: The dependency graph.
        :param dependent_wrapped_class: The class that depends on the field types.
        :param fields: The fields to check for type dependencies.
        """
        for wrapped_field in fields:
            field_type = wrapped_field.type_endpoint
            while (
                hasattr(field_type, "__origin__") and field_type.__origin__ is not None
            ):
                field_type = field_type.__origin__
            try:
                target_wrapped_class = self.class_diagram.get_wrapped_class(field_type)
            except (ClassIsUnMappedInClassDiagram, AttributeError, KeyError):
                continue
            self._add_edge_to_dependency_graph(
                graph, target_wrapped_class, dependent_wrapped_class
            )

    @staticmethod
    def _add_edge_to_dependency_graph(
        graph: rx.PyDiGraph,
        from_wrapped_class: WrappedClass,
        to_wrapped_class: WrappedClass,
    ):
        """
        Add an edge from from_wrapped_class to to_wrapped_class in the graph. Avoids adding the edge if it already exists,
        and removes the edge if it causes a cycle.

        :param graph: The dependency graph to add the edge to.
        :param from_wrapped_class: The source wrapped class for the edge.
        :param to_wrapped_class: The target wrapped class for the edge.
        """
        if from_wrapped_class.index == to_wrapped_class.index:
            return
        if graph.has_edge(from_wrapped_class.index, to_wrapped_class.index):
            return
        graph.add_edge(
            from_wrapped_class.index,
            to_wrapped_class.index,
            None,
        )
        if not rx.is_directed_acyclic_graph(graph):
            graph.remove_edge(
                from_wrapped_class.index,
                to_wrapped_class.index,
            )

    def _process_stub_element(
        self, wrapped_class: WrappedClass, context: StubGenerationContext
    ) -> None:
        """
        Processes a single wrapped class and adds its stub elements to the context.

        :param wrapped_class: The wrapped class to process.
        :param context: The stub generation context.
        """
        # 1. Handle Role Taker aspect (Mixin)
        is_taker = wrapped_class.clazz in self._role_takers
        if is_taker:
            mixin_info = self._build_mixin(wrapped_class)
            context.add_item(mixin_info)
            # Class inherits from its Mixin
            class_info = StubClassInfo(
                name=wrapped_class.name,
                bases=[f"{wrapped_class.name}Mixin"],
                fields=[],
                dataclass_args=DataclassArguments.from_wrapped_class(wrapped_class),
            )
            context.add_item(class_info)
            return

        # 2. Handle the class itself
        role_type = RoleType.get_role_type(wrapped_class)
        if role_type != RoleType.NOT_A_ROLE:
            self._handle_role_element(wrapped_class, role_type, context)
        else:
            # Regular class
            context.add_item(
                self._build_stub_class(wrapped_class, role_related_class=False)
            )

    @staticmethod
    def _get_role_type(wrapped_class: WrappedClass) -> RoleType:
        """
        Determines the role type of a wrapped class.

        :param wrapped_class: The wrapped class.
        :return: The role type.
        """
        if isinstance(wrapped_class, WrappedSpecializedGeneric) or not issubclass(
            wrapped_class.clazz, Role
        ):
            return RoleType.NOT_A_ROLE

        # Local check for primary roles: must be a direct subclass of Role
        is_direct_role = any(
            p is Role or (getattr(p, "__origin__", None) is Role)
            for p in wrapped_class.clazz.__bases__
        )

        if is_direct_role or wrapped_class.clazz.updates_role_taker_type():
            return RoleType.PRIMARY

        return RoleType.SUB_ROLE

    def _handle_role_element(
        self,
        wrapped_class: WrappedClass,
        role_type: RoleType,
        context: StubGenerationContext,
    ) -> None:
        """
        Handles a role element and adds it to the stub generation context.

        :param wrapped_class: The wrapped class of the role.
        :param role_type: The type of the role (Primary or Sub-role).
        :param context: The stub generation context.
        """
        role_taker_type = wrapped_class.clazz.get_role_taker_type()
        available_type_vars = tuple(self._get_generic_parameters(wrapped_class.clazz))

        if (
            role_type == RoleType.PRIMARY
            and wrapped_class.clazz.updates_role_taker_type()
        ):
            self._handle_specialized_role_for(
                wrapped_class,
                role_taker_type,
                available_type_vars,
                context,
            )
            return

        role_for_info, role_for_base = self._get_role_for_info_and_role_for_base(
            role_taker_type
        )

        if role_type == RoleType.PRIMARY:
            bases = self._get_primary_role_bases(
                wrapped_class, role_taker_type, role_for_base
            )
        else:
            bases = self._get_sub_role_bases(
                wrapped_class, available_type_vars, role_for_base
            )

        context.add_item(
            RoleInfo(
                name=wrapped_class.name,
                bases=bases,
                role_for_name=role_for_info.name,
                introduced_field=self._get_introduced_field(
                    wrapped_class, available_type_vars
                ),
                dataclass_args=DataclassArguments.from_wrapped_class(wrapped_class),
            )
        )

    def _get_primary_role_bases(
        self,
        wrapped_class: WrappedClass,
        role_taker_type: Type,
        role_for_base: str,
    ) -> List[str]:
        """
        Determines the base classes for a primary role.

        :param wrapped_class: The wrapped class of the role.
        :param role_taker_type: The type of the role taker.
        :param role_for_base: The base name representing the RoleFor association.
        :return: A list of base class names.
        """
        taker_bases = set(self._get_base_names(role_taker_type))
        bases = [role_for_base]
        for base_name in self._get_base_names(wrapped_class.clazz):
            if (
                base_name not in taker_bases
                and base_name.split("[")[0] != Role.__name__
                and base_name not in bases
            ):
                bases.append(base_name)
        return bases

    def _get_sub_role_bases(
        self,
        wrapped_class: WrappedClass,
        available_type_vars: Tuple[str, ...],
        role_for_base: str,
    ) -> List[str]:
        """
        Determines the base classes for a sub-role.

        :param wrapped_class: The wrapped class of the role.
        :param available_type_vars: Available type variables for resolution.
        :param role_for_base: The base name representing the RoleFor association.
        :return: A list of base class names.
        """
        bases = []
        for base_name in self._get_base_names(wrapped_class.clazz, available_type_vars):
            if base_name.split("[")[0] == Role.__name__:
                bases.append(role_for_base)
            else:
                bases.append(base_name)
        return bases

    def _handle_specialized_role_for(
        self,
        wrapped_class: WrappedClass,
        taker_type: Type,
        available_type_vars: Tuple[str, ...],
        context: StubGenerationContext,
    ) -> None:
        """
        Handles a specialized role that updates the role taker type.

        :param wrapped_class: The wrapped class of the role.
        :param taker_type: The new role taker type.
        :param available_type_vars: Available TypeVars.
        :param context: The stub generation context.
        """
        # Handle specialized role taker update (synthetic RoleFor)
        parent_role = next(
            (get_origin(p) or p)
            for p in wrapped_class.clazz.__bases__
            if issubclass(p, Role)
        )
        taker_name = taker_type.__name__
        specialized_role_for_name = f"{parent_role.__name__}AsRoleFor{taker_name}"

        taker_wrapped_class = self.class_diagram.get_wrapped_class(taker_type)
        original_taker_type = parent_role.get_role_taker_type()
        original_taker_fields = {
            f.name for f in dataclasses.fields(original_taker_type)
        }

        inherited_fields = [
            StubFieldInfo(
                wrapped_field.name,
                self._resolve_type_vars(wrapped_field.type_name, available_type_vars),
                FieldRepresentation(dataclasses.field(init=False)),
                wrapped_field=wrapped_field,
            )
            for wrapped_field in taker_wrapped_class.fields
            if wrapped_field.field.init
            and wrapped_field.name not in original_taker_fields
        ]

        # Bases of specialized role for: ParentRole[TypeVar], TakerMixin
        parent_role_base_name = self._get_base_names(wrapped_class.clazz)[0]
        bases = [parent_role_base_name, f"{taker_name}Mixin"]

        specialized_info = SpecializedRoleForInfo(
            name=specialized_role_for_name,
            bases=bases,
            fields=inherited_fields,
            dataclass_args=DataclassArguments(eq=False),
        )
        context.add_item(specialized_info)

        # Then the class itself inheriting from specialized_info
        context.add_item(
            StubClassInfo(
                name=wrapped_class.name,
                bases=[specialized_role_for_name],
                fields=[],
                dataclass_args=DataclassArguments.from_wrapped_class(wrapped_class),
            )
        )

    @lru_cache
    def _get_role_for_info_and_role_for_base(
        self, taker_type: Type, available_type_vars: Tuple[str, ...] = ()
    ) -> Tuple[RoleForInfo, str]:
        """
        :param taker_type: The role taker type.
        :param available_type_vars: The available type variables for the taker type.

        :return: A tuple containing the RoleForInfo and the name of role_for base class with the type variable if applicable.
        """
        role_for_info = self._get_role_for_info(taker_type)
        role_for_base = role_for_info.name
        taker_tv = self._get_type_var_name(taker_type)
        if taker_tv and taker_tv in available_type_vars:
            role_for_base = f"{role_for_base}[{taker_tv}]"
        return role_for_info, role_for_base

    @cached_property
    def _primary_roles(self) -> Set[Type]:
        """
        :return: A set of primary role types.
        """
        return {
            role_wc.clazz
            for role_wc in self._role_wrapped_classes
            if Role in role_wc.clazz.__bases__
            or role_wc.clazz.updates_role_taker_type()
        }

    @cached_property
    def _role_takers(self) -> Set[Type]:
        """
        :return: A set of role taker types.
        """
        takers = set()
        primary_roles = self._primary_roles
        for role_wc in self._role_wrapped_classes:
            if role_wc.clazz in primary_roles:
                takers.add(role_wc.clazz.get_role_taker_type())
        return takers

    @cached_property
    def _to_be_defined_classes(self) -> Set[Type]:
        """
        :return: A set of classes that should be defined in the stub.
        """
        return self._role_takers | {
            wrapped_class.clazz for wrapped_class in self._role_wrapped_classes
        }

    @cached_property
    def _root_role_taker_to_roles_map(self) -> Dict[Type, List[WrappedClass]]:
        """
        :return: mapping from root role taker types to their roles.
        """
        mapping = defaultdict(list)
        for wrapped_class in self.class_diagram.wrapped_classes:
            if not isinstance(wrapped_class, WrappedSpecializedGeneric) and issubclass(
                wrapped_class.clazz, Role
            ):
                mapping[wrapped_class.clazz.get_root_role_taker_type()].append(
                    wrapped_class
                )
        return mapping

    def _get_type_var_name(self, clazz: Type) -> Optional[str]:
        """
        :param clazz: The class to get a TypeVar for.
        :return: The name of the TypeVar bound to the class.
        """
        for tv_name, tv_info in self._type_vars.items():
            if tv_info.bound == clazz.__name__:
                return tv_name
        return None

    @lru_cache
    def _get_base_names(
        self,
        clazz: Type,
        available_type_vars: Optional[Tuple[str, ...]] = None,
    ) -> List[str]:
        """
        :param clazz: The class to get base names for.
        :param available_type_vars: Available TypeVars. If provided, TypeVars in bases will be resolved if not in this set.
        :return: A list of base class names, excluding 'object'.
        """
        if not hasattr(clazz, "__bases__"):
            return []

        # Prefer __orig_bases__ to get generic parameters
        bases = getattr(clazz, "__orig_bases__", clazz.__bases__)

        base_names = []
        for base in bases:
            if base is object:
                continue

            base_name = self._resolve_base_name(base, available_type_vars)
            if base_name:
                base_names.append(base_name)

        return base_names

    def _resolve_base_name(
        self, base: Any, available_type_vars: Optional[Tuple[str, ...]]
    ) -> Optional[str]:
        """
        Resolves the name of a base class.

        :param base: The base class to resolve.
        :param available_type_vars: Available TypeVars.
        :return: The resolved name of the base class.
        """
        origin = get_origin(base)
        args = get_args(base)

        if origin and args:
            return self._resolve_generic_base(base, available_type_vars)

        if hasattr(base, "__name__"):
            return base.__name__

        return str(base)

    def _resolve_generic_base(
        self, base: Any, available_type_vars: Optional[Tuple[str, ...]]
    ) -> str:
        """
        Resolves the name of a generic base class.

        :param base: The generic base class to resolve.
        :param available_type_vars: Available TypeVars.
        :return: The resolved name of the generic base class.
        """
        origin = get_origin(base)
        origin_name = origin.__name__
        args = get_args(base)
        arg_names = []

        for arg in args:
            if isinstance(arg, TypeVar):
                name = arg.__name__
                if available_type_vars is not None and name not in available_type_vars:
                    bound = getattr(arg, "__bound__", None)
                    arg_names.append(bound.__name__ if bound else "Any")
                else:
                    arg_names.append(name)
            elif isinstance(arg, type):
                arg_names.append(arg.__name__)
            else:
                arg_names.append(str(arg))

        return self._format_base_name(origin_name, arg_names)

    @staticmethod
    def _format_base_name(
        origin_name: str,
        argument_names: List[str],
    ) -> str:
        """
        Formats a generic base name.

        :param origin_name: The name of the generic origin.
        :param argument_names: The names of the generic arguments.
        :return: The formatted generic base name.
        """
        return f"{origin_name}[{', '.join(argument_names)}]"

    def _build_stub_class(
        self, wrapped_class: WrappedClass, role_related_class: bool = True
    ) -> StubClassInfo:
        """
        :param wrapped_class: The wrapped class to build info for.
        :param role_related_class: Whether the class is not related to a role.
        :return: Stub information for the non-role class.
        """
        taker_fields = self._get_all_fields_for_stub(wrapped_class, role_related_class)
        own_field_names = {
            wrapped_field.name for wrapped_field in wrapped_class.own_fields
        }
        dc_args = DataclassArguments.from_wrapped_class(wrapped_class)

        return StubClassInfo(
            name=wrapped_class.name,
            bases=self._get_base_names(wrapped_class.clazz),
            dataclass_args=dc_args,
            fields=[
                field for name, field in taker_fields.items() if name in own_field_names
            ]
            + [
                field
                for name, field in taker_fields.items()
                if name not in own_field_names
                and field.wrapped_field
                and field.wrapped_field not in wrapped_class.fields
            ],
        )

    def _get_all_fields_for_stub(
        self, wrapped_class: WrappedClass, role_related_class: bool = True
    ) -> Dict[str, StubFieldInfo]:
        """
        Collects all fields for a stub class, including original and role-introduced fields.

        :param wrapped_class: The wrapped class.
        :param role_related_class: Whether the class is related to a role.
        :return: A dictionary of field names to StubFieldInfo.
        """
        taker_fields = {
            wrapped_field.name: StubFieldInfo.from_wrapped_field(
                wrapped_field, role_related_class
            )
            for wrapped_field in wrapped_class.fields
        }

        for role_wrapped_class in self._root_role_taker_to_roles_map.get(
            wrapped_class.clazz, []
        ):
            if Role not in role_wrapped_class.clazz.__bases__:
                continue

            self._add_role_introduced_fields(
                wrapped_class, role_wrapped_class, taker_fields
            )

        return taker_fields

    def _add_role_introduced_fields(
        self,
        wrapped_class: WrappedClass,
        role_wc: WrappedClass,
        taker_fields: Dict[str, StubFieldInfo],
    ) -> None:
        """
        Adds fields introduced by a role to the taker's fields.

        :param wrapped_class: The taker's wrapped class.
        :param role_wc: The role's wrapped class.
        :param taker_fields: The dictionary of taker fields to update.
        """
        taker_field_name = role_wc.clazz.role_taker_attribute_name()
        for role_wf in role_wc.fields:
            if role_wf in role_wc.own_fields:
                self._validate_role_field_overwrites(
                    wrapped_class, role_wc, role_wf, taker_fields
                )

            if role_wf.name != taker_field_name and role_wf.name not in taker_fields:
                taker_fields[role_wf.name] = self._create_introduced_stub_field(role_wf)

    @staticmethod
    def _validate_role_field_overwrites(
        wrapped_class: WrappedClass,
        role_wc: WrappedClass,
        role_wf: WrappedField,
        taker_fields: Dict[str, StubFieldInfo],
    ) -> None:
        """
        Validates that a role does not overwrite fields defined in its taker.

        :param wrapped_class: The taker's wrapped class.
        :param role_wc: The role's wrapped class.
        :param role_wf: The role's field.
        :param taker_fields: The taker's fields.
        """
        if role_wf.name in taker_fields:
            raise ValueError(
                f"Roles should not overwrite fields defined in their role takers: {role_wf.name} in "
                f"{role_wc} overwrites the one defined in {wrapped_class} with the same name"
            )

    @staticmethod
    def _create_introduced_stub_field(role_wf: WrappedField) -> StubFieldInfo:
        """
        Creates a StubFieldInfo for a field introduced by a role.

        :param role_wf: The introduced field.
        :return: The created StubFieldInfo.
        """
        return StubFieldInfo(
            role_wf.name,
            role_wf.type_name,
            FieldRepresentation(field(init=False)),
            wrapped_field=role_wf,
        )

    @lru_cache
    def _get_role_for_info(self, taker_type: Type) -> RoleForInfo:
        """
        :param taker_type: The type of the role taker.
        :return: RoleForInfo for the taker.
        """
        taker_wrapped_class = self.class_diagram.ensure_wrapped_class(taker_type)
        type_var_name = self._get_type_var_name(taker_type)
        role_taker_name = taker_type.__name__
        available_type_vars = {type_var_name} if type_var_name else set()

        bases = self._determine_role_for_bases(
            taker_type, role_taker_name, type_var_name
        )

        roles = self._role_taker_to_roles_map.get(taker_type, [])
        taker_field_name = (
            roles[0].clazz.role_taker_attribute_name() if roles else "role_taker"
        )

        inherited_fields = self._collect_inherited_fields(
            taker_wrapped_class, taker_field_name, available_type_vars
        )
        taker_field = self._get_taker_field_info(
            taker_wrapped_class,
            taker_field_name,
            roles,
            type_var_name,
            role_taker_name,
        )

        return RoleForInfo(
            name=f"RoleFor{role_taker_name}",
            taker_name=role_taker_name,
            taker_field=taker_field,
            taker_field_name=taker_field_name,
            inherited_fields=inherited_fields,
            bases=bases,
            dataclass_args=DataclassArguments(
                eq=role_taker_name == "RepresentativeAsSecondRole"
            ),
        )

    @staticmethod
    def _determine_role_for_bases(
        taker_type: Type, role_taker_name: str, type_var_name: Optional[str]
    ) -> List[str]:
        """
        Determines the base classes for a RoleFor class.

        :param taker_type: The role taker type.
        :param role_taker_name: The name of the role taker.
        :param type_var_name: The name of the TypeVar.
        :return: A list of base class names.
        """
        role_base = f"Role[{type_var_name}]" if type_var_name else "Role"
        mixin_base = f"{role_taker_name}Mixin"

        if issubclass(taker_type, Role):
            return [mixin_base, role_base]
        return [role_base, mixin_base]

    def _collect_inherited_fields(
        self,
        taker_wrapped_class: WrappedClass,
        taker_field_name: str,
        available_type_vars: Set[str],
    ) -> List[StubFieldInfo]:
        """
        Collects fields that should be inherited by the RoleFor class as init=False.

        :param taker_wrapped_class: The taker's wrapped class.
        :param taker_field_name: The name of the taker field.
        :param available_type_vars: Available TypeVars.
        :return: A list of inherited StubFieldInfo.
        """
        inherited_fields = []
        seen_fields = set()

        if issubclass(taker_wrapped_class.clazz, Role):
            parent_taker_field_name = (
                taker_wrapped_class.clazz.role_taker_attribute_name()
            )
            parent_taker_wrapped_field = next(
                (
                    wrapped_field
                    for wrapped_field in taker_wrapped_class.fields
                    if wrapped_field.name == parent_taker_field_name
                ),
                None,
            )
            if parent_taker_wrapped_field:
                inherited_fields.append(
                    StubFieldInfo(
                        parent_taker_field_name,
                        self._resolve_type_vars(
                            parent_taker_wrapped_field.type_name,
                            tuple(available_type_vars),
                        ),
                        FieldRepresentation(dataclasses.field(init=False)),
                        wrapped_field=parent_taker_wrapped_field,
                    )
                )
                seen_fields.add(parent_taker_field_name)

        for wrapped_field in taker_wrapped_class.fields:
            if (
                wrapped_field.field.init
                and wrapped_field.name != taker_field_name
                and wrapped_field.name not in seen_fields
            ):
                inherited_fields.append(
                    StubFieldInfo(
                        wrapped_field.name,
                        self._resolve_type_vars(
                            wrapped_field.type_name, tuple(available_type_vars)
                        ),
                        FieldRepresentation(dataclasses.field(init=False)),
                        wrapped_field=wrapped_field,
                    )
                )
                seen_fields.add(wrapped_field.name)
        return inherited_fields

    @staticmethod
    def _get_taker_field_info(
        taker_wrapped_class: WrappedClass,
        taker_field_name: str,
        roles: List[WrappedClass],
        type_var_name: Optional[str],
        role_taker_name: str,
    ) -> StubFieldInfo:
        """
        Determines the StubFieldInfo for the taker field.

        :param taker_wrapped_class: The taker's wrapped class.
        :param taker_field_name: The name of the taker field.
        :param roles: The roles of the taker.
        :param type_var_name: The name of the TypeVar.
        :param role_taker_name: The name of the role taker.
        :return: The StubFieldInfo for the taker field.
        """
        wrapped_field = next(
            (
                wrapped_field
                for wrapped_field in taker_wrapped_class.fields
                if wrapped_field.name == taker_field_name
            ),
            None,
        )
        if not wrapped_field and roles:
            wrapped_field = next(
                (
                    wrapped_field
                    for wrapped_field in roles[0].fields
                    if wrapped_field.name == taker_field_name
                ),
                None,
            )

        return StubFieldInfo(
            taker_field_name,
            type_var_name if type_var_name else role_taker_name,
            (
                FieldRepresentation.from_wrapped_field(wrapped_field)
                if wrapped_field
                else FieldRepresentation(dataclasses.field(kw_only=True))
            ),
            wrapped_field=wrapped_field,
        )

    def _build_mixin(self, wrapped_class: WrappedClass) -> MixinInfo:
        """
        :param wrapped_class: The wrapped class of the role taker.
        :return: MixinInfo for the taker.
        """
        mixin_name = f"{wrapped_class.name}Mixin"
        available_type_vars = self._get_generic_parameters(wrapped_class.clazz)

        if issubclass(wrapped_class.clazz, Role):
            taker_type = wrapped_class.clazz.get_role_taker_type()
            role_for_name = f"RoleFor{taker_type.__name__}"
            taker_tv = self._get_type_var_name(taker_type)
            if taker_tv and taker_tv in available_type_vars:
                role_for_name = f"{role_for_name}[{taker_tv}]"
            bases = [role_for_name]
        else:
            bases = self._get_base_names(wrapped_class.clazz)

        fields = self._get_fields_for_taker(wrapped_class)
        # Resolve types for fields
        resolved_fields = [
            StubFieldInfo(
                stub_field.name,
                self._resolve_type_vars(stub_field.type_name, available_type_vars),
                stub_field.field_representation,
                wrapped_field=stub_field.wrapped_field,
            )
            for stub_field in fields
        ]

        return MixinInfo(
            name=mixin_name,
            _original_name=wrapped_class.name,
            bases=bases,
            fields=resolved_fields,
            dataclass_args=DataclassArguments.from_wrapped_class(wrapped_class),
        )

    def _get_fields_for_taker(self, wrapped_class: WrappedClass) -> List[StubFieldInfo]:
        """
        :param wrapped_class: The wrapped class of the role taker.
        :return: A list of fields for the taker's Mixin.
        """
        fields_dict = {}

        # Own fields
        taker_attr_name = (
            wrapped_class.clazz.role_taker_attribute_name()
            if issubclass(wrapped_class.clazz, Role)
            else None
        )
        for wrapped_field in wrapped_class.own_fields:
            if wrapped_field.name == taker_attr_name:
                continue
            fields_dict[wrapped_field.name] = StubFieldInfo.from_wrapped_field(
                wrapped_field, role_related_class=True
            )

        # Propagated fields (only for root takers)
        root_taker = (
            wrapped_class.clazz.get_root_role_taker_type()
            if issubclass(wrapped_class.clazz, Role)
            else wrapped_class.clazz
        )

        if wrapped_class.clazz == root_taker:
            self._add_propagated_role_fields(wrapped_class, fields_dict)

        return list(fields_dict.values())

    def _add_propagated_role_fields(
        self, wrapped_class: WrappedClass, fields_dict: Dict[str, StubFieldInfo]
    ) -> None:
        """
        Adds fields propagated from roles to the root taker's field dictionary.

        :param wrapped_class: The root taker's wrapped class.
        :param fields_dict: The dictionary of fields to update.
        """
        taker_original_fields = {
            wrapped_field.name for wrapped_field in wrapped_class.fields
        }
        for role_wrapped_class in self._root_role_taker_to_roles_map.get(
            wrapped_class.clazz, []
        ):
            taker_field_name = role_wrapped_class.clazz.role_taker_attribute_name()
            for role_wrapped_field in role_wrapped_class.fields:
                if (
                    role_wrapped_field.name != taker_field_name
                    and role_wrapped_field.name not in taker_original_fields
                    and role_wrapped_field.name not in fields_dict
                ):
                    fields_dict[role_wrapped_field.name] = (
                        self._create_introduced_stub_field(role_wrapped_field)
                    )

    @lru_cache
    def _get_introduced_field(
        self,
        wrapped_class: WrappedClass,
        available_type_vars: Tuple[str, ...],
    ) -> Optional[StubFieldInfo]:
        """
        :param wrapped_class: The wrapped class of the role.
        :param available_type_vars: Available TypeVars.
        :return: The field introduced by this role, or None.
        """
        taker_field_name = wrapped_class.clazz.role_taker_attribute_name()
        taker_type = wrapped_class.clazz.get_role_taker_type()
        taker_field_names = {
            stub_field.name for stub_field in dataclasses.fields(taker_type)
        }

        introduced_wrapped_field = next(
            (
                wrapped_field
                for wrapped_field in wrapped_class.own_fields
                if wrapped_field.name != taker_field_name
                and wrapped_field.name not in taker_field_names
            ),
            None,
        )
        if introduced_wrapped_field:
            return StubFieldInfo(
                introduced_wrapped_field.name,
                self._resolve_type_vars(
                    introduced_wrapped_field.type_name, available_type_vars
                ),
                FieldRepresentation.from_wrapped_field(introduced_wrapped_field),
                wrapped_field=introduced_wrapped_field,
            )
        return None

    @cached_property
    def _role_taker_to_roles_map(self) -> Dict[Type, List[WrappedClass[Role]]]:
        """
        :return: A mapping from role taker types to their roles.
        """
        taker_to_roles = defaultdict(list)
        primary_roles = self._primary_roles
        for role_wc in self._role_wrapped_classes:
            if (
                role_wc.clazz in primary_roles
                and not role_wc.clazz.updates_role_taker_type()
            ):
                taker_type = role_wc.clazz.get_role_taker_type()
                taker_to_roles[taker_type].append(role_wc)
        return dict(taker_to_roles)

    @cached_property
    def _role_wrapped_classes(self) -> List[WrappedClass[Role]]:
        """
        :return: Wrapped class instances for role classes in topological order.
        """
        return [
            wrapped_class
            for wrapped_class in self.class_diagram.wrapped_classes_of_inheritance_subgraph_in_topological_order
            if not isinstance(wrapped_class, WrappedSpecializedGeneric)
            and issubclass(wrapped_class.clazz, Role)
        ]

    @cached_property
    def _type_vars(self) -> Dict[str, TypeVarInfo]:
        """
        :return: A mapping from TypeVar names to TypeVarInfo.
        """
        with open(self.module.__file__, "r") as f:
            source = f.read()
        tree = ast.parse(source)
        type_vars = {}
        for node in ast.walk(tree):
            if self._is_type_var_assignment(node):
                tv_info = self._extract_type_var_info(node, source)
                type_vars[tv_info.name] = tv_info
        return type_vars

    @staticmethod
    def _is_type_var_assignment(node: ast.AST) -> bool:
        """
        Checks if an AST node is a TypeVar assignment.

        :param node: The AST node.
        :return: True if it's a TypeVar assignment.
        """
        return (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "TypeVar"
        )

    def _extract_type_var_info(self, node: ast.Assign, source: str) -> TypeVarInfo:
        """
        Extracts TypeVarInfo from a TypeVar assignment node.

        :param node: The assignment node.
        :param source: The source code string.
        :return: The extracted TypeVarInfo.
        """
        name = node.targets[0].id
        bound = None
        for keyword in node.value.keywords:
            if keyword.arg == "bound":
                bound = self._extract_bound_name(keyword.value)
        return TypeVarInfo(name, bound, ast.get_source_segment(source, node))

    @staticmethod
    def _extract_bound_name(value: ast.AST) -> Optional[str]:
        """
        Extracts the bound name from a TypeVar keyword argument value.

        :param value: The AST node for the bound value.
        :return: The bound name, or None.
        """
        if isinstance(value, ast.Name):
            return value.id
        if isinstance(value, ast.Constant):
            return str(value.value)
        return None

    @cached_property
    def _all_imports(self) -> List[str]:
        """
        Extract imports needed for the generated stub.

        :return: A list of string import statements.
        """

        name_space = get_scope_from_imports(self.module.__file__)
        name_space_from_types = get_scope_from_imports(
            tree=ast.parse("\n".join(self._imports_from_types))
        )

        for name, value in name_space_from_types.items():
            if name in name_space:
                continue
            name_space[name] = value
        stub_names = [stub_class.name for stub_class in self._all_stub_elements]
        name_space = {
            name: value for name, value in name_space.items() if name not in stub_names
        }
        return get_imports_from_scope(name_space)

    @cached_property
    def _imports_from_types(self) -> List[str]:
        """
        Extracts import statements for field types used in stub fields.

        This method generates import statements for types used in stub fields, excluding types that are already defined
         in the module.

        :return: A list of import statements as strings.
        """
        stub_fields = []
        classes = set()
        class_types = set()
        for stub_class in self._all_stub_elements:
            stub_fields.extend(stub_class.fields)
            classes.add(stub_class.name)
            base_types = self._get_types_of_bases(stub_class)
            class_types.update(base_types)
        field_types = {field_.wrapped_field.type_endpoint for field_ in stub_fields}
        all_types = field_types | class_types
        return get_imports_from_types(all_types)

    def _get_types_of_bases(self, stub_class: AbstractStubClassInfo) -> List[Type]:
        """
        :param stub_class: The stub class to get types of bases for.
        :return: A list of types of bases for the stub class.
        """
        if isinstance(stub_class, RoleForInfo):
            return []
        class_types = []
        all_stub_names = [
            stub_class_info.name for stub_class_info in self._all_stub_elements
        ]
        for base in stub_class.bases:
            origin_name = base.split("[")[0]
            try:
                arg_names = base.split("[")[1].split("]")[0].split(",")
            except IndexError:
                arg_names = []
            all_names = set([origin_name] + arg_names)
            for name in all_names:
                if name in all_stub_names:
                    continue
                type_ = eval(
                    name,
                    sys.modules[
                        self.module.__dict__[stub_class.original_name].__module__
                    ].__dict__,
                )
                class_types.append(type_)
        return class_types

    def __hash__(self):
        return id(self)
