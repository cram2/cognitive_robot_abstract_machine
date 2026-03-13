from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

from krrood.ripple_down_rules.utils import (
    get_scope_from_imports,
    get_imports_from_scope,
)

"""
This module provides functionality to generate Python stub files (.pyi) for classes following the Role pattern.
"""

import __future__
import dataclasses
import inspect
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, Field, field, fields
from functools import cached_property
from typing import Any, Type, List, Dict, Optional, Set, Union

import jinja2
import rustworkx as rx

from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.class_diagram import WrappedClass, WrappedSpecializedGeneric
from krrood.class_diagrams.utils import classes_of_module
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.patterns import Role
from krrood.utils import (
    extract_imports_from_module,
    get_imports_from_types,
    run_black_on_file,
    run_ruff_on_file,
)


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
        return f"{self.name}={self.value}"

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
            return f" = {non_default_field_assignments[0].value}"

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
    fields: List[StubFieldInfo] = field(default_factory=list, kw_only=True)
    """
    The fields of the class.
    """


@dataclass
class StubClassInfo(AbstractStubClassInfo):
    """
    Contains information about a class for stub generation.
    """

    bases: List[str]
    """
    The base classes of the class.
    """
    dataclass_args: DataclassArguments
    """
    The dataclass decorator arguments.
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
        cls, taker_wc: WrappedClass, roles: List[WrappedClass[Role]]
    ):
        """
        Creates RoleForInfo from a role taker and its roles.

        :param taker_wc: The wrapped class of the role taker.
        :param roles: The list of roles associated with the taker.
        """
        # Inherited fields are all fields of the taker that are init=True
        inherited_fields = [
            StubFieldInfo(
                wf.name,
                wf.type_name,
                FieldRepresentation(field(init=False)),
                wrapped_field=wf,
            )
            for wf in taker_wc.fields
            if wf.field.init
        ]
        taker_field_name = roles[0].clazz.role_taker_field().name
        wrapped_field = next(
            f for f in roles[0].own_fields if f.name == taker_field_name
        )
        taker_field = StubFieldInfo(
            taker_field_name,
            taker_wc.name,
            FieldRepresentation.from_wrapped_field(wrapped_field),
            wrapped_field=wrapped_field,
        )
        return cls(
            name=f"RoleFor{taker_wc.name}",
            taker_name=taker_wc.name,
            taker_field=taker_field,
            taker_field_name=taker_field_name,
            inherited_fields=inherited_fields,
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
        cls, role: WrappedClass[Role], role_for_name: str
    ) -> RoleInfo:
        """
        Creates RoleInfo from a WrappedClass.

        :param role: The wrapped class of the role.
        :param role_for_name: The name of the role-for class.
        """
        taker_field_name = role.clazz.role_taker_field().name
        taker_field_names = [f.name for f in fields(role.clazz.get_role_taker_type())]

        intro_field_wc = next(
            (
                wf
                for wf in role.own_fields
                if wf.name != taker_field_name and wf.name not in taker_field_names
            ),
            None,
        )
        intro_field_stub = None
        if intro_field_wc:
            assignment = FieldRepresentation.from_wrapped_field(intro_field_wc)
            intro_field_stub = StubFieldInfo(
                intro_field_wc.name,
                intro_field_wc.type_name,
                assignment,
                wrapped_field=intro_field_wc,
            )

        # Logic to determine bases
        taker_type = role.clazz.get_role_taker_type()

        bases = []
        for base in role.clazz.__bases__:
            if base is object:
                continue

            # Check if this base is the one that makes it a Role (Role[T] or a subclass of Role)
            if base is Role and role_for_name not in bases:
                bases.insert(0, role_for_name)
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
        self.class_diagram = (
            class_diagram if class_diagram else ClassDiagram(classes_of_module(module))
        )
        self.module = module
        self.path = (
            Path(self.module.__file__).parent
            / f"{self.module.__name__.split('.')[-1]}.pyi"
        )

    def generate_stub(self, write: bool = False) -> str:
        """
        Generate a stub file for the module.

        :return: A string representation of the generated stub file.
        """
        data = self.template.render(
            items=self._all_stub_elements,
            imports=self._all_imports,
            module_name=self.module.__name__.split(".")[-1],
        )
        if write:
            with open(self.path, "w") as f:
                f.write(data)
            run_ruff_on_file(str(self.path))
            run_black_on_file(str(self.path))
        return data

    @cached_property
    def _all_stub_elements(self) -> List[AbstractStubClassInfo]:
        """
        :return: All stub elements in topological order.
        """
        graph = self.class_diagram.inheritance_subgraph.copy()

        # Add reversed role association edges: Taker -> Primary Role
        # This ensures Taker is defined before its roles.
        for taker_type, roles in self._role_taker_to_roles_map.items():
            try:
                taker_wc = self.class_diagram.get_wrapped_class(taker_type)
            except Exception:
                # Taker might be in another module
                continue
            for role_wc in roles:
                graph.add_edge(taker_wc.index, role_wc.index, None)

        topological_order = [graph[i] for i in rx.topological_sort(graph)]

        rendered_items = []
        rendered_role_for = set()

        for wc in topological_order:
            if wc.clazz in self._primary_roles:
                taker_type = wc.clazz.get_role_taker_type()
                role_for_info = self._role_taker_to_role_for_map[taker_type]
                if taker_type not in rendered_role_for:
                    rendered_items.append(role_for_info)
                    rendered_role_for.add(taker_type)

                rendered_items.append(
                    RoleInfo.from_wrapped_class(wc, role_for_info.name)
                )

            elif wc.clazz in self._role_takers:
                # Render the taker class itself
                rendered_items.append(self._build_stub_class(wc))
                # Then its RoleFor
                if wc.clazz not in rendered_role_for:
                    role_for_info = self._role_taker_to_role_for_map[wc.clazz]
                    rendered_items.append(role_for_info)
                    rendered_role_for.add(wc.clazz)

            elif not isinstance(wc, WrappedSpecializedGeneric) and issubclass(
                wc.clazz, Role
            ):
                # Role subclass (not primary)
                rendered_items.append(
                    self._build_stub_class(wc, role_related_class=True)
                )

            else:
                # Regular class (not role related)
                is_related = wc.clazz in self._to_be_defined_classes
                rendered_items.append(
                    self._build_stub_class(wc, role_related_class=is_related)
                )

        # Handle roles whose takers are in other modules
        for role_wc in self._role_wrapped_classes:
            if role_wc.clazz in self._primary_roles:
                taker_type = role_wc.clazz.get_role_taker_type()
                if taker_type not in rendered_role_for:
                    role_for_info = self._role_taker_to_role_for_map[taker_type]
                    rendered_items.append(role_for_info)
                    rendered_role_for.add(taker_type)
                    rendered_items.append(
                        RoleInfo.from_wrapped_class(role_wc, role_for_info.name)
                    )

        # Remove duplicates while preserving order
        unique_items = []
        seen_names = set()
        for item in rendered_items:
            # We use name as identifier for synthetic classes and real classes
            if item.name not in seen_names:
                unique_items.append(item)
                seen_names.add(item.name)

        return unique_items

    @cached_property
    def _primary_roles(self) -> Set[Type]:
        """
        :return: A set of primary role types.
        """
        return {
            role_wc.clazz
            for role_wc in self._role_wrapped_classes
            if Role in role_wc.clazz.__bases__
        }

    @cached_property
    def _role_takers(self) -> Set[Type]:
        """
        :return: A set of role taker types.
        """
        return set(self._role_taker_to_roles_map.keys())

    @cached_property
    def _to_be_defined_classes(self) -> Set[Type]:
        """
        :return: A set of classes that should be defined in the stub.
        """
        return self._role_takers | {wc.clazz for wc in self._role_wrapped_classes}

    @cached_property
    def _root_role_taker_to_roles_map(self) -> Dict[Type, List[WrappedClass]]:
        """
        :return: mapping from root role taker types to their roles.
        """
        mapping = defaultdict(list)
        for wc in self.class_diagram.wrapped_classes:
            if not isinstance(wc, WrappedSpecializedGeneric) and issubclass(
                wc.clazz, Role
            ):
                mapping[wc.clazz.get_root_role_taker_type()].append(wc)
        return mapping

    def _get_base_names(self, clazz: Type) -> List[str]:
        """
        :param clazz: The class to get base names for.
        :return: A list of base class names, excluding 'object'.
        """
        if not hasattr(clazz, "__bases__"):
            return []
        return [base.__name__ for base in clazz.__bases__ if base is not object]

    def _build_stub_class(
        self, wrapped_class: WrappedClass, role_related_class: bool = True
    ) -> StubClassInfo:
        """
        :param wrapped_class: The wrapped class to build info for.
        :param role_related_class: Whether the class is not related to a role.
        :return: Stub information for the non-role class.
        """
        # Add original fields
        taker_fields = {
            wf.name: StubFieldInfo.from_wrapped_field(wf, role_related_class)
            for wf in wrapped_class.fields
        }
        taker_field_names = [wf.name for wf in wrapped_class.own_fields]

        # Add role-introduced fields as init=False
        introduced_fields = {}
        for role_wc in self._root_role_taker_to_roles_map.get(wrapped_class.clazz, []):
            if Role not in role_wc.clazz.__bases__:
                continue
            taker_field_name = role_wc.clazz.role_taker_field().name
            for role_wf in role_wc.fields:
                is_owned_field = role_wf in role_wc.own_fields
                if is_owned_field and role_wf.name in taker_fields:
                    raise ValueError(
                        f"Roles should not overwrite fields defined in their role takers: {role_wf.name} in "
                        f"{role_wc} overwrites the one defined in {wrapped_class} with the same name"
                    )
                if (
                    role_wf.name != taker_field_name
                    and role_wf.name not in taker_fields
                ):
                    stub_field = StubFieldInfo(
                        role_wf.name,
                        role_wf.type_name,
                        FieldRepresentation(field(init=False)),
                        wrapped_field=role_wf,
                    )
                    introduced_fields[role_wf.name] = stub_field
                    taker_fields[role_wf.name] = stub_field

        dc_args = DataclassArguments.from_wrapped_class(wrapped_class)

        return StubClassInfo(
            wrapped_class.name,
            self._get_base_names(wrapped_class.clazz),
            dc_args,
            fields=[
                stub_field
                for name, stub_field in taker_fields.items()
                if name in taker_field_names
            ]
            + list(introduced_fields.values()),
        )

    @cached_property
    def _role_taker_to_role_for_map(self) -> Dict[Type, RoleForInfo]:
        """
        :return: A mapping from role taker types to their RoleFor information.
        """
        return {
            taker_type: RoleForInfo.from_taker_wrapped_class_and_roles(
                self.class_diagram.ensure_wrapped_class(taker_type), roles
            )
            for taker_type, roles in self._role_taker_to_roles_map.items()
        }

    @cached_property
    def _role_taker_to_roles_map(self) -> Dict[Type, List[WrappedClass[Role]]]:
        """
        :return: A mapping from role taker types to their roles.
        """
        taker_to_roles = defaultdict(list)
        primary_roles = self._primary_roles
        for role_wc in self._role_wrapped_classes:
            if role_wc.clazz in primary_roles:
                taker_type = role_wc.clazz.get_role_taker_type()
                taker_to_roles[taker_type].append(role_wc)
        return taker_to_roles

    @cached_property
    def _role_wrapped_classes(self) -> List[WrappedClass[Role]]:
        """
        :return: Wrapped class instances for role classes in topological order.
        """
        return [
            wc
            for wc in self.class_diagram.wrapped_classes_of_inheritance_subgraph_in_topological_order
            if not isinstance(wc, WrappedSpecializedGeneric)
            and issubclass(wc.clazz, Role)
        ]

    @cached_property
    def _all_imports(self) -> List[str]:
        """
        Extract imports needed for the generated stub.

        :return: A list of string import statements.
        """

        name_space = get_scope_from_imports(self.module.__file__)
        name_space_from_types = get_scope_from_imports(
            tree=ast.parse("\n".join(self._imports_from_field_types))
        )

        for name, value in name_space_from_types.items():
            if name in name_space:
                continue
            name_space[name] = value

        return get_imports_from_scope(name_space)

    @cached_property
    def _imports_from_field_types(self) -> List[str]:
        """
        Extracts import statements for field types used in stub fields.

        This method generates import statements for types used in stub fields, excluding types that are already defined
         in the module.

        :return: A list of import statements as strings.
        """
        stub_fields = []
        classes = set()
        for stub_class in self._all_stub_elements:
            stub_fields.extend(stub_class.fields)
            classes.add(stub_class.name)

        field_types = {field_.wrapped_field.type_endpoint for field_ in stub_fields}
        # Remove types that are already defined in the module
        field_types = {
            field_type
            for field_type in field_types
            if field_type.__name__ not in classes
        }

        return get_imports_from_types(field_types)
