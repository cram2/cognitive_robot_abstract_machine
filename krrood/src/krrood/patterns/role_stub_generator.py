from __future__ import annotations

from pathlib import Path
from typing_extensions import Tuple

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
from typing import Any, Type, List, Dict, Optional, Set, TYPE_CHECKING

import jinja2

from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.utils import classes_of_module
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.patterns import Role
from krrood.utils import extract_imports


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
    def from_wrapped_field(cls, wf: WrappedField) -> FieldRepresentation:
        """
        Creates a FieldRepresentation from a WrappedField.

        :param wf: The wrapped field to convert.
        """
        current_field = copy(wf.field)
        current_field.kw_only = wf.field.kw_only or (
            not wf.is_required and wf.field.init
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

    @classmethod
    def from_wrapped_field(cls, wf: WrappedField) -> StubFieldInfo:
        """
        Creates StubFieldInfo from a WrappedField.

        :param wf: The wrapped field to convert.
        """
        return cls(wf.name, wf.type_name, FieldRepresentation.from_wrapped_field(wf))


@dataclass(frozen=True)
class StubClassInfo:
    """
    Contains information about a class for stub generation.
    """

    name: str
    """
    The name of the class.
    """
    bases: List[str]
    """
    The base classes of the class.
    """
    fields: List[StubFieldInfo]
    """
    The fields of the class.
    """
    dataclass_args: DataclassArguments
    """
    The dataclass decorator arguments.
    """


@dataclass(frozen=True)
class RoleTakerInfo:
    """
    Information about a role taker for stub generation.
    """

    role_for_name: str
    """
    The name of the role-for class.
    """
    taker_field_name: str
    """
    The name of the field holding the taker.
    """
    inherited_fields: List[StubFieldInfo]
    """
    Fields inherited by the role-for class.
    """
    roles: List[RoleInfo]
    """
    List of roles associated with the taker.
    """

    @classmethod
    def from_taker_wrapped_class_and_roles(
        cls, taker_wc: WrappedClass, roles: List[WrappedClass[Role]]
    ):
        """
        Creates RoleTakerInfo from a role taker and its roles.

        :param taker_wc: The wrapped class of the role taker.
        :param roles: The list of roles associated with the taker.
        """
        role_infos = [RoleInfo.from_wrapped_class(role) for role in roles]

        # Inherited fields are all fields of the taker that are init=True
        inherited_fields = [
            StubFieldInfo(wf.name, wf.type_name, FieldRepresentation(field(init=False)))
            for wf in taker_wc.fields
            if wf.field.init
        ]

        return cls(
            role_for_name=f"RoleFor{taker_wc.name}",
            taker_field_name=roles[0].clazz.role_taker_field().name,
            inherited_fields=inherited_fields,
            roles=role_infos,
        )


@dataclass(frozen=True)
class RoleInfo:
    """
    Information about a role for stub generation.
    """

    name: str
    """
    The name of the role class.
    """
    dataclass_args: DataclassArguments
    """
    Dataclass arguments for the role class.
    """
    introduced_field: Optional[StubFieldInfo] = None
    """
    The field introduced by this role.
    """

    @classmethod
    def from_wrapped_class(cls, role: WrappedClass[Role]) -> RoleInfo:
        """
        Creates RoleInfo from a WrappedClass.

        :param role: The wrapped class of the role.
        """
        taker_field_name = role.clazz.role_taker_field().name

        intro_field_wc = next(
            (wf for wf in role.fields if wf.name != taker_field_name), None
        )
        intro_field_stub = None
        if intro_field_wc:
            assignment = FieldRepresentation.from_wrapped_field(intro_field_wc)
            intro_field_stub = StubFieldInfo(
                intro_field_wc.name, intro_field_wc.type_name, assignment
            )
        dc_args = DataclassArguments.from_wrapped_class(role)
        return cls(
            name=role.name, dataclass_args=dc_args, introduced_field=intro_field_stub
        )


class RoleStubGenerator:
    """
    Automates the generation of stub python files (.pyi) for classes following the Role pattern.
    """

    def __init__(self, module: Any):
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
        self.class_diagram = ClassDiagram(classes_of_module(module))
        self.module = module
        self.path = Path(self.module.__file__).parent / f"{self.module.__name__.split('.')[-1]}.pyi"

    def generate_stub(self, write: bool = False) -> str:
        """
        Generate a stub file for the module.

        :return: A string representation of the generated stub file.
        """
        # Sort role takers to ensure they are defined after their dependencies
        role_takers = self._role_taker_to_info_map
        sorted_taker_types = [
            wc.clazz
            for wc in reversed(
                self.class_diagram.wrapped_classes_of_role_associations_subgraph_in_topological_order
            )
            if wc.clazz in role_takers
        ]

        # In case some taker types are not in the topological sort
        for taker_type in role_takers:
            if taker_type not in sorted_taker_types:
                sorted_taker_types.append(taker_type)

        sorted_role_takers = {
            taker_type: role_takers[taker_type] for taker_type in sorted_taker_types
        }

        data = self.template.render(
            stub_classes=self._non_role_stub_classes,
            role_takers=sorted_role_takers,
            imports=self._extract_imports(),
            normal_module_imports=self._normal_module_imports,
            type_checking_module_imports=self._type_checking_module_imports,
            module_name=self.module.__name__.split(".")[-1],
        )
        if write:
            with open(self.path, "w") as f:
                f.write(data)
        return data

    @cached_property
    def _non_role_stub_classes(self) -> List[StubClassInfo]:
        """
        :return: Stub information for non-role classes in topological order.
        """
        return [
            self._build_stub_class(wc)
            for wc in self.class_diagram.wrapped_classes_of_inheritance_subgraph_in_topological_order
            if not issubclass(wc.clazz, Role) and wc.clazz in self._role_takers
        ]

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
    def _to_be_imported_classes(self) -> Set[Type]:
        """
        :return: A set of classes from the module that should be imported.
        """
        module_classes = set(classes_of_module(self.module))
        return module_classes - self._to_be_defined_classes

    @cached_property
    def _normal_module_imports(self) -> List[str]:
        """
        :return: A sorted list of class names from the module that should be imported normally.
        """
        normal = set()
        for clazz in self._to_be_imported_classes:
            is_base = any(
                issubclass(defined_class, clazz)
                for defined_class in self._to_be_defined_classes
                if defined_class != clazz
            )
            if is_base:
                normal.add(clazz.__name__)
        return sorted(list(normal))

    @cached_property
    def _type_checking_module_imports(self) -> List[str]:
        """
        :return: A sorted list of class names from the module that should be imported under TYPE_CHECKING.
        """
        all_imported = {clazz.__name__ for clazz in self._to_be_imported_classes}
        normal = set(self._normal_module_imports)
        return sorted(list(all_imported - normal))

    @cached_property
    def _root_role_taker_to_roles_map(self) -> Dict[Type, List[WrappedClass]]:
        """
        :return: mapping from root role taker types to their roles.
        """
        mapping = defaultdict(list)
        for wc in self.class_diagram.wrapped_classes:
            if issubclass(wc.clazz, Role):
                mapping[wc.clazz.get_root_role_taker_type()].append(wc)
        return mapping

    def _build_stub_class(self, wrapped_class: WrappedClass) -> StubClassInfo:
        """
        :param wrapped_class: The wrapped class to build info for.
        :return: Stub information for the non-role class.
        """
        # Add original fields
        taker_fields = [
            StubFieldInfo.from_wrapped_field(wf) for wf in wrapped_class.fields
        ]

        # Add role-introduced fields as init=False
        for role_wc in self._root_role_taker_to_roles_map.get(wrapped_class.clazz, []):
            taker_field_name = role_wc.clazz.role_taker_field().name
            for role_wf in role_wc.fields:
                if any(taker_wf.name == role_wf.name for taker_wf in taker_fields):
                    raise ValueError(
                        f"Roles should not overwrite fields defined in their role takers: {role_wf.name} in "
                        f"{role_wc} overwrites the one defined in {wrapped_class} with the same name"
                    )
                if role_wf.name != taker_field_name:
                    taker_fields.append(
                        StubFieldInfo(
                            role_wf.name,
                            role_wf.type_name,
                            FieldRepresentation(field(init=False)),
                        )
                    )

        dc_args = DataclassArguments.from_wrapped_class(wrapped_class)

        return StubClassInfo(
            wrapped_class.name,
            [clazz.__name__ for clazz in wrapped_class.clazz.__bases__],
            taker_fields,
            dc_args,
        )

    @cached_property
    def _role_taker_to_info_map(self) -> Dict[Type, RoleTakerInfo]:
        """
        :return: A mapping from role taker types to their information.
        """
        return {
            taker_type: RoleTakerInfo.from_taker_wrapped_class_and_roles(
                self.class_diagram.get_wrapped_class(taker_type), roles
            )
            for taker_type, roles in self._role_taker_to_roles_map.items()
        }

    @cached_property
    def _role_taker_to_roles_map(self) -> Dict[Type, List[WrappedClass[Role]]]:
        """
        :return: A mapping from role taker types to their roles.
        """
        taker_to_roles = defaultdict(list)
        for role_wc in self._role_wrapped_classes:
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
            if issubclass(wc.clazz, Role)
        ]

    def _extract_imports(self) -> List[str]:
        """
        Extract imports from the module source.

        :return: A list of string import statements.
        """
        return extract_imports(
            self.module,
            [
                Role.__module__,
                __future__.__name__,
                dataclasses.__name__,
            ],
        )
