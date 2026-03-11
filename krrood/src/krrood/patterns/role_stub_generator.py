from __future__ import annotations

import inspect
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, Field, field, fields
from functools import cached_property
from typing import Any, Type, List, Dict, Optional

import jinja2

from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.utils import classes_of_module
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.patterns import Role


@dataclass(frozen=True)
class Assignment:
    name: str
    value: Any

    def __str__(self) -> str:
        return f"{self.name}={self.value}"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class FieldRepresentation:
    """Represents a field in a .pyi file."""

    current_field: Field = field(default_factory=field)

    @classmethod
    def from_wrapped_field(cls, wf: WrappedField) -> FieldRepresentation:
        """Creates a FieldAssignment instance from a WrappedField."""
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
        non_default_field_assignments = []
        default_field = field()
        field_arguments = inspect.signature(field).parameters
        for parameter in field_arguments.values():
            current_value = getattr(self.current_field, parameter.name)
            if current_value != getattr(default_field, parameter.name):
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
    """Represents arguments for the @dataclass decorator."""

    eq: bool = True
    unsafe_hash: bool = False
    kw_only: bool = False

    @classmethod
    def from_wrapped_class(cls, wrapped_class: WrappedClass) -> DataclassArguments:
        """Creates a DataclassArguments instance from a WrappedClass."""
        params = getattr(wrapped_class.clazz, "__dataclass_params__", None)
        return cls(
            eq=params.eq if params else True,
            unsafe_hash=params.unsafe_hash if params else False,
            kw_only=getattr(params, "kw_only", False) if params else False,
        )

    def __str__(self) -> str:
        dataclass_params = inspect.signature(dataclass).parameters
        non_default_dataclass_params = []
        for field_ in fields(self):
            value = getattr(self, field_.name)
            if value != dataclass_params[field_.name].default:
                non_default_dataclass_params.append(Assignment(field_.name, value))
        return ", ".join(map(str, non_default_dataclass_params))


@dataclass
class StubFieldInfo:
    """Information about a field as it should appear in the stub."""

    name: str
    type_name: str
    assignment: FieldRepresentation

    @classmethod
    def from_wrapped_field(cls, wf: WrappedField) -> StubFieldInfo:
        """Creates a StubFieldInfo instance from a WrappedField."""
        return cls(wf.name, wf.type_name, FieldRepresentation.from_wrapped_field(wf))


@dataclass
class StubClassInfo:
    """Information about a class as it should appear in the stub."""

    name: str
    bases: List[str]
    fields: List[StubFieldInfo]
    dataclass_args: DataclassArguments


@dataclass
class RoleTakerInfo:
    """Information about a role taker for stub generation."""

    role_for_name: str
    taker_field_name: str
    inherited_fields: List[StubFieldInfo]
    roles: List[RoleInfo]

    @classmethod
    def from_taker_wrapped_class_and_roles(
        cls, taker_wc: WrappedClass, roles: List[WrappedClass[Role]]
    ):
        """Creates a RoleTakerInfo instance from a role taker type."""
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


@dataclass
class RoleInfo:
    """Information about a role for stub generation."""

    name: str
    dataclass_args: DataclassArguments
    introduced_field: Optional[StubFieldInfo] = None

    @classmethod
    def from_wrapped_class(cls, role: WrappedClass[Role]) -> RoleInfo:
        """Creates a RoleInfo instance from a WrappedClass."""
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
    Automates the generation of stub python files (.pyi) for classes that follow the Role Pattern.
    """

    def __init__(self, module: Any):
        """Initializes the generator with the Jinja template."""
        loader = jinja2.PackageLoader("krrood.patterns", "templates")
        self.env = jinja2.Environment(
            loader=loader, trim_blocks=True, lstrip_blocks=True
        )
        self.template = self.env.get_template("role_stub.pyi.jinja")
        self.class_diagram = ClassDiagram(classes_of_module(module))
        self.module = module

    def generate_stub(self) -> str:
        """
        Generates a stub file for the given module.

        :return: The generated stub file as a string.
        """
        return self.template.render(
            stub_classes=self._non_role_stub_classes,
            role_takers=self._role_taker_to_info_map,
            imports=self._extract_imports(),
        )

    @cached_property
    def _non_role_stub_classes(self) -> List[StubClassInfo]:
        """Stub Classes for non-role entities."""
        return [
            self._build_stub_class(wc)
            for wc in self.class_diagram.wrapped_classes
            if not issubclass(wc.clazz, Role)
        ]

    @cached_property
    def _root_role_taker_to_roles_map(self) -> Dict[Type, List[WrappedClass]]:
        """Mapping from root role taker types to their roles."""
        mapping = defaultdict(list)
        for wc in self.class_diagram.wrapped_classes:
            if issubclass(wc.clazz, Role):
                mapping[wc.root_role_taker_type].append(wc)
        return mapping

    def _build_stub_class(self, wrapped_class: WrappedClass) -> StubClassInfo:
        """Builds stub information for a non-role class."""
        # Add original fields
        taker_fields = [
            StubFieldInfo.from_wrapped_field(wf) for wf in wrapped_class.fields
        ]

        # Add role-introduced fields as init=False
        for role_wc in self._role_taker_to_roles_map.get(wrapped_class.clazz, []):
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
        """Prepares role taker metadata for the template."""
        return {
            taker_type: RoleTakerInfo.from_taker_wrapped_class_and_roles(
                self.class_diagram.get_wrapped_class(taker_type), roles
            )
            for taker_type, roles in self._role_taker_to_roles_map.items()
        }

    @cached_property
    def _role_taker_to_roles_map(self) -> Dict[Type, List[WrappedClass[Role]]]:
        """Mapping from role taker types to their roles."""
        taker_to_roles = defaultdict(list)
        for role_wc in self._role_wrapped_classes:
            taker_type = role_wc.clazz.get_role_taker_type()
            taker_to_roles[taker_type].append(role_wc)
        return taker_to_roles

    @cached_property
    def _role_wrapped_classes(self) -> List[WrappedClass[Role]]:
        """Returns a list of WrappedClass instances for role classes."""
        return [
            wc
            for wc in self.class_diagram.wrapped_classes
            if issubclass(wc.clazz, Role)
        ]

    def _extract_imports(self) -> List[str]:
        """Extracts imports from module source, excluding internal role/dataclass modules."""
        lines, _ = inspect.getsourcelines(self.module)
        forbidden = {"krrood.patterns.role", "dataclasses", "typing", "__future__"}
        imports = {
            line.strip()
            for line in lines
            if line.strip().startswith(("from ", "import "))
            and not any(f in line for f in forbidden)
        }
        return sorted(list(imports))
