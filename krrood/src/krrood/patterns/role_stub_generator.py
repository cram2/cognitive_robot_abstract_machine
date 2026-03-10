from __future__ import annotations

import inspect
from dataclasses import dataclass, is_dataclass, field, MISSING, Field
from typing import Any, Type, List, Dict, Optional, Tuple, Union

import jinja2

from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.utils import classes_of_module

class _Missing:
    def __repr__(self):
        return "MISSING"

MISSING_VAL = _Missing()

@dataclass
class FieldAssignment:
    """Represents a field assignment in a .pyi file."""
    init: bool = True
    kw_only: bool = False
    default: Any = MISSING_VAL
    default_factory: Any = MISSING_VAL

    def __str__(self) -> str:
        # Use the automatically generated repr and perform minimal edits
        content = repr(self).split("(", 1)[1][:-1]
        
        # Filter out default flags and MISSING markers
        parts = [p for p in content.split(", ") 
                 if p not in ("init=True", "kw_only=False") and "MISSING" not in p]
        
        if not parts:
            return ""

        # Handle simple assignment (e.g., " = value")
        if len(parts) == 1 and parts[0].startswith("default="):
            return f" = {parts[0].split('=', 1)[1]}"

        # Format as field(...) and clean up type names (e.g., <class 'list'> -> list)
        args_str = ", ".join(parts).replace("<class '", "").replace("'>", "")
        return f" = field({args_str})"

@dataclass
class DataclassArguments:
    """Represents arguments for the @dataclass decorator."""
    eq: bool = True
    unsafe_hash: bool = False
    kw_only: bool = False

    def __str__(self) -> str:
        # Use repr() and filter out arguments that match standard @dataclass defaults
        content = repr(self).split("(", 1)[1][:-1]
        parts = [p for p in content.split(", ") 
                 if p not in ("eq=True", "unsafe_hash=False", "kw_only=False")]
        return ", ".join(parts)

@dataclass
class StubFieldInfo:
    """Information about a field as it should appear in the stub."""
    name: str
    type_name: str
    assignment: FieldAssignment

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

@dataclass
class RoleInfo:
    """Information about a role for stub generation."""
    name: str
    dataclass_args: DataclassArguments
    introduced_field: StubFieldInfo

class RoleStubGenerator:
    """
    Automates the generation of stub python files (.pyi) for classes that follow the Role Pattern.
    """

    def __init__(self):
        """Initializes the generator with the Jinja template."""
        loader = jinja2.PackageLoader("krrood.patterns", "templates")
        self.env = jinja2.Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
        self.template = self.env.get_template("role_stub.pyi.jinja")

    def generate_stub(self, module: Any) -> str:
        """
        Generates a stub file for the given module.

        :param module: The Python module to analyze.
        :return: The generated stub file as a string.
        """
        diagram = ClassDiagram(classes_of_module(module))
        
        # 1. Prepare Stub Classes for non-role entities
        stub_classes = []
        root_to_roles = self._map_root_to_roles(diagram)
        
        for wc in diagram.wrapped_classes:
            if not wc.is_role:
                stub_classes.append(self._build_stub_class(wc, root_to_roles.get(wc.name, [])))

        # 2. Prepare Role Taker Hierarchy
        role_takers_map = self._prepare_role_takers_map(diagram)
        
        return self.template.render(
            stub_classes=stub_classes,
            role_takers=role_takers_map,
            imports=self._extract_imports(module)
        )

    def _map_root_to_roles(self, diagram: ClassDiagram) -> Dict[str, List[WrappedClass]]:
        """Maps root role taker names to their roles."""
        mapping = {}
        for wc in diagram.wrapped_classes:
            if wc.is_role:
                root_type = wc.root_role_taker_type
                root_name = root_type.__origin__.__name__ if hasattr(root_type, "__origin__") else root_type.__name__
                mapping.setdefault(root_name, []).append(wc)
        return mapping

    def _build_stub_class(self, wrapped_class: WrappedClass, roles: List[WrappedClass]) -> StubClassInfo:
        """Builds stub information for a non-role class."""
        fields = []
        # Add original fields
        for wf in wrapped_class.fields:
            assignment = FieldAssignment(
                init=wf.field.init,
                kw_only=getattr(wf.field, "kw_only", False) or (not wf.is_required and wf.field.init),
                default=wf.field.default if wf.field.default is not MISSING else MISSING_VAL,
                default_factory=wf.field.default_factory if wf.field.default_factory is not MISSING else MISSING_VAL
            )
            fields.append(StubFieldInfo(wf.name, wf.type_name, assignment))
            
        # Add role-introduced fields as init=False
        for role_wc in roles:
            taker_field_name = role_wc.clazz.role_taker_field().name
            for wf in role_wc.fields:
                if wf.name != taker_field_name and not any(f.name == wf.name for f in fields):
                    fields.append(StubFieldInfo(wf.name, wf.type_name, FieldAssignment(init=False)))

        params = getattr(wrapped_class.clazz, "__dataclass_params__", None)
        dc_args = DataclassArguments(
            eq=params.eq if params else True,
            unsafe_hash=params.unsafe_hash if params else False,
            kw_only=getattr(params, "kw_only", False) if params else False
        )

        return StubClassInfo(wrapped_class.name, wrapped_class.bases, fields, dc_args)

    def _prepare_role_takers_map(self, diagram: ClassDiagram) -> Dict[str, RoleTakerInfo]:
        """Prepares role taker metadata for the template."""
        role_classes = [wc for wc in diagram.wrapped_classes if wc.is_role]
        taker_to_roles = {}
        for role_wc in role_classes:
            taker_type = role_wc.role_taker_type
            taker_name = taker_type.__origin__.__name__ if hasattr(taker_type, "__origin__") else taker_type.__name__
            taker_to_roles.setdefault(taker_name, []).append(role_wc)

        result = {}
        for taker_name, roles in taker_to_roles.items():
            taker_field_name = roles[0].clazz.role_taker_field().name
            
            role_infos = []
            for r in roles:
                intro_field_wc = next((wf for wf in r.fields if wf.name != taker_field_name), None)
                if intro_field_wc:
                    assignment = FieldAssignment(
                        init=intro_field_wc.field.init,
                        kw_only=True, # Roles' introduced fields are always kw_only in stub
                        default=intro_field_wc.field.default if intro_field_wc.field.default is not MISSING else MISSING_VAL,
                        default_factory=intro_field_wc.field.default_factory if intro_field_wc.field.default_factory is not MISSING else MISSING_VAL
                    )
                    intro_field_stub = StubFieldInfo(intro_field_wc.name, intro_field_wc.type_name, assignment)
                    
                    params = getattr(r.clazz, "__dataclass_params__", None)
                    dc_args = DataclassArguments(
                        eq=params.eq if params else True,
                        unsafe_hash=params.unsafe_hash if params else False,
                        kw_only=getattr(params, "kw_only", False) if params else False
                    )
                    role_infos.append(RoleInfo(name=r.name, dataclass_args=dc_args, introduced_field=intro_field_stub))

            # Inherited fields are all fields of the taker that are init=True
            taker_wc = diagram.get_wrapped_class(next(c.role_taker_type for c in roles if self._get_name(c.role_taker_type) == taker_name))
            inherited_fields = [
                StubFieldInfo(wf.name, wf.type_name, FieldAssignment(init=False))
                for wf in taker_wc.fields if wf.field.init
            ]

            result[taker_name] = RoleTakerInfo(
                role_for_name=self._generate_role_for_name(taker_name),
                taker_field_name=taker_field_name,
                inherited_fields=inherited_fields,
                roles=role_infos
            )
        return result

    def _get_name(self, cls_type: Type) -> str:
        """Returns the name of a class type, handling GenericAlias."""
        return cls_type.__origin__.__name__ if hasattr(cls_type, "__origin__") else cls_type.__name__

    def _generate_role_for_name(self, taker_name: str) -> str:
        """Generates the RoleFor[Class] name."""
        name = f"RoleFor{taker_name}"
        for suffix in ["AsFirstRole", "AsSecondRole", "AsThirdRole"]:
            name = name.replace(suffix, "")
        return name

    def _extract_imports(self, module: Any) -> List[str]:
        """Extracts imports from module source, excluding internal role/dataclass modules."""
        lines, _ = inspect.getsourcelines(module)
        forbidden = {"krrood.patterns.role", "dataclasses", "typing", "__future__"}
        imports = {line.strip() for line in lines if line.strip().startswith(("from ", "import ")) 
                   and not any(f in line for f in forbidden)}
        return sorted(list(imports))
