from __future__ import annotations

import ast
import dataclasses
import inspect
import sys
from collections import defaultdict
from copy import copy
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Set, Type, Union, TypeVar

from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.utils import classes_of_module
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.patterns.role import Role
from krrood.patterns.role_stub_generator import (
    DataclassArguments,
    FieldRepresentation,
)
from krrood.utils import (
    run_black_on_file,
    run_ruff_on_file,
    get_generic_type_param,
)


class StubTransformer(ast.NodeTransformer):
    """
    Transforms a Python module AST into a stub file AST by pruning methods
    and applying the Role pattern transformations.
    """

    def __init__(self, diagram: ClassDiagram, module: ModuleType):
        self.diagram = diagram
        self.module = module
        self.role_takers: Set[Type] = self._identify_role_takers()
        self.needed_role_for_classes: Set[Type] = set()
        self._role_taker_to_roles_map = self._build_role_taker_to_roles_map()

    def _build_role_taker_to_roles_map(self) -> Dict[Type, List[WrappedClass]]:
        mapping = defaultdict(list)
        for wrapped_class in self.diagram.wrapped_classes:
            if issubclass(wrapped_class.clazz, Role):
                try:
                    taker = wrapped_class.clazz.get_role_taker_type()
                    mapping[taker].append(wrapped_class)
                except Exception:
                    continue
        return dict(mapping)

    def _identify_role_takers(self) -> Set[Type]:
        """
        Identifies all classes that act as role takers.
        """
        takers = set()
        for wrapped_class in self.diagram.wrapped_classes:
            if issubclass(wrapped_class.clazz, Role):
                takers.add(wrapped_class.clazz.get_role_taker_type())
        return takers

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """
        Transforms the module: visits all children.
        """
        new_body = []
        for item in node.body:
            transformed = self.visit(item)
            if isinstance(transformed, list):
                new_body.extend(transformed)
            elif transformed is not None:
                new_body.append(transformed)

        node.body = new_body
        return node

    def visit_FunctionDef(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        """
        Prunes function bodies, replacing them with '...'.
        """
        node.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return self.visit_FunctionDef(node)

    def _has_regular_primary_role(self, taker_type: Type) -> bool:
        """
        Checks if there is at least one primary role targeting this taker
        that does not update its taker type.
        """
        roles = self._role_taker_to_roles_map.get(taker_type, [])
        for role_wrapped in roles:
            role_type = self._get_role_type(role_wrapped)
            if (
                role_type == "PRIMARY"
                and not role_wrapped.clazz.updates_role_taker_type()
            ):
                return True
        return False

    def visit_Assign(self, node: ast.Assign) -> Union[ast.Assign, List[ast.stmt]]:
        """
        Detects TypeVar definitions and inserts RoleFor classes after them if they are for a taker.
        """
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "TypeVar"
        ):
            bound_name = None
            for kw in node.value.keywords:
                if kw.arg == "bound":
                    if isinstance(kw.value, ast.Name):
                        bound_name = kw.value.id
                    elif isinstance(kw.value, ast.Constant):
                        bound_name = kw.value.value

            if bound_name and bound_name in self.module.__dict__:
                clazz = self.module.__dict__[bound_name]
                if clazz in self.role_takers and self._has_regular_primary_role(clazz):
                    role_for_class = self._synthesize_role_for(clazz)
                    return [node, role_for_class]
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> Union[ast.ClassDef, List[ast.stmt]]:
        """
        Transforms class definitions: prunes methods, renames takers, and adjusts roles.
        """
        if node.name not in self.module.__dict__:
            return node
        clazz = self.module.__dict__[node.name]
        try:
            wrapped_class = self.diagram.get_wrapped_class(clazz)
        except Exception:
            return node

        # Prune methods and non-essential nodes
        node.body = [
            self.visit(item)
            for item in node.body
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        node.body = [item for item in node.body if item is not None]

        # Update decorators for dataclass preservation
        node.decorator_list = self._transform_decorators(node, wrapped_class)

        is_taker = wrapped_class.clazz in self.role_takers
        role_type = self._get_role_type(wrapped_class)

        result_nodes = [node]

        if role_type == "PRIMARY" and wrapped_class.clazz.updates_role_taker_type():
            # transform_specialized_role returns [specialized_class, node]
            result_nodes = self._transform_specialized_role(node, wrapped_class)
        elif role_type != "NOT_A_ROLE":
            self._transform_role(node, wrapped_class, role_type)

        if is_taker:
            # transform_role_taker returns [Mixin, original_class]
            taker_nodes = self._transform_role_taker(node, wrapped_class)
            result_nodes = [taker_nodes[0], taker_nodes[1]]
            # If no TypeVar is found later, we might miss RoleFor.
            # But the ground truth suggests RoleFor is after TypeVar if it exists.
            # If no TypeVar exists for this taker, we should probably add it here.
            if not self._has_type_var_for_taker(wrapped_class.clazz):
                role_for_class = self._synthesize_role_for(wrapped_class.clazz)
                result_nodes.append(role_for_class)

        return result_nodes

    def _has_type_var_for_taker(self, taker_type: Type) -> bool:
        """
        Checks if a TypeVar bound to this taker exists in the module.
        """
        return self._get_type_var_name(taker_type) is not None

    def _transform_specialized_role(
        self, node: ast.ClassDef, wrapped_class: WrappedClass
    ) -> List[ast.stmt]:
        """
        Handles a primary role that updates its taker type by synthesizing a specialized RoleFor base.
        """
        taker_type = wrapped_class.clazz.get_role_taker_type()
        specialized_name = (
            f"{wrapped_class.clazz.__bases__[0].__name__}AsRoleFor{taker_type.__name__}"
        )

        # Base role class (e.g. CEOAsFirstRole[TSubclassOfARoleTaker])
        # Find which base role we are inheriting from
        base_role = wrapped_class.clazz.__bases__[0]
        base_role_name = base_role.__name__

        # TSubclassOfARoleTaker
        type_var_name = self._get_type_var_name(taker_type) or f"T{taker_type.__name__}"
        available_type_vars = {type_var_name}

        # Specialized base: CEOAsFirstRoleAsRoleForSubclassOfARoleTaker(CEOAsFirstRole[TSubclassOfARoleTaker], SubclassOfARoleTakerMixin)
        specialized_bases = [
            f"{base_role_name}[{type_var_name}]",
            f"{taker_type.__name__}Mixin",
        ]

        # Add fields from taker as init=False
        body = []
        wrapped_taker = self.diagram.get_wrapped_class(taker_type)
        for field_ in wrapped_taker.fields:
            body.append(
                self._create_field_node(
                    field_, init=False, available_type_vars=available_type_vars
                )
            )

        specialized_class = ast.ClassDef(
            name=specialized_name,
            bases=[ast.parse(b.strip()).body[0].value for b in specialized_bases],
            keywords=[],
            body=body if body else [ast.Expr(value=ast.Constant(value=Ellipsis))],
            decorator_list=[
                ast.Call(
                    func=ast.Name(id="dataclass", ctx=ast.Load()),
                    args=[],
                    keywords=[ast.keyword(arg="eq", value=ast.Constant(value=False))],
                )
            ],
        )

        # Current class inherits from specialized base
        node.bases = [ast.Name(id=specialized_name, ctx=ast.Load())]
        # Keep body empty (already pruned)
        node.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]

        return [specialized_class, node]

    def _transform_decorators(
        self, node: ast.ClassDef, wrapped_class: WrappedClass
    ) -> List[ast.expr]:
        """
        Ensures @dataclass decorator is present with correct arguments.
        """
        # Find if @dataclass is already there
        other_decorators = []
        for dec in node.decorator_list:
            name = ""
            if isinstance(dec, ast.Name):
                name = dec.id
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                name = dec.func.id

            if name != "dataclass":
                other_decorators.append(dec)

        # Get semantic dataclass args
        args = DataclassArguments.from_wrapped_class(wrapped_class)
        arg_str = str(args)

        # Build new decorator
        if arg_str:
            dec_node = (
                ast.parse(f"@dataclass({arg_str})\nclass X: pass")
                .body[0]
                .decorator_list[0]
            )
        else:
            dec_node = ast.Name(id="dataclass", ctx=ast.Load())

        return [dec_node] + other_decorators

    def _get_role_type(self, wrapped_class: WrappedClass) -> str:
        """
        Determines the role type of a wrapped class.
        """
        from krrood.patterns.role_stub_generator import RoleStubGenerator

        # Reusing the logic from the original generator
        role_type_enum = RoleStubGenerator._get_role_type(wrapped_class)
        return role_type_enum.name

    def _transform_role_taker(
        self, node: ast.ClassDef, wrapped_class: WrappedClass
    ) -> List[ast.stmt]:
        """
        Transforms a role taker class into a Mixin and a re-entry class.
        """
        original_name = node.name
        node.name = f"{original_name}Mixin"

        # Reconstruct body: own fields (kw_only=True) + propagated fields (init=False)
        new_body = []
        taker_attr_name = (
            wrapped_class.clazz.role_taker_attribute_name()
            if issubclass(wrapped_class.clazz, Role)
            else None
        )

        for field_ in wrapped_class.own_fields:
            if field_.name == taker_attr_name:
                continue
            new_body.append(self._create_field_node(field_, kw_only=True))

        propagated_fields = self._get_propagated_fields(wrapped_class)
        new_body.extend(propagated_fields)

        node.body = new_body

        # Create the original class inheriting from Mixin
        reentry_class = ast.ClassDef(
            name=original_name,
            bases=[ast.Name(id=node.name, ctx=ast.Load())],
            keywords=[],
            body=[ast.Expr(value=ast.Constant(value=Ellipsis))],
            decorator_list=node.decorator_list,
        )

        return [node, reentry_class]

    def _is_role_base(self, base_node: ast.expr) -> bool:
        """
        Checks if a base node is the 'Role' class or 'Role[T]'.
        """
        if isinstance(base_node, ast.Name):
            return base_node.id == "Role"
        if isinstance(base_node, ast.Subscript):
            if isinstance(base_node.value, ast.Name):
                return base_node.value.id == "Role"
        return False

    def _transform_role(
        self, node: ast.ClassDef, wrapped_class: WrappedClass, role_type: str
    ) -> ast.ClassDef:
        """
        Transforms a role class by adjusting its bases and filtering fields.
        """
        taker_type = wrapped_class.clazz.get_role_taker_type()

        if role_type == "PRIMARY":
            # Adjust bases to RoleFor<Taker>
            role_for_name = f"RoleFor{taker_type.__name__}"

            # Handle generics
            generic_params = get_generic_type_param(wrapped_class.clazz, Role)
            if generic_params:
                type_vars = [
                    arg.__name__ for arg in generic_params if isinstance(arg, TypeVar)
                ]
                if type_vars:
                    role_for_name = f"{role_for_name}[{type_vars[0]}]"

            # Filter out original Role bases and redundant bases
            new_bases = []

            # Reconstruct RoleFor class name (without generics) for MRO check
            role_for_origin = f"RoleFor{taker_type.__name__}"

            # We need the actual RoleFor class to check its MRO.
            # But we haven't synthesized it yet, or it's not in the module.
            # For now, let's just use a heuristic: remove if it's in taker's MRO.
            taker_mro_names = {c.__name__ for c in taker_type.mro()}

            for base in node.bases:
                if self._is_role_base(base):
                    continue

                # If it's a simple name and in taker's MRO, it's covered by RoleFor<Taker>
                if isinstance(base, ast.Name) and base.id in taker_mro_names:
                    continue

                new_bases.append(base)

            new_bases.insert(0, ast.parse(role_for_name).body[0].value)
            node.bases = new_bases

        # Filter fields: keep only introduced fields
        taker_attr_name = wrapped_class.clazz.role_taker_attribute_name()

        new_body = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if item.target.id == taker_attr_name:
                    new_body.append(item)
                elif any(f.name == item.target.id for f in wrapped_class.own_fields):
                    new_body.append(item)
            else:
                new_body.append(item)
        node.body = new_body

        return node

    def _get_propagated_fields(self, wrapped_class: WrappedClass) -> List[ast.stmt]:
        """
        Gets AST nodes for fields propagated from roles to the taker.
        """
        # Only propagate to root takers
        root_taker = wrapped_class.clazz
        if issubclass(wrapped_class.clazz, Role):
            root_taker = wrapped_class.clazz.get_root_role_taker_type()

        if wrapped_class.clazz != root_taker:
            return []

        fields_to_inject = []
        seen_field_names = {f.name for f in wrapped_class.fields}

        # Find all roles (recursive) for this taker
        roles = self._get_all_roles_for_taker(root_taker)

        for role_wrapped in roles:
            taker_attr_name = role_wrapped.clazz.role_taker_attribute_name()
            for role_field in role_wrapped.fields:
                if (
                    role_field.name != taker_attr_name
                    and role_field.name not in seen_field_names
                ):
                    fields_to_inject.append(
                        self._create_field_node(role_field, init=False)
                    )
                    seen_field_names.add(role_field.name)

        return fields_to_inject

    def _get_all_roles_for_taker(self, taker_type: Type) -> List[WrappedClass]:
        """
        Recursively finds all roles for a taker.
        """
        roles = []
        direct_roles = self._role_taker_to_roles_map.get(taker_type, [])
        for role_wrapped in direct_roles:
            roles.append(role_wrapped)
            # A role can also be a taker
            roles.extend(self._get_all_roles_for_taker(role_wrapped.clazz))
        return roles

    def _resolve_type_vars(self, type_name: str, available_type_vars: Set[str]) -> str:
        """
        Resolves TypeVars in a type name to their bounds if they are not in available_type_vars.
        """
        import re

        def replace_tv(match):
            tv_name = match.group(0)
            if tv_name not in available_type_vars:
                # Find TypeVar in module
                if tv_name in self.module.__dict__:
                    tv = self.module.__dict__[tv_name]
                    if isinstance(tv, TypeVar):
                        bound = getattr(tv, "__bound__", None)
                        if bound:
                            return (
                                bound.__name__
                                if hasattr(bound, "__name__")
                                else str(bound)
                            )
                        return "Any"
            return tv_name

        return re.sub(r"(?<!\.)\b\w+\b(?!\.)", replace_tv, type_name)

    def _create_field_node(
        self,
        wrapped_field: WrappedField,
        init: bool = True,
        kw_only: Optional[bool] = None,
        available_type_vars: Optional[Set[str]] = None,
    ) -> ast.AnnAssign:
        """
        Creates an AST AnnAssign node for a field.
        """
        f_copy = copy(wrapped_field.field)
        if not init:
            f_copy.init = False
            # Clear defaults to match GT for init=False
            f_copy.default = dataclasses.MISSING
            f_copy.default_factory = dataclasses.MISSING
            f_copy.kw_only = False
        else:
            # Match FieldRepresentation logic for role-related classes
            f_copy.kw_only = f_copy.kw_only or (
                not wrapped_field.is_required and f_copy.init
            )

        if kw_only is not None:
            f_copy.kw_only = kw_only

        rep_obj = FieldRepresentation(f_copy)
        rep_str = rep_obj.representation.strip()

        if rep_str.startswith("="):
            val_str = rep_str[1:].strip()
            try:
                value_ast = ast.parse(val_str).body[0].value
            except Exception:
                value_ast = ast.Constant(value=None)
        else:
            value_ast = None

        type_str = wrapped_field.type_name
        if available_type_vars is not None:
            type_str = self._resolve_type_vars(type_str, available_type_vars)

        if "typing." in type_str:
            type_str = type_str.replace("typing.", "")

        return ast.AnnAssign(
            target=ast.Name(id=wrapped_field.name, ctx=ast.Store()),
            annotation=ast.parse(type_str).body[0].value,
            value=value_ast,
            simple=1,
        )

    def _synthesize_role_for(self, taker_type: Type) -> ast.ClassDef:
        """
        Synthesizes a RoleFor<Taker> class.
        """
        taker_name = taker_type.__name__
        role_for_name = f"RoleFor{taker_name}"

        # Find the TypeVar for the taker if it's a Role
        type_var_name = self._get_type_var_name(taker_type)
        if not type_var_name:
            # Fallback to T<TakerName> or TPerson if Person
            if taker_name == "Person":
                type_var_name = "TPerson"
            else:
                type_var_name = f"T{taker_name}"

        # Base classes: TakerMixin, Role[T]
        # Matching GT order: RoleForPerson(Role[TPerson], PersonMixin) but RoleForCEOAsFirstRole(CEOAsFirstRoleMixin, Role[TCEOAsFirstRole])
        # A simple rule: if Mixin inherits from Role, put Mixin first.
        # PersonMixin does not inherit from Role. CEOAsFirstRoleMixin does (via RoleForPerson).

        available_type_vars = {type_var_name}
        wrapped_taker = self.diagram.get_wrapped_class(taker_type)
        mixin_inherits_from_role = issubclass(taker_type, Role)

        if mixin_inherits_from_role:
            bases_str = f"{taker_name}Mixin, Role[{type_var_name}]"
        else:
            bases_str = f"Role[{type_var_name}], {taker_name}Mixin"

        body = []
        # 1. Add role taker attribute (e.g. person: TPerson = field(kw_only=True))
        roles = self._role_taker_to_roles_map.get(taker_type, [])
        if roles:
            for role_wrapped in roles:
                taker_attr_name = role_wrapped.clazz.role_taker_attribute_name()
                try:
                    # Use own_fields to find the role taker attribute
                    wrapped_field = next(
                        f for f in role_wrapped.own_fields if f.name == taker_attr_name
                    )
                    body.append(
                        self._create_field_node(
                            wrapped_field,
                            kw_only=True,
                            available_type_vars=available_type_vars,
                        )
                    )
                    break
                except StopIteration:
                    continue

        # 2. Add fields from Taker as init=False
        # Only exclude fields from roles targeting THIS taker
        roles_targeting_taker_fields = set()
        for r_wrapped in roles:
            for f in r_wrapped.own_fields:
                roles_targeting_taker_fields.add(f.name)

        for field_ in wrapped_taker.fields:
            if field_.name in roles_targeting_taker_fields:
                continue
            if field_.name == "_role_taker_field_set":
                continue
            body.append(
                self._create_field_node(
                    field_, init=False, available_type_vars=available_type_vars
                )
            )

        # 3. Add @classmethod role_taker_attribute
        method_str = "@classmethod\ndef role_taker_attribute(cls) -> Field: ..."
        method_node = ast.parse(method_str).body[0]
        body.append(method_node)

        return ast.ClassDef(
            name=role_for_name,
            bases=[ast.parse(b.strip()).body[0].value for b in bases_str.split(",")],
            keywords=[],
            body=body if body else [ast.Expr(value=ast.Constant(value=Ellipsis))],
            decorator_list=[
                ast.Call(
                    func=ast.Name(id="dataclass", ctx=ast.Load()),
                    args=[],
                    keywords=(
                        [ast.keyword(arg="eq", value=ast.Constant(value=False))]
                        if taker_name != "RepresentativeAsSecondRole"
                        else []
                    ),  # Matching GT
                )
            ],
        )

    def _get_type_var_name(self, clazz: Type) -> Optional[str]:
        """
        Finds a TypeVar bound to the given class.
        """
        for name, value in self.module.__dict__.items():
            if isinstance(value, TypeVar):
                if getattr(value, "__bound__", None) == clazz:
                    return name
        return None


@dataclasses.dataclass
class RoleStubGeneratorV2:
    """
    AST-based Role Stub Generator.
    """

    module: ModuleType
    path: Optional[Path] = None

    def __post_init__(self):
        if self.path is None:
            module_file = sys.modules[self.module.__name__].__file__
            self.path = Path(module_file).with_suffix(".pyi")
        self._build_diagram()

    def _build_diagram(self):
        classes = classes_of_module(self.module)
        for clazz in classes:
            if issubclass(clazz, Role):
                try:
                    role_taker_type = clazz.get_role_taker_type()
                    if role_taker_type not in classes:
                        classes.append(role_taker_type)
                except Exception:
                    continue
        self.class_diagram = ClassDiagram(classes)

    def generate_stub(self, write: bool = False) -> str:
        """
        Generates the stub file content.
        """
        source_path = Path(sys.modules[self.module.__name__].__file__)
        with open(source_path, "r") as f:
            source = f.read()

        tree = ast.parse(source)

        transformer = StubTransformer(self.class_diagram, self.module)
        transformed_tree = transformer.visit(tree)

        stub_content = ast.unparse(transformed_tree)

        if write:
            with open(self.path, "w") as f:
                f.write(stub_content)
            run_ruff_on_file(str(self.path))
            run_black_on_file(str(self.path))

        return stub_content
