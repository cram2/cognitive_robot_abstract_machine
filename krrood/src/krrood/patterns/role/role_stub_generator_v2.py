from __future__ import annotations

import ast
import dataclasses
import sys
from collections import defaultdict
from copy import copy
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Set, Type, Union, TypeVar, Callable

from black.handle_ipynb_magics import lru_cache

from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.utils import classes_of_module
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.patterns.role.meta_data import RoleType
from krrood.patterns.role.role import Role
from krrood.patterns.role.role_stub_generator import (
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
        self.needed_role_for_classes: Set[Type] = set()
        self._role_taker_to_roles_map = self._build_role_taker_to_roles_map()

    def _build_role_taker_to_roles_map(self) -> Dict[Type, List[WrappedClass]]:
        mapping = defaultdict(list)
        for wrapped_class in self.diagram.wrapped_classes:
            if issubclass(wrapped_class.clazz, Role):
                taker = wrapped_class.clazz.get_role_taker_type()
                mapping[taker].append(wrapped_class)
        return dict(mapping)

    @property
    def role_takers(self) -> Set[Type]:
        """
        :return: A set of all classes that act as role takers.
        """
        return set(self._role_taker_to_roles_map.keys())

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

    @lru_cache
    def _has_primary_role(self, taker_type: Type) -> bool:
        """
        Checks if there is at least one primary role targeting this taker
        that does not update its taker type.
        """
        roles = self._role_taker_to_roles_map.get(taker_type, [])
        return any(RoleType.get_role_type(role_wrapped) == RoleType.PRIMARY for role_wrapped in roles)

    @lru_cache
    def _get_primary_roles(self, taker_type: Type) -> List[WrappedClass[Role]]:
        """
        :return: A list of all primary roles targeting this taker.
        """
        roles = self._role_taker_to_roles_map.get(taker_type, [])
        return [role_wrapped for role_wrapped in roles if RoleType.get_role_type(role_wrapped) == RoleType.PRIMARY]

    def visit_Assign(self, node: ast.Assign) -> Union[ast.Assign, List[ast.stmt]]:
        """
        Detects TypeVar definitions and inserts RoleFor classes after them if they are for a taker.
        """
        if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == TypeVar.__name__
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
                if clazz in self.role_takers and self._has_primary_role(clazz):
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
        role_type = RoleType.get_role_type(wrapped_class)

        result_nodes = [node]

        if role_type == RoleType.PRIMARY and wrapped_class.clazz.updates_role_taker_type():
            # transform_specialized_role returns [specialized_class, node]
            result_nodes = self._transform_specialized_role(node, wrapped_class)
        elif role_type != RoleType.NOT_A_ROLE:
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

    @staticmethod
    def _transform_decorators(
            node: ast.ClassDef, wrapped_class: WrappedClass
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
            self, node: ast.ClassDef, wrapped_class: WrappedClass, role_type: RoleType
    ) -> ast.ClassDef:
        """
        Transforms a role class by adjusting its bases and filtering fields.
        """
        taker_type = wrapped_class.clazz.get_role_taker_type()

        if role_type == RoleType.PRIMARY:
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

        # Find the TypeVar for the taker if it's a Role
        type_var_name = self._get_type_var_name(taker_type)
        if not type_var_name:
            # Fallback to T<TakerName>
            type_var_name = f"T{taker_name}"

        # Base classes: TakerMixin, Role[T]
        # Matching GT order: RoleForPerson(Role[TPerson], PersonMixin) but RoleForCEOAsFirstRole(CEOAsFirstRoleMixin, Role[TCEOAsFirstRole])
        # A simple rule: if Mixin inherits from Role, put Mixin first.
        # PersonMixin does not inherit from Role. CEOAsFirstRoleMixin does (via RoleForPerson).

        available_type_vars = {type_var_name}
        wrapped_taker = self.diagram.get_wrapped_class(taker_type)
        mixin_inherits_from_role = issubclass(taker_type, Role)

        mixin_base_class_name = f"{taker_name}Mixin"
        role_base_class_name = f"{Role.__name__}[{type_var_name}]"
        if mixin_inherits_from_role:
            bases = [mixin_base_class_name, role_base_class_name]
        else:
            bases = [role_base_class_name, mixin_base_class_name]

        body = []
        # 1. Add role taker attribute (e.g. person: TPerson = field(kw_only=True))
        # TODO: This should be added to the classes that inherit from RoleFor class, not in here, as the role taker
        # attribute name could be different for different primary roles.
        primary_role_wrapped = self._get_primary_roles(taker_type)[0]
        taker_attr_name = primary_role_wrapped.clazz.role_taker_attribute_name()
        # Use own_fields to find the role taker attribute
        wrapped_field = next(
            f for f in primary_role_wrapped.own_fields if f.name == taker_attr_name
        )
        body.append(
            self._create_field_node(
                wrapped_field,
                kw_only=True,
                available_type_vars=available_type_vars,
            )
        )

        # 2. Add fields from Taker as init=False
        # Only exclude fields from roles targeting THIS taker
        roles_targeting_taker_fields = set()
        roles = self._role_taker_to_roles_map.get(taker_type, [])
        for role_wrapped in roles:
            for f in role_wrapped.own_fields:
                roles_targeting_taker_fields.add(f.name)
        for field_ in wrapped_taker.fields:
            if field_.name in roles_targeting_taker_fields:
                continue
            if field_.name in self._base_role_class_field_names:
                continue
            body.append(
                self._create_field_node(
                    field_, init=False, available_type_vars=available_type_vars
                )
            )

        # 3. Add @classmethod role_taker_attribute
        method_node = ast.FunctionDef(
            name=Role.role_taker_attribute_name.__name__,
            args=self.make_ast_args("cls"),
            body=[self.make_ellipsis_expression()],
            decorator_list=[self.to_ast_name(classmethod)],
            returns=self.to_ast_name(taker_type),
            type_comment=None,
            type_params=[],
        )
        body.append(method_node)

        return ast.ClassDef(
            name=f"RoleFor{taker_name}",
            bases=list(map(self.to_ast_name, bases)),
            keywords=[],
            body=body if body else [self.make_ellipsis_expression()],
            decorator_list=[self._dataclass_decorator],
        )

    @staticmethod
    def make_class_method(name: str, args: Optional[List[str]] = None,
                          returns: Type | Callable | str = None) -> ast.FunctionDef:
        """
        :return: AST FunctionDef object for the given class method.
        """
        args = ["cls"] + (args or [])
        return ast.FunctionDef(
            name=name,
            args=StubTransformer.make_ast_args(*args),
            body=[StubTransformer.make_ellipsis_expression()],
            decorator_list=[StubTransformer.to_ast_name(classmethod)],
            returns=StubTransformer.to_ast_name(returns),
            type_comment=None,
            type_params=[],
        )

    @staticmethod
    def to_ast_name(has_name: Type | Callable | str) -> ast.Name:
        """
        :return: AST Name object for the given name.
        """
        name = has_name if isinstance(has_name, str) else has_name.__name__
        return ast.Name(id=name, ctx=ast.Load())

    @staticmethod
    def make_ast_args(*names) -> ast.arguments:
        """
        :return: AST arguments object for the given names.
        """
        return ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=n) for n in names],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        )

    @cached_property
    def _dataclass_decorator(self) -> ast.expr:
        return ast.Call(
            func=ast.Name(id="dataclass", ctx=ast.Load()),
            args=[],
            keywords=[ast.keyword(arg="eq", value=False)]
        )

    @staticmethod
    def make_ellipsis_expression() -> ast.Expr:
        return ast.Expr(value=ast.Constant(value=Ellipsis))

    @cached_property
    def _base_role_class_field_names(self):
        """
        :return: List of field names for the base Role class.
        """
        return [f.name for f in dataclasses.fields(Role)]

    @lru_cache
    def _get_type_var_name(self, clazz: Type) -> Optional[str]:
        """
        Finds a TypeVar bound to the given class.
        """
        for name, value in self.module.__dict__.items():
            if isinstance(value, TypeVar):
                if getattr(value, "__bound__", None) == clazz:
                    return name
        return None

    def __hash__(self):
        return hash((self.__class__, self.module))


@dataclasses.dataclass
class RoleStubGeneratorV2:
    """
    AST-based Role Stub Generator.
    """

    module: ModuleType
    path: Optional[Path] = None

    def __post_init__(self):
        if self.path is None:
            self.path = Path(self.module_file_path).with_suffix(".pyi")
        self._build_diagram()

    def _build_diagram(self):
        classes = classes_of_module(self.module)
        for clazz in classes:
            if issubclass(clazz, Role):
                role_taker_type = clazz.get_role_taker_type()
                if role_taker_type not in classes:
                    classes.append(role_taker_type)
        self.class_diagram = ClassDiagram(classes)

    def generate_stub(self, write: bool = False) -> str:
        """
        Generates the stub file content.
        """
        with open(self.module_file_path, "r") as f:
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

    @property
    def module_file_path(self) -> Path:
        """
        :return: Path to the module file.
        """
        return Path(sys.modules[self.module.__name__].__file__)
