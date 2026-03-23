from __future__ import annotations

import dataclasses
import sys
from copy import copy
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import (
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Callable,
)

import libcst
from black.handle_ipynb_magics import lru_cache

from krrood.class_diagrams import ClassDiagram
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.exceptions import ClassIsUnMappedInClassDiagram
from krrood.class_diagrams.utils import classes_of_module
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.patterns.role.meta_data import RoleType
from krrood.patterns.role.role import Role
from krrood.patterns.role.role_stub_generator import (
    FieldRepresentation,
)
from krrood.utils import (
    run_black_on_file,
    run_ruff_on_file,
    get_generic_type_param,
)


class StubTransformer(libcst.CSTTransformer):
    """
    Transforms a Python module AST into a stub file AST by pruning methods
    and applying the Role pattern transformations.
    """

    def __init__(self, diagram: ClassDiagram, module: ModuleType):
        super().__init__()
        self.diagram = diagram
        self.module = module
        self.needed_role_for_classes: Set[Type] = set()

    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.FunctionDef:
        """
        Prunes function bodies, replacing them with '...'.
        """
        return updated_node.with_changes(
            body=libcst.IndentedBlock(body=[self.make_ellipsis_expression()])
        )

    @lru_cache
    def _has_primary_role(self, taker_type: Type) -> bool:
        """
        Checks if there is at least one primary role targeting this taker
        that does not update its taker type.
        """
        roles = self.diagram.get_roles_of_class(taker_type)
        return any(
            RoleType.get_role_type(role_wrapped) == RoleType.PRIMARY
            for role_wrapped in roles
        )

    def leave_SimpleStatementLine(
        self,
        original_node: libcst.SimpleStatementLine,
        updated_node: libcst.SimpleStatementLine,
    ) -> (
        libcst.SimpleStatementLine | libcst.FlattenSentinel[libcst.SimpleStatementLine]
    ):
        """
        Detects TypeVar definitions and inserts RoleFor classes after them if they are for a taker.
        """
        if len(updated_node.body) != 1:
            return updated_node
        node = updated_node.body[0]
        if not (
            isinstance(node, libcst.Assign)
            and isinstance(node.value, libcst.Call)
            and isinstance(node.value.func, libcst.Name)
            and node.value.func.value == TypeVar.__name__
        ):
            return updated_node

        bound_value = self.get_keyword_value_from_call(node.value, "bound")
        bound_name = self.unparse_type_value(bound_value) if bound_value else None

        if bound_name and bound_name in self.module.__dict__:
            clazz = self.module.__dict__[bound_name]
            if clazz in self.diagram.role_takers and self._has_primary_role(clazz):
                # role_for_class = self._synthesize_role_for(clazz)
                # return libcst.FlattenSentinel([updated_node, role_for_class])
                return updated_node
        return updated_node

    @classmethod
    def unparse_type_value(cls, value: libcst.BaseExpression) -> Optional[str]:
        """
        Unparses a libcst expression to get the type name as a string.
        """
        if isinstance(value, libcst.Name):
            return value.value
        elif isinstance(value, libcst.SimpleString):
            return value.evaluated_value
        else:
            raise ValueError(f"Unsupported type value: {value}")

    @classmethod
    def get_keyword_value_from_call(
        cls, call: libcst.Call, keyword: str
    ) -> Optional[libcst.BaseExpression]:
        for kw in call.args:
            if kw.keyword and kw.keyword.value == keyword:
                return kw.value
        return None

    def leave_ClassDef(
        self, original_node: libcst.ClassDef, updated_node: libcst.ClassDef
    ) -> libcst.ClassDef | libcst.FlattenSentinel[libcst.BaseCompoundStatement]:
        """
        Transforms class definitions: prunes methods, renames takers, and adjusts roles.
        """
        if updated_node.name.value not in self.module.__dict__:
            return updated_node
        clazz = self.module.__dict__[updated_node.name.value]
        try:
            wrapped_class = self.diagram.get_wrapped_class(clazz)
        except ClassIsUnMappedInClassDiagram:
            return updated_node

        # Prune methods and non-essential nodes
        new_body_list = [item.visit(self) for item in updated_node.body.body]
        updated_node = updated_node.with_changes(
            body=updated_node.body.with_changes(body=new_body_list)
        )

        is_taker = wrapped_class.clazz in self.diagram.role_takers

        if is_taker:
            # transform_role_taker returns [Mixin, original_class]
            taker_nodes = self._transform_role_taker(updated_node, wrapped_class)
            result_nodes = [taker_nodes[0], taker_nodes[1]]
            return libcst.FlattenSentinel(result_nodes)
        #     # If no TypeVar is found later, we might miss RoleFor.
        #     # But the ground truth suggests RoleFor is after TypeVar if it exists.
        #     # If no TypeVar exists for this taker, we should probably add it here.
        #     if not self._has_type_var_for_taker(wrapped_class.clazz):
        #         role_for_class = self._synthesize_role_for(wrapped_class.clazz)
        #         result_nodes.append(role_for_class)
        else:
            result_nodes = [updated_node]

        role_type = RoleType.get_role_type(wrapped_class)

        match role_type:
            case RoleType.NOT_A_ROLE:
                ...
            # case RoleType.SPECIALIZED_ROLE_FOR:
            #     # transform_specialized_role returns [specialized_class, node]
            #     result_nodes = self._transform_specialized_role(
            #         updated_node, wrapped_class
            #     )
            case _:
                updated_node = self._transform_role(
                    updated_node, wrapped_class, role_type
                )
                result_nodes = [updated_node]

        if len(result_nodes) > 1:
            return libcst.FlattenSentinel(result_nodes)
        return result_nodes[0]

    def _has_type_var_for_taker(self, taker_type: Type) -> bool:
        """
        Checks if a TypeVar bound to this taker exists in the module.
        """
        return self._get_type_var_name(taker_type) is not None

    def _transform_specialized_role(
        self, node: libcst.ClassDef, wrapped_class: WrappedClass[Role]
    ) -> List[libcst.ClassDef]:
        """
        Handles a role that updates its taker type by synthesizing a specialized (due to role taker type update)
         RoleFor base.

        :param node: The original class node.
        :param wrapped_class: The wrapped class for the role.
        """
        taker_type = wrapped_class.clazz.get_role_taker_type()
        base_role = next(
            clazz for clazz in wrapped_class.clazz.__bases__ if issubclass(clazz, Role)
        )
        specialized_name = f"{base_role.__name__}AsRoleFor{taker_type.__name__}"

        # TSubclassOfARoleTaker
        type_var_name = self._get_type_var_name(taker_type) or f"T{taker_type.__name__}"
        available_type_vars = {type_var_name}

        # Specialized base: CEOAsFirstRoleAsRoleForSubclassOfARoleTaker(CEOAsFirstRole[TSubclassOfARoleTaker], SubclassOfARoleTakerMixin)
        specialized_bases = [
            f"{taker_type.__name__}Mixin",
            f"{base_role.__name__}[{type_var_name}]",
        ]

        # Add fields from taker as init=False
        wrapped_taker = self.diagram.get_wrapped_class(taker_type)
        body = [
            self._create_field_node(
                field_, init=False, available_type_vars=available_type_vars
            )
            for field_ in wrapped_taker.fields
            if field_.field.init or field_.field.kw_only
        ]

        specialized_class = self.make_dataclass(
            name=specialized_name,
            bases=specialized_bases,
            body=body,
        )

        # Current class inherits from specialized base
        node = node.with_changes(
            bases=[self.make_argument(specialized_name)],
            body=self.make_ellipsis_body(),
        )

        return [specialized_class, node]

    def _transform_role_taker(
        self, node: libcst.ClassDef, wrapped_class: WrappedClass
    ) -> List[libcst.ClassDef]:
        """
        Transforms a role taker class into a Mixin and a re-entry class.
        """
        original_name = node.name.value
        mixin_name = f"{original_name}Mixin"
        mixin_node = self.get_renamed_node(node, mixin_name)

        # Reconstruct body: own fields + propagated fields (init=False)
        mixin_body = []
        bases_that_are_takers = {
            b.__name__: b
            for b in wrapped_class.clazz.__bases__
            if b in self.diagram.role_takers
        }
        all_taker_fields = []
        for base_name, taker_type in bases_that_are_takers.items():
            wrapped_taker = self.diagram.get_wrapped_class(taker_type)
            all_taker_fields.extend([f.name for f in wrapped_taker.fields])
        for field_ in wrapped_class.fields:
            if field_.name in all_taker_fields:
                continue
            if field_.field.kw_only or field_.field.init:
                mixin_body.append(self._create_field_node(field_, init=False))

        propagated_fields = self._get_propagated_fields(wrapped_class)
        mixin_body.extend(propagated_fields)

        new_bases = []
        for base in node.bases:
            base_name = self.get_name_from_base_node(base.value)
            if base_name in bases_that_are_takers:
                new_bases.append(
                    self.make_argument(
                        self._get_mixin_name(
                            bases_that_are_takers[base_name], wrapped_class
                        )
                    )
                )
                continue
            if self._is_role_base(base.value) and issubclass(wrapped_class.clazz, Role):
                taker_type = wrapped_class.clazz.get_role_taker_type()
                new_bases.append(
                    self.make_argument(
                        self._get_mixin_name(taker_type, wrapped_class, False)
                    )
                )
            new_bases.append(base)

        mixin_node = self.get_node_with_new_body(mixin_node, mixin_body)
        mixin_node = mixin_node.with_changes(bases=new_bases)

        # Create the original class inheriting from Mixin
        reentry_class = libcst.ClassDef(
            name=libcst.Name(original_name),
            bases=[
                self.make_argument(
                    self._get_mixin_name(wrapped_class.clazz, wrapped_class)
                )
            ],
            body=node.body,
            decorators=node.decorators,
        )

        return [mixin_node, reentry_class]

    @classmethod
    def get_node_with_new_body(
        cls, node: libcst.ClassDef, new_body: List[libcst.BaseStatement]
    ) -> libcst.ClassDef:
        """
        :param node: The node to update.
        :param new_body: The new body for the node.
        :return: A new node that is a copy of the original node but with the new body.
        """
        return node.with_changes(body=node.body.with_changes(body=new_body))

    @classmethod
    def get_renamed_node(cls, node, new_name):
        """
        :param node: The node to rename.
        :param new_name: The new name for the node.
        :return: A new node that is a copy of the original node but with the new name.
        """
        return node.with_changes(name=libcst.Name(new_name))

    @classmethod
    def make_argument(cls, value: str) -> libcst.Arg:
        """
        :param value: The value of the argument.
        :return: libcst Arg object with the given value.
        """
        return libcst.Arg(value=libcst.parse_expression(value))

    @classmethod
    def _is_role_base(cls, base_node: libcst.BaseExpression) -> bool:
        """
        Checks if a base node is the 'Role' class or 'Role[T]'.

        :param base_node: The base node to check.
        :return: True if the base node is 'Role' or 'Role[T]', False otherwise.
        """
        name = cls.get_name_from_base_node(base_node)
        return name == "Role"

    @classmethod
    def get_name_from_base_node(cls, base_node: libcst.BaseExpression) -> str:
        """
        Extracts the class name from a base node, handling both simple names and sub-scripted types.

        :param base_node: The base node to extract the class name from.
        :return: The class name as a string.
        """
        if isinstance(base_node, libcst.Name):
            return base_node.value
        if isinstance(base_node, libcst.Subscript):
            if isinstance(base_node.value, libcst.Name):
                return base_node.value.value
        raise ValueError(f"Unexpected base node type: {base_node}")

    def _transform_role(
        self, node: libcst.ClassDef, wrapped_class: WrappedClass, role_type: RoleType
    ) -> libcst.ClassDef:
        """
        Transforms a role class by adjusting its bases and filtering fields.
        """

        if role_type in [RoleType.PRIMARY, RoleType.SPECIALIZED_ROLE_FOR]:
            node = self._transform_primary_role(node, wrapped_class)

        # Filter fields: keep only introduced fields
        new_body = []
        taker_attr_name = wrapped_class.clazz.role_taker_attribute_name()
        kept_field_names = [f.name for f in wrapped_class.own_fields] + [
            taker_attr_name
        ]
        for item in node.body.body:
            field_name = self._get_field_name_if_statement_is_field_definition(item)
            if field_name and field_name not in kept_field_names:
                continue
            new_body.append(item)

        updated_node = node.with_changes(body=node.body.with_changes(body=new_body))

        return updated_node

    @classmethod
    def _get_field_name_if_statement_is_field_definition(
        cls, item: libcst.BaseStatement
    ) -> Optional[str]:
        """
        :param item: The statement to check.
        :return: The field name if the statement is a field definition, otherwise None.
        """
        if (
            isinstance(item, libcst.SimpleStatementLine)
            and len(item.body) == 1
            and isinstance(ann_assign := item.body[0], libcst.AnnAssign)
            and isinstance(field_name := ann_assign.target, libcst.Name)
        ):
            return field_name.value
        return None

    def _transform_primary_role(
        self, node: libcst.ClassDef, wrapped_class: WrappedClass
    ) -> libcst.ClassDef:
        """
        Transforms a primary role class by adjusting its bases and filtering fields.

        :param node: The original class node.
        :param wrapped_class: The wrapped class information.
        """
        taker_type = wrapped_class.clazz.get_role_taker_type()

        # Filter out original Role bases and redundant bases
        new_bases = []

        taker_mro_names = {c.__name__ for c in taker_type.mro()}

        for base in node.bases:

            # If it's a simple name and in taker's MRO, it's covered by RoleFor<Taker>
            if (
                isinstance(base.value, libcst.Name)
                and base.value.value in taker_mro_names
            ):
                continue

            if self._is_role_base(base.value):
                new_bases.append(
                    self.make_argument(
                        self._get_mixin_name(taker_type, wrapped_class, False)
                    )
                )
            new_bases.append(base)

        return node.with_changes(bases=new_bases)

    @classmethod
    def _get_mixin_name(
        cls,
        taker_type: Type,
        wrapped_class: WrappedClass[Role],
        add_generic: bool = True,
    ) -> str:
        """
        Generates a name for the mixin class with the given taker type and wrapped class.

        :param taker_type: The type of the taker.
        :param wrapped_class: The wrapped class of the original role class.
        :param add_generic: Whether to include generic type parameters in the mixin name.
        """
        mixin_name = f"{taker_type.__name__}Mixin"

        if not issubclass(taker_type, Role) or not add_generic:
            return mixin_name

        # Handle generics
        generic_params = get_generic_type_param(wrapped_class.clazz, Role)
        if generic_params:
            type_vars = [
                arg.__name__ for arg in generic_params if isinstance(arg, TypeVar)
            ]
            if type_vars:
                mixin_name = f"{mixin_name}[{type_vars[0]}]"
        return mixin_name

    def _get_propagated_fields(
        self, wrapped_class: WrappedClass
    ) -> List[libcst.BaseStatement]:
        """
        Get libcst nodes for fields propagated from roles to the root role taker.

        :param wrapped_class: The wrapped class for the role.
        :return: A list of libcst nodes representing the fields to be propagated.
        """
        # Only propagate to root takers
        root_taker = wrapped_class.clazz
        if issubclass(root_taker, Role):
            root_taker = root_taker.get_root_role_taker_type()

        if wrapped_class.clazz != root_taker:
            return []

        # Find all roles (recursive) for this taker
        roles = self._get_all_roles_for_taker(root_taker)

        # Exclude taker fields from roles
        possible_fields_to_propagate = []
        for role_wrapped in roles:
            taker_attr_name = role_wrapped.clazz.role_taker_attribute_name()
            possible_fields_to_propagate.extend(
                [
                    role_field
                    for role_field in role_wrapped.fields
                    if role_field.name != taker_attr_name
                ]
            )

        # Add unseen fields from roles to the root taker
        fields_to_propagate = []
        seen_field_names = {f.name for f in wrapped_class.fields}
        for role_field in possible_fields_to_propagate:
            if role_field.name not in seen_field_names:
                fields_to_propagate.append(
                    self._create_field_node(role_field, init=False)
                )
                seen_field_names.add(role_field.name)

        return fields_to_propagate

    def _get_all_roles_for_taker(self, taker_type: Type) -> List[WrappedClass]:
        """
        Recursively finds all roles for a taker.
        """
        roles = []
        direct_roles = self.diagram.get_roles_of_class(taker_type)
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
    ) -> libcst.SimpleStatementLine:
        """
        Creates a libcst SimpleStatementLine node for a field.
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
                value_cst = libcst.parse_expression(val_str)
            except Exception:
                value_cst = libcst.Name("None")
        else:
            value_cst = None

        type_str = wrapped_field.type_name
        if available_type_vars is not None:
            type_str = self._resolve_type_vars(type_str, available_type_vars)

        type_str = type_str.replace("typing.", "").replace("typing_extensions.", "")

        return libcst.SimpleStatementLine(
            body=[
                libcst.AnnAssign(
                    target=libcst.Name(wrapped_field.name),
                    annotation=libcst.Annotation(
                        annotation=self.to_cst_expression(type_str)
                    ),
                    value=value_cst,
                )
            ]
        )

    def _synthesize_role_for(self, taker_type: Type) -> libcst.ClassDef:
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
        available_type_vars = {type_var_name}
        wrapped_taker = self.diagram.get_wrapped_class(taker_type)
        mixin_base_class_name = f"{taker_name}Mixin"
        role_base_class_name = f"{Role.__name__}[{type_var_name}]"
        bases = [mixin_base_class_name, role_base_class_name]

        body = []
        # 1. Add fields from Taker as init=False
        # Only exclude fields from roles targeting THIS taker
        roles_targeting_taker_fields = set()
        roles = self.diagram.get_roles_of_class(taker_type)
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

        # 2. Add @classmethod role_taker_attribute
        method_node = self.make_class_method(
            Role.role_taker_attribute.__name__, returns=taker_type
        )
        body.append(method_node)

        return self.make_dataclass(f"RoleFor{taker_name}", bases, body)

    @classmethod
    def make_dataclass(
        cls,
        name: str,
        bases: Optional[List[Type | str]] = None,
        body: Optional[List[libcst.BaseStatement]] = None,
    ) -> libcst.ClassDef:
        """
        :param name: Name of the dataclass.
        :param bases: Base classes of the dataclass.
        :param body: Body of the dataclass.
        :return: libcst ClassDef object for the given dataclass.
        """
        return libcst.ClassDef(
            name=libcst.Name(name),
            bases=[libcst.Arg(value=cls.to_cst_expression(b)) for b in (bases or [])],
            body=libcst.IndentedBlock(
                body=body if body else [cls.make_ellipsis_expression()]
            ),
            decorators=[cls.make_dataclass_decorator()],
        )

    @classmethod
    def make_class_method(
        cls,
        name: str,
        args: Optional[List[str]] = None,
        returns: Type | Callable | str = None,
    ) -> libcst.FunctionDef:
        """
        :return: libcst FunctionDef object for the given class method.
        """
        params = ["cls"] + (args or [])
        return libcst.FunctionDef(
            name=libcst.Name(name),
            params=cls.make_cst_args(*params),
            body=cls.make_ellipsis_body(),
            decorators=[libcst.Decorator(decorator=libcst.Name("classmethod"))],
            returns=(
                libcst.Annotation(annotation=cls.to_cst_expression(returns))
                if returns
                else None
            ),
        )

    @classmethod
    def make_ellipsis_body(cls) -> libcst.IndentedBlock:
        """
        :return: libcst IndentedBlock object with a single ellipsis expression.
        """
        return libcst.IndentedBlock(body=[cls.make_ellipsis_expression()])

    @classmethod
    def to_cst_expression(
        cls, has_name: Type | Callable | str
    ) -> libcst.BaseExpression:
        """
        :param has_name: An object that has a `__name__` attribute, one of class, method, or function.
        :return: libcst Expression object for the given name.
        """
        if isinstance(has_name, str):
            try:
                return libcst.parse_expression(has_name)
            except Exception:
                return libcst.Name(has_name)
        name = has_name.__name__
        return libcst.Name(name)

    @classmethod
    def make_cst_args(cls, *names) -> libcst.Parameters:
        """
        :return: libcst Parameters object for the given names.
        """
        return libcst.Parameters(
            params=[libcst.Param(name=libcst.Name(n)) for n in names]
        )

    @classmethod
    def make_dataclass_decorator(cls) -> libcst.Decorator:
        return libcst.Decorator(
            decorator=libcst.parse_expression("dataclass(eq=False)")
        )

    @classmethod
    def make_ellipsis_expression(cls) -> libcst.SimpleStatementLine:
        return libcst.SimpleStatementLine(body=[libcst.Expr(value=libcst.Ellipsis())])

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

        tree = libcst.parse_module(source)

        transformer = StubTransformer(self.class_diagram, self.module)
        transformed_tree = tree.visit(transformer)

        stub_content = transformed_tree.code

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
