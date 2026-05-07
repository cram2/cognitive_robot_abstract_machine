"""
Generator that produces delegation property and method nodes for a delegatee class.
"""

from __future__ import annotations

import dataclasses
import inspect
from textwrap import dedent
from typing import Any, Callable

import libcst

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.utils import (
    GenericTypeSubstitution,
    get_type_hints_of_object,
    resolve_name_in_hierarchy,
    same_package,
    get_property_return_type,
)
from krrood.patterns.code_generation.exceptions import CodeGenerationError
from krrood.patterns.code_generation.import_name_resolver import ImportNameResolver
from krrood.patterns.code_generation.libcst_node_factory import LibCSTNodeFactory
from krrood.patterns.code_generation.type_normaliser import TypeNormaliser


def _is_original_field_definer(klass: type, field_name: str) -> bool:
    """Return True if klass is where field_name is first introduced in its own MRO.

    Uses per-class ``__annotations__`` (not the cumulative ``__dataclass_fields__``)
    and confirms no ancestor of klass also declares the same annotation, which
    filters out annotations injected by generic-resolution machinery.

    :param klass: The class to test.
    :param field_name: The dataclass field name to look up.
    :return: True only when klass is the original introducer of the field.
    """
    if field_name not in vars(klass).get("__annotations__", {}):
        return False
    return not any(
        field_name in vars(ancestor).get("__annotations__", {})
        for ancestor in klass.__mro__[1:]
        if ancestor is not object
    )


def _find_defining_class(
    name: str,
    clazz: type,
    module_name: str,
    already_covered_bases: set[type],
    is_member: Callable[[type], bool],
    is_excluded_defining_class: Callable[[type], bool] | None = None,
) -> type | None:
    """Return the first class in clazz's MRO that defines name under the given membership test.

    Returns None when the name belongs directly to the class, to an excluded base,
    to a class rejected by the exclusion predicate, or to a class in a different package.

    :param name: The attribute name to find.
    :param clazz: The class whose MRO is walked.
    :param module_name: The source module name used for package comparison.
    :param already_covered_bases: The set of base types whose members are already delegated.
    :param is_member: Callable that returns True when a class defines the name.
    :param is_excluded_defining_class: Optional predicate; when it returns True for a class
        in the MRO, that class is skipped as a valid defining class.
    :return: The defining class, or None.
    """
    for klass in clazz.__mro__[1:]:
        if klass is object:
            return None
        if is_member(klass):
            if is_excluded_defining_class is not None and is_excluded_defining_class(
                klass
            ):
                return None
            if klass in already_covered_bases:
                return None
            if klass.__module__ == module_name or same_package(
                klass.__module__, module_name
            ):
                return klass
            return None
    return None


@dataclasses.dataclass
class DelegationGenerator:
    """
    Generates getter, setter, and method delegation nodes that expose a delegatee class's
    members via a named attribute on the delegating class.
    """

    delegatee_attribute_name: str
    node_factory: LibCSTNodeFactory
    type_normaliser: TypeNormaliser
    already_covered_bases: set[type] = dataclasses.field(default_factory=set)
    excluded_method_names: frozenset[str] = dataclasses.field(default_factory=frozenset)
    excluded_member_predicate: Callable[[str, type], bool] | None = None
    is_excluded_defining_class: Callable[[type], bool] | None = None
    name_resolver: ImportNameResolver | None = None

    def collect_delegation_groups(
        self,
        wrapped_class: WrappedClass,
        module_name: str,
        already_delegated_field_names: list[str] | None = None,
        additional_skip_bases: set[type] | None = None,
    ) -> dict[type | None, dict[str, list[libcst.FunctionDef]]]:
        """Return delegation nodes for a class grouped by the defining ancestor.

        :param wrapped_class: The class whose members are to be delegated.
        :param module_name: The source module name, used for package boundary checks.
        :param already_delegated_field_names: Field names already covered by a parent
            delegation mixin and therefore excluded from the output.
        :param additional_skip_bases: Extra base classes whose methods should not be
            delegated even if they are not in ``already_covered_bases``.
        :return: Mapping of defining class (None = the class itself) to member name to nodes.
        """
        taker_fields = already_delegated_field_names or []
        skip_bases = self.already_covered_bases | (additional_skip_bases or set())
        groups: dict[type | None, dict[str, list]] = {}
        self._collect_field_delegations(
            wrapped_class, taker_fields, module_name, groups
        )
        self._collect_property_delegations(wrapped_class, module_name, groups)
        self._collect_method_delegations(wrapped_class, module_name, groups, skip_bases)
        return groups

    def _collect_field_delegations(
        self,
        wrapped_class: WrappedClass,
        taker_fields: list[str],
        module_name: str,
        groups: dict[type | None, dict[str, list]],
    ) -> None:
        """Populate groups with getter/setter delegation nodes for each dataclass field.

        :param wrapped_class: The class whose fields are delegated.
        :param taker_fields: Field names already covered by a parent delegation class.
        :param module_name: The source module name for package comparison.
        :param groups: The accumulator dict to populate.
        """
        for field_ in wrapped_class.fields:
            if field_.name in taker_fields:
                continue
            if not (field_.field.kw_only or field_.field.init):
                continue
            defining_base = _find_defining_class(
                field_.name,
                wrapped_class.clazz,
                module_name,
                self.already_covered_bases,
                lambda klass: _is_original_field_definer(klass, field_.name),
                self.is_excluded_defining_class,
            )
            if defining_base is not None:
                self._delegate_inherited_field(
                    field_.name,
                    field_.field,
                    wrapped_class.clazz,
                    defining_base,
                    groups,
                    module_name,
                )
            else:
                field_type_name = self._normalise_type_name(field_.field.type)
                prop_nodes = self.node_factory.make_property_getter_and_setter_nodes(
                    field_.name,
                    field_type_name,
                    f"self.{self.delegatee_attribute_name}.{field_.name}",
                    f"self.{self.delegatee_attribute_name}.{field_.name} = value",
                )
                groups.setdefault(None, {})[field_.name] = prop_nodes

    def _delegate_inherited_field(
        self,
        field_name: str,
        field: Any,
        concrete_class: type,
        defining_base: type,
        groups: dict[type | None, dict[str, list]],
        module_name: str,
    ) -> None:
        """Place delegation nodes for a field inherited from a generic base.

        Adds a re-declaration in the concrete class's own body when the base TypeVar
        is substituted. Also adds re-declarations in any same-package non-covered
        intermediate ancestor that narrows the TypeVar.

        :param field_name: The dataclass field name.
        :param field: The dataclass Field object.
        :param concrete_class: The class being processed.
        :param defining_base: The ancestor class that originally defines the field.
        :param groups: The accumulator dict to populate.
        :param module_name: The source module name for package comparison.
        """
        try:
            base_hints = get_type_hints_of_object(defining_base)
            base_type = base_hints.get(field_name, field.type)
        except TypeError:
            base_type = field.type

        base_type_name = self._normalise_type_name(base_type)
        prop_nodes = self.node_factory.make_property_getter_and_setter_nodes(
            field_name,
            base_type_name,
            f"self.{self.delegatee_attribute_name}.{field_name}",
            f"self.{self.delegatee_attribute_name}.{field_name} = value",
        )
        groups.setdefault(defining_base, {})[field_name] = prop_nodes

        last_narrowed_type: Any = None
        for ancestor in concrete_class.__mro__[1:]:
            if ancestor is defining_base:
                break
            if ancestor in self.already_covered_bases:
                continue
            if ancestor.__module__ != module_name and not same_package(
                ancestor.__module__, module_name
            ):
                continue
            narrowed = self._add_narrowing_redeclaration(
                field_name, base_type, ancestor, defining_base, groups
            )
            if narrowed is not None:
                last_narrowed_type = narrowed

        substitution = GenericTypeSubstitution.from_specialization(
            concrete_class, defining_base
        )
        if substitution.has_substitutions:
            result = substitution.apply(base_type)
            if result.resolved and result.resolved_type is not last_narrowed_type:
                concrete_type_name = self._normalise_type_name(result.resolved_type)
                redecl_nodes = self.node_factory.make_property_getter_and_setter_nodes(
                    field_name,
                    concrete_type_name,
                    f"self.{self.delegatee_attribute_name}.{field_name}",
                    f"self.{self.delegatee_attribute_name}.{field_name} = value",
                )
                groups.setdefault(None, {})[field_name] = redecl_nodes

    def _add_narrowing_redeclaration(
        self,
        field_name: str,
        base_type: Any,
        ancestor: type,
        defining_base: type,
        groups: dict[type | None, dict[str, list]],
    ) -> Any | None:
        """Add a narrowing re-declaration to groups[ancestor] if ancestor substitutes the TypeVar.

        :param field_name: The dataclass field name.
        :param base_type: The type annotation from defining_base.
        :param ancestor: The intermediate ancestor class to check.
        :param defining_base: The class that originally defines the field.
        :param groups: The accumulator dict to populate.
        :return: The resolved type object if a narrowing was added, None otherwise.
        """
        substitution = GenericTypeSubstitution.from_specialization(
            ancestor, defining_base
        )
        if not substitution.has_substitutions:
            return None
        result = substitution.apply(base_type)
        if not result.resolved:
            return None
        type_name = self._normalise_type_name(result.resolved_type)
        nodes = self.node_factory.make_property_getter_and_setter_nodes(
            field_name,
            type_name,
            f"self.{self.delegatee_attribute_name}.{field_name}",
            f"self.{self.delegatee_attribute_name}.{field_name} = value",
        )
        groups.setdefault(ancestor, {}).setdefault(field_name, nodes)
        return result.resolved_type

    def _collect_property_delegations(
        self,
        wrapped_class: WrappedClass,
        module_name: str,
        groups: dict[type | None, dict[str, list]],
    ) -> None:
        """Populate groups with getter/setter delegation nodes for each data descriptor.

        :param wrapped_class: The class whose properties are delegated.
        :param module_name: The source module name for package comparison.
        :param groups: The accumulator dict to populate.
        """
        for property_name, property_value in inspect.getmembers(
            wrapped_class.clazz, inspect.isdatadescriptor
        ):
            if not isinstance(property_value, property):
                continue
            if (
                self.excluded_member_predicate is not None
                and self.excluded_member_predicate(property_name, wrapped_class.clazz)
            ):
                continue

            defining_base = _find_defining_class(
                property_name,
                wrapped_class.clazz,
                module_name,
                self.already_covered_bases,
                lambda klass: property_name in vars(klass),
                self.is_excluded_defining_class,
            )

            has_setter = property_value.fset is not None
            base_return_type = get_property_return_type(property_value)
            return_annotation = (
                self._normalise_type_name(base_return_type)
                if base_return_type
                else None
            )

            prop_nodes = self._make_property_delegation_nodes(
                property_name, return_annotation, has_setter
            )
            groups.setdefault(defining_base, {})[property_name] = prop_nodes

            if defining_base is not None and base_return_type is not None:
                substitution = GenericTypeSubstitution.from_specialization(
                    wrapped_class.clazz, defining_base
                )
                if substitution.has_substitutions:
                    result = substitution.apply(base_return_type)
                    if result.resolved:
                        concrete_annotation = self._normalise_type_name(
                            result.resolved_type
                        )
                        redeclared_nodes = self._make_property_delegation_nodes(
                            property_name, concrete_annotation, has_setter
                        )
                        groups.setdefault(None, {})[property_name] = redeclared_nodes

    def _make_property_delegation_nodes(
        self,
        property_name: str,
        return_annotation: str | None,
        has_setter: bool,
    ) -> list:
        """Build getter (and optionally setter) delegation nodes for a property.

        :param property_name: The property name.
        :param return_annotation: The normalised return type string, or None.
        :param has_setter: Whether to also generate a setter node.
        :return: A list of FunctionDef nodes.
        """
        if has_setter:
            return self.node_factory.make_property_getter_and_setter_nodes(
                property_name,
                return_annotation,
                f"self.{self.delegatee_attribute_name}.{property_name}",
                f"self.{self.delegatee_attribute_name}.{property_name} = value",
            )
        return [
            self.node_factory.make_property_getter_node(
                property_name,
                return_annotation,
                f"self.{self.delegatee_attribute_name}.{property_name}",
            )
        ]

    def _collect_method_delegations(
        self,
        wrapped_class: WrappedClass,
        module_name: str,
        groups: dict[type | None, dict[str, list]],
        skip_bases: set[type],
    ) -> None:
        """Populate groups with delegation nodes for each delegatable method.

        :param wrapped_class: The class whose methods are delegated.
        :param module_name: The source module name for package comparison.
        :param groups: The accumulator dict to populate.
        :param skip_bases: Combined set of bases whose methods should not be delegated.
        """
        skip_base_list = [
            base for base in wrapped_class.clazz.__mro__[1:] if base in skip_bases
        ]

        for method_name, method_object in inspect.getmembers(
            wrapped_class.clazz, predicate=inspect.isfunction
        ):
            if method_name in self.excluded_method_names:
                continue
            if (
                self.excluded_member_predicate is not None
                and self.excluded_member_predicate(method_name, wrapped_class.clazz)
            ):
                continue
            if any(method_name in dir(base) for base in skip_base_list):
                continue
            method_node = self.make_delegation_method_node(
                method_name, method_object, wrapped_class.clazz
            )
            if method_node is not None:
                defining_base = _find_defining_class(
                    method_name,
                    wrapped_class.clazz,
                    module_name,
                    self.already_covered_bases,
                    lambda klass: method_name in vars(klass),
                    self.is_excluded_defining_class,
                )
                groups.setdefault(defining_base, {})[method_name] = [method_node]

    def make_delegation_method_node(
        self,
        name: str,
        method: Callable,
        source_class: type,
    ) -> libcst.FunctionDef | None:
        """Create a method node that delegates to the corresponding method on the delegatee.

        :param name: The method name.
        :param method: The live method object.
        :param source_class: The class the method belongs to.
        :return: A libcst FunctionDef node, or None if the source is unavailable.
        """
        try:
            method_source = inspect.getsource(method)
        except OSError:
            return None

        method_node = libcst.parse_module(dedent(method_source)).body[0]
        if not isinstance(method_node, libcst.FunctionDef):
            raise CodeGenerationError(
                f"Expected FunctionDef, got {type(method_node).__name__}"
            )

        self._resolve_signature_types(method)
        self._register_decorator_imports(method_node, method)

        return self._generate_delegation_body(method_node, name, method, source_class)

    def _resolve_signature_types(self, method: Callable) -> None:
        """Register name-to-module mappings for all types in a method's signature.

        :param method: The method whose type annotations should be registered.
        """
        try:
            self._register_type_hints_from_method(method)
        except Exception:
            self._register_signature_annotations(method)
        if self.name_resolver is not None:
            self.name_resolver.register_from_callable_globals(method)

    def _register_type_hints_from_method(self, method: Callable) -> None:
        """Register types via get_type_hints (preferred path)."""
        type_hints = get_type_hints_of_object(method)
        for type_obj in type_hints.values():
            self._normalise_type_name(type_obj)

    def _register_signature_annotations(self, method: Callable) -> None:
        """Register types via inspect.signature (fallback path)."""
        sig = inspect.signature(method)
        for param in sig.parameters.values():
            if param.annotation is not inspect.Parameter.empty:
                self._normalise_type_name(param.annotation)
        if sig.return_annotation is not inspect.Signature.empty:
            self._normalise_type_name(sig.return_annotation)

    def _register_decorator_imports(
        self, method_node: libcst.FunctionDef, method: Callable
    ) -> None:
        """Register the source module of each decorator used on the method.

        :param method_node: The parsed FunctionDef node containing decorator nodes.
        :param method: The live method object, used for runtime name resolution.
        """
        if self.name_resolver is None:
            return
        for decorator in method_node.decorators:
            decorator_name = self.node_factory._get_decorator_name(decorator.decorator)
            if decorator_name:
                try:
                    decorator_object = resolve_name_in_hierarchy(decorator_name, method)
                    if hasattr(decorator_object, "__module__"):
                        self.name_resolver.name_to_module_map[decorator_name] = (
                            decorator_object.__module__
                        )
                except Exception:
                    pass

    def _generate_delegation_body(
        self,
        method_node: libcst.FunctionDef,
        name: str,
        method: Callable,
        source_class: type,
    ) -> libcst.FunctionDef:
        """Replace the method body with a delegation call to the delegatee attribute.

        :param method_node: The original parsed method node.
        :param name: The method name to delegate to on the delegatee.
        :param method: The live method object used to obtain parameter names.
        :param source_class: The class to which the method belongs.
        :return: The method node with a delegation body.
        """
        parameters = inspect.signature(method).parameters
        call_params = [p for p in parameters.keys() if p != "self"]
        import_statement = []
        if "self" in parameters.keys():
            attribute_source = f"self.{self.delegatee_attribute_name}"
        else:
            attribute_source = source_class.__name__
            import_statement = [
                libcst.parse_statement(
                    f"from {source_class.__module__} import {source_class.__name__}"
                )
            ]
        return method_node.with_changes(
            body=libcst.IndentedBlock(
                import_statement
                + [
                    libcst.parse_statement(
                        f"return {attribute_source}.{name}({', '.join(call_params)})"
                    )
                ]
            )
        )

    def _normalise_type_name(self, type_obj: Any) -> str:
        """Return a normalised string representation of a type for use in generated code.

        :param type_obj: The type object to normalise.
        :return: A string type name suitable for inclusion in generated source code.
        """
        return self.type_normaliser.normalise(type_obj)
