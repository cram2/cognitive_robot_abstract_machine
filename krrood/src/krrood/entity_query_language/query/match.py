"""
Pattern-matching helpers for the Entity Query Language.

This module provides high-level match abstractions that build symbolic expressions for variables and attributes
from concise, readable matching syntax.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from inspect import ismethod, isfunction, isclass
from typing import assert_never, Any

import rustworkx as rx
from typing_extensions import (
    Optional,
    Type,
    List,
    Union,
    Generic,
    TYPE_CHECKING,
    Self,
    Iterator,
    get_type_hints,
)

from krrood.entity_query_language.core.base_expressions import (
    Selectable,
    SymbolicExpression,
)
from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    FlatVariable,
    CanBehaveLikeAVariable,
    MappedVariable,
    Index,
)
from krrood.entity_query_language.core.variable import Literal, DomainType, Variable
from krrood.entity_query_language.exceptions import (
    NoKwargsInMatchVar,
    CalledMatchMultipleTimes,
)
from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.utils import T
from krrood.patterns.factory_and_kwargs import HasFactoryAndKwargs
from krrood.rustworkx_utils import RWXNode
from krrood.symbol_graph.helpers import get_field_type_endpoint

if TYPE_CHECKING:
    from krrood.entity_query_language.factories import ConditionType
    from krrood.entity_query_language.query.query import Entity, Query

from typing import get_type_hints


@dataclass
class AbstractMatchExpression(Generic[T], ABC):
    """
    Abstract base class for constructing and handling a match expression.

    This class is intended to provide a framework for defining and managing match expressions,
    which are used to structural pattern matching in the form of nested match expressions with keyword arguments.
    """

    type_: Optional[Type[T]] = field(default=None, kw_only=True)
    """
    The type of the variable.
    """
    variable: Optional[Variable[T]] = field(default=None, kw_only=True)
    """
    The created variable from the type and kwargs.
    """
    conditions: List[ConditionType] = field(init=False, default_factory=list)
    """
    The conditions that define the match.
    """
    parent: Optional[AbstractMatchExpression] = field(init=False, default=None)
    """
    The parent match if this is a nested match.
    """
    resolved: bool = field(init=False, default=False)
    """
    Whether the match is resolved or not.
    """
    id: uuid.UUID = field(init=False, default_factory=uuid.uuid4)
    """
    The unique identifier of the match expression.
    """
    children: List[AttributeMatch] = field(init=False, default_factory=list)
    """
    The child matches of this match expression.
    """

    @cached_property
    @abstractmethod
    def expression(self) -> Union[CanBehaveLikeAVariable[T], T]:
        """
        :return: the entity expression corresponding to the match query.
        """
        ...

    def resolve(self, *args, **kwargs) -> Self:
        """
        Resolve the match by creating the variable and conditions expressions.
        """
        if self.resolved:
            return self
        self._resolve(*args, **kwargs)
        self.resolved = True
        return self

    @abstractmethod
    def _resolve(self, *args, **kwargs):
        """
        This method serves as an abstract definition to be implemented by subclasses,
        aimed at handling specific resolution logic for the derived class. The method
        is designed to be flexible in accepting any number and type of input
        parameters through positional (*args) and keyword (**kwargs) arguments. Subclasses
        must extend this method to provide concrete implementations tailored to their
        unique behaviors and requirements.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def type(self) -> Optional[Type[T]]:
        """
        If type is predefined return it, else if the variable is available return its type, else return None.
        """
        if self.type_ is not None:
            return self.type_
        if self.variable is None:
            return None
        return self.variable._type_

    @property
    def root(self) -> Match:
        """
        :return: The root match expression.
        """
        parent = self
        while parent.parent is not None:
            parent = parent.parent
        return parent

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(self.id)

    @property
    def descendants(self) -> Iterator[AbstractMatchExpression]:
        """
        :return: All descendants of this expression in breadth first order
        """
        queue = deque(self.children)
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    @property
    def matches_with_variables(self) -> Iterator[AttributeMatch]:
        """
        :return: All attribute matches where the assigned variable is a variable.
        These matches are typically the leaves of a match expression.
        """
        self.resolve()
        for expression in self.descendants:
            if isinstance(expression.assigned_variable, Variable):
                yield expression


@dataclass(eq=False)
class Match(AbstractMatchExpression[T], HasFactoryAndKwargs[T]):
    """
    Construct a query that looks for the pattern provided by the type and the keyword arguments.
    Example usage where we look for an object of type Drawer with body of type Body that has the name"drawer_1":
        >>> @dataclass
        >>> class Body:
        >>>     name: str
        >>> @dataclass
        >>> class Drawer:
        >>>     body: Body
        >>> drawer = match_variable(Drawer, domain=None)(body=match(Body)(name="drawer_1")))

    .. warning::
        Match can take a factory as a mean to construct `T`. If the keyword argument names of the match are not
        available in the class itself, the variables reffered to in the `where` conditions will not align with the
        variables from the factory. It is strongly recommended to have the names of the factory available in the class,
        either as field or as property.
        Dataclass-generated `__init__` never have this problem unless `InitVar` is used.
    """

    _expression: Query = field(init=False, default=None)
    """
    Cache for the expression (the actual EQL query) as soon as it has been calculated.
    This is needed to apply where conditions directly to the match instance. 
    """

    _where_conditions_: List[ConditionType] = field(init=False, default_factory=list)
    """
    A list of all conditions that have been applied to this instance using the `where` method.
    """

    _has_been_called: bool = field(init=False, default=False)
    """
    Flag indicating whether the match instance has been called with keyword arguments.
    """

    def __post_init__(self):
        if self.type_ is None:
            if isclass(self.factory):
                self.type_ = self.factory
            elif ismethod(self.factory):
                self.type_ = self.factory.__class__
            elif isfunction(self.factory):

                type_ = get_type_hints(self.factory)["return"]
                assert isclass(
                    type_
                ), f"factory must return a class, got {type_} which is of type {type(type_)}"
                self.type_ = type_

            else:
                assert_never(self.factory)

    def __call__(self, **kwargs) -> Union[T, Self, CanBehaveLikeAVariable[T]]:
        """
        Update the match with new keyword arguments to constrain the type we are matching with.

        :param kwargs: The keyword arguments to match against.
        :return: The current match instance after updating it with the new keyword arguments.
        """
        if self._has_been_called:
            raise CalledMatchMultipleTimes(self)
        self.kwargs = kwargs
        self._has_been_called = True
        return self

    @property
    def expression(self) -> Union[Entity[T], T]:
        """
        Return the entity expression corresponding to the match query.
        """
        from krrood.entity_query_language.factories import entity

        if self._expression is not None:
            return self._expression

        if self.variable is None:
            self.resolve()
        entity_ = entity(self.variable)
        if self.conditions:
            entity_ = entity_.where(*self.conditions)
        self._expression = entity_
        return entity_

    def _resolve(
        self,
        variable: Optional[Selectable] = None,
        parent: Optional[MatchVariable] = None,
    ):
        """
        Resolve the match by creating the variable and conditions expressions in-place.

        :param variable: An optional pre-existing variable to use for the match; if not provided, a new variable will
         be created.
        :param parent: The parent match if this is a nested match.
        """

        parent = parent or self
        self.update_fields(variable, parent)
        for attr_name, attr_assigned_value in self.kwargs.items():
            if isinstance(attr_assigned_value, (list, tuple)) and any(
                isinstance(element, AbstractMatchExpression)
                for element in attr_assigned_value
            ):
                self._resolve_list_like_value(attr_name, attr_assigned_value, parent)
                continue
            self._create_attribute_match_and_resolve(
                parent=parent,
                attribute_name=attr_name,
                assigned_value=attr_assigned_value,
            )

    def _create_attribute_match_and_resolve(
        self,
        parent: MatchVariable,
        attribute_name: str,
        assigned_value: Any,
        index_access: Optional[Any] = None,
    ) -> AttributeMatch:
        """
        Create an attribute match and resolve it recursively.

        :param parent: The parent match instance.
        :param attribute_name: The name of the attribute to create.
        :param assigned_value: The value assigned to the attribute.
        :param index_access: The index access to the attribute.
        :return: The created instance after every child has been resolved.
        """
        attr_match = AttributeMatch(
            parent=parent,
            attribute_name=attribute_name,
            index_access=index_access,
            assigned_value=assigned_value,
        )
        attr_match.resolve()
        self.children.append(attr_match)
        self.conditions.extend(attr_match.conditions)
        return attr_match

    def _resolve_list_like_value(
        self, key: str, value: Union[list, tuple], parent: MatchVariable
    ):
        """
        Resolves list-like values by iterating over their elements and creating attribute
        matches for the parent match variable.

        :param key: The attribute name being processed.
        :param value: The list or tuple containing elements to be resolved.
        :param parent: The parent match variable associated with the provided key.
        """

        # handle list like classes by wrapping the index access
        for index, element in enumerate(value):
            self._create_attribute_match_and_resolve(
                parent=parent,
                attribute_name=key,
                assigned_value=element,
                index_access=index,
            )

    def update_fields(
        self,
        variable: Optional[Selectable] = None,
        parent: Optional[AbstractMatchExpression] = None,
    ):
        """
        Update the match variable, and parent.

        :param variable: The variable to use for the match.
         If None, a new variable will be created.
        :param parent: The parent match if this is a nested match.
        """

        if variable is not None:
            self.variable = variable
        elif self.variable is None:
            self.create_variable()

        self.parent = parent

    def create_variable(self):
        from krrood.entity_query_language.factories import variable

        self.variable = variable(self.type, domain=None)

    def evaluate(self):
        """
        Evaluate the match expression and return the result.
        """
        return self.expression.evaluate()

    @property
    def name(self) -> str:
        return f"Match({self.type.__name__})"

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def where(self, *conditions: ConditionType) -> Match[T]:
        _ = self.expression
        self._where_conditions_.extend(conditions)
        self.expression.where(*conditions)
        self.expression.build()
        return self

    def _update_kwargs_from_literal_values(self):
        """
        Update the kwargs dictionary with values from this statements leaves.
        """
        for attribute_match in self.matches_with_variables:
            attribute_match._update_kwargs_from(self)

    def _get_mapped_variable_by_name(self, name: str) -> MappedVariable:
        """
        Get a mapped variable by its name in the path.
        :param name: The name
        :return: The mapped variable
        """
        [result] = [
            attribute_match.assigned_variable
            for attribute_match in self.matches_with_variables
            if attribute_match.name_from_variable_access_path == name
        ]
        return result


@dataclass(eq=False)
class MatchVariable(Match[T]):
    """
    Represents a match variable that operates within a specified domain.

    A class designed to create and manage a variable constrained by a defined
    domain. It provides functionality to add additional constraints via
    keyword arguments and return an expression representing the resolved
    constraints.
    """

    domain: Optional[DomainType] = field(default=None, kw_only=True)
    """
    The domain to use for the variable created by the match.
    """

    def create_variable(self):
        from krrood.entity_query_language.factories import variable

        self.variable = variable(self.type, domain=self.domain)

    def __call__(self, **kwargs) -> Union[Entity[T], T]:
        """
        Add kwargs constraints and return the resolved expression as An() instance.
        """
        if not kwargs:
            raise NoKwargsInMatchVar(self)
        super().__call__(**kwargs)
        return self.expression


@dataclass(eq=False)
class AttributeMatch(AbstractMatchExpression[T]):
    """
    A class representing an attribute assignment in a Match statement.
    """

    parent: AbstractMatchExpression = field(kw_only=True)
    """
    The parent match expression.
    """
    attribute_name: str = field(kw_only=True)
    """
    The name of the attribute to assign the value to.
    """

    index_access: Optional[Any] = None
    """
    The index  that is accessed.
    Is not None if the attribute is an indexable object.
    """

    assigned_value: Optional[Union[Literal, Match]] = None
    """
    The value to assign to the attribute, which can be a Match instance or a Literal.
    """
    variable: Union[Attribute, FlatVariable] = field(default=None, kw_only=True)
    """
    The symbolic variable representing the attribute.
    """

    def __post_init__(self):
        if isinstance(self.assigned_value, Match):
            self.children = self.assigned_value.children

    @cached_property
    def expression(self) -> Union[CanBehaveLikeAVariable[T], T]:
        """
        Return the entity expression corresponding to the match query.
        """
        if not self.variable:
            self.resolve()
        return self.variable

    def _resolve(self):
        """
        Resolve the attribute assignment by creating the conditions and applying the necessary mappings
        to the attribute.
        """
        if not isinstance(self.assigned_value, AbstractMatchExpression) or (
            self.assigned_value.variable or self.assigned_value.resolved
        ):
            self.conditions.append(self.attribute == self.assigned_variable)
            return

        self.assigned_value.resolve(self.attribute, self)

        if self.is_type_filter_needed:
            self.conditions.append(HasType(self.attribute, self.assigned_value.type))

        self.conditions.extend(self.assigned_value.conditions)

    @cached_property
    def assigned_variable(self) -> Selectable:
        """
        :return: The symbolic variable representing the assigned value.
        """
        if isinstance(self.assigned_value, AbstractMatchExpression):
            return self.assigned_value.variable
        elif not isinstance(self.assigned_value, SymbolicExpression):
            return Literal(
                _name__=self.variable._name_,
                _type_=self.type,
                _value_=self.assigned_value,
            )
        else:
            return self.assigned_value

    @cached_property
    def attribute(self) -> Attribute:
        """
        :return: the attribute of the variable.
        :raises NoneWrappedFieldError: If the attribute does not have a WrappedField.
        """
        if self.variable is not None:
            return self.variable

        attr: Attribute = getattr(self.parent.variable, self.attribute_name)
        if self.index_access is not None:
            attr = attr[self.index_access]
        self.variable = attr
        return attr

    @cached_property
    def is_type_filter_needed(self):
        """
        :return: True if a type filter condition is needed for the attribute assignment, else False.
        """
        attr_type = self.type
        return (not attr_type) or (
            (self.assigned_value.type and self.assigned_value.type is not attr_type)
            and issubclass(self.assigned_value.type, attr_type)
        )

    @property
    def name(self) -> str:
        return f"{self.parent.name}.{self.attribute_name}"

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def _update_kwargs_from(self, match: Match[T]):
        """
        Update the kwargs of the parent match with the values of the assigned variable.
        Only works if this is a variable assignment.
        """
        current_value = match
        for step in self.variable._access_path_[:-1]:
            if isinstance(step, Attribute):
                current_value = current_value.kwargs[step._attribute_name_]
            elif isinstance(step, Index):
                current_value = current_value[step._key_]
            else:
                assert_never(step)

        final_step = self.variable._access_path_[-1]

        if isinstance(final_step, Attribute):
            current_value.kwargs[final_step._attribute_name_] = (
                self.assigned_variable._value_
            )
        else:

            final_step._set_child_instance_value_(
                current_value, self.assigned_variable._value_
            )

    @property
    def name_from_variable_access_path(self):
        """
        :return: The last name from the variables access path. This is similar to `self.name` but without `Match`
        specific wrappings.
        """
        return self.variable._access_path_[-1]._name_

    @property
    def type(self) -> Optional[Type[T]]:
        result = super().type
        if result is not None:
            return result

        if isinstance(self.parent, AttributeMatch):
            return get_field_type_endpoint(
                self.parent.assigned_value.type, self.variable._attribute_name_
            )
        return None


def construct_graph_and_get_root(
    node_data: AbstractMatchExpression, graph: Optional[rx.PyDAG] = None
) -> RWXNode:
    """
    Construct a graph representation of the match expression and return the root node.

    :param node_data: The root node of the match expression.
    :param graph: The graph to construct the subgraph in.
    :return: The root node of the constructed subgraph.
    """
    graph = graph or rx.PyDAG()
    node = RWXNode(node_data.name, graph, data=node_data)
    for child in node_data.children:
        child_node = construct_graph_and_get_root(child, graph=graph)
        child_node.parent = node
    return node
