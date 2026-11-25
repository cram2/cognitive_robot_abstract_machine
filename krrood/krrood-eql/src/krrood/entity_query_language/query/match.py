"""
Pattern-matching helpers for the Entity Query Language.

This module provides high-level match abstractions that build symbolic expressions for
variables and attributes from concise, readable matching syntax.

The match classes keep their own noun state in underscore-wrapped names (``_type_``,
``_variable_``, ...), following the convention of
:class:`~krrood.entity_query_language.core.mapped_variable.CanBehaveLikeAVariable`: the
public attribute namespace of a :class:`Match` belongs to the matched class, whose
fields are reachable directly as symbolic attributes.
"""

from __future__ import annotations

import numbers
import operator
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property, lru_cache
from inspect import isclass
from typing import TypeVar
from types import UnionType
from typing_extensions import (
    Callable,
    Generic,
    TYPE_CHECKING,
    Self,
    Iterator,
    Iterable,
    Any,
    get_args,
    get_origin,
)

import rustworkx as rx

from krrood.class_diagrams.utils import (
    common_base_class,
    get_type_hints_of_object,
    is_union_annotation,
)
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.entity_query_language.core.base_expressions import (
    Bindings,
    HasExpression,
    MatchAssignedValue,
    OperationResult,
    Selectable,
    SymbolicExpression,
)
from krrood.entity_query_language.operators.causal import (
    CausalRoleMarker,
    Cause,
    CausesEffect,
)
from krrood.entity_query_language.core.helpers import _resolve_domain
from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    CanBehaveLikeAVariable,
    HasSymbolicOperations,
)
from krrood.entity_query_language.core.variable import Literal, DomainType, Variable
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.exceptions import (
    CalledMatchAfterResolution,
    CalledMatchMultipleTimes,
    ElementQuantifierWithoutAttribute,
    MatchTypeCannotBeDetermined,
    PositionalArgumentsInMatchPattern,
    NoAttributeWithAccessPathName,
    UnboundPatternVariable,
    ValueFitsNeitherElementNorCollection,
)
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    chained_logic,
)
from krrood.entity_query_language.operators.logical_quantifiers import Exists, ForAll
from krrood.entity_query_language.predicate import HasType, length
from krrood.entity_query_language.query.quantifiers import An, ResultQuantifier
from krrood.entity_query_language.query.query import MatchedEntity
from krrood.entity_query_language.query.query_modifiers import HasQueryModifiers
from krrood.entity_query_language.utils import T
from krrood.patterns.factory_and_kwargs import HasFactoryAndKwargs
from krrood.class_diagrams.call_signature import CallSignature
from krrood.rustworkx_utils.rxnode import RWXNode
from krrood.entity_query_language.class_diagram_lookup import (
    get_field_type_endpoint,
    get_wrapped_field,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.backends import QueryBackend
    from krrood.entity_query_language.factories import ConditionType
    from krrood.entity_query_language.ground_term import TermReader
    from krrood.entity_query_language.query.query import Entity, Query


# %% element quantifiers


class ElementQuantifier(Enum):
    """
    How many elements of a collection attribute a pattern given as its value describes.
    """

    SOME = auto()
    """
    At least one element matches the pattern.
    """

    EVERY = auto()
    """
    Every element matches the pattern, which an empty collection satisfies.
    """

    NO = auto()
    """
    No element matches the pattern, which an empty collection satisfies.
    """

    def condition_on_elements(
        self, *, element: CanBehaveLikeAVariable, condition: ConditionType
    ) -> SymbolicExpression:
        """
        :param element: A variable over the elements of a collection.
        :param condition: The condition of the pattern on *element*.
        :return: The condition that this many elements satisfy *condition*.
        """
        match self:
            case ElementQuantifier.SOME:
                return Exists(left=element, right=condition)
            case ElementQuantifier.EVERY:
                return ForAll(left=element, right=condition)
            case ElementQuantifier.NO:
                return Exists(left=element, right=condition)._invert_()

    def conditions_on_value(
        self, *, conditions: list[ConditionType]
    ) -> list[ConditionType]:
        """
        :param conditions: The conditions of the pattern on an attribute holding one
            value.
        :return: The conditions that the value satisfies as this many elements of a
            collection of it alone would.
        """
        if self is ElementQuantifier.NO:
            return [chained_logic(AND, *conditions)._invert_()]
        return conditions


# %% match expressions


@dataclass(kw_only=True)
class AbstractMatchExpression(MatchAssignedValue, Generic[T], ABC):
    """
    Abstract base class for constructing and handling a match expression.

    This class is intended to provide a framework for defining and managing match
    expressions, which are used to structural pattern matching in the form of nested
    match expressions with keyword arguments.
    """

    _declared_type_: type[T] | None = field(default=None, kw_only=True)
    """
    The matched type as explicitly given (or inferred from the factory); ``None`` until
    known.
    """

    _variable_: Variable[T] | None = field(default=None, kw_only=True)
    """
    The created variable from the type and keyword arguments.
    """

    _conditions_: list[ConditionType] = field(init=False, default_factory=list)
    """
    The conditions that define the match.
    """

    _parent_: AbstractMatchExpression | None = field(init=False, default=None)
    """
    The parent match if this is a nested match.
    """

    _resolved_: bool = field(init=False, default=False)
    """
    Whether the match is resolved or not.
    """

    _id_: uuid.UUID = field(init=False, default_factory=uuid.uuid4)
    """
    The unique identifier of the match expression.
    """

    _children_: list[AttributeMatch] = field(init=False, default_factory=list)
    """
    The child matches of this match expression.
    """

    @property
    @abstractmethod
    def _symbolic_expression_(self) -> CanBehaveLikeAVariable[T] | T:
        """
        :return: The expression this match expression stands for, on which every symbolic
            operation written on it is built.
        """
        ...

    def _as_assigned_variable_(
        self, *, attribute_type: type | None
    ) -> Variable[T] | None:
        """
        :param attribute_type: The type of the attribute this match is assigned to.
        :return: The variable this match creates, which the attribute's value is bound to.
        """
        return self._variable_

    def resolve(self, **kwargs: Any) -> Self:
        """
        Resolve the match by creating the variable and conditions expressions.
        """
        if self._resolved_:
            return self
        self._resolve(**kwargs)
        self._resolved_ = True
        return self

    @abstractmethod
    def _resolve(self, **kwargs: Any) -> None:
        """
        This method serves as an abstract definition to be implemented by subclasses,
        aimed at handling specific resolution logic for the derived class.

        Subclasses take the arguments of :meth:`resolve` as keywords.
        """
        ...

    @property
    @abstractmethod
    def _name_(self) -> str: ...

    @property
    def _type_(self) -> type[T] | None:
        """
        If the declared type is available return it, else if the variable is available
        return its type, else return None.
        """
        if self._declared_type_ is not None:
            return self._declared_type_
        if self._variable_ is None:
            return None
        return self._variable_._type_

    @property
    def _root_(self) -> Match:
        """
        :return: The root match expression.
        """
        parent = self
        while parent._parent_ is not None:
            parent = parent._parent_
        return parent

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(self._id_)

    @property
    def _descendants_(self) -> Iterator[AbstractMatchExpression]:
        """
        :return: All descendants of this expression in breadth first order
        """
        queue = deque(self._children_)
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node._children_)

    @property
    def _matches_with_variables_(self) -> Iterator[AttributeMatch]:
        """
        :return: All attribute matches where the assigned variable is a variable, or
            where the pattern assigns an expression derived from one, such as its
            attribute or its flattened values. These matches are typically the leaves of
            a match expression.
        """
        self.resolve()
        for expression in self._descendants_:
            if isinstance(expression.assigned_variable, Variable) or isinstance(
                expression.assigned_value, CanBehaveLikeAVariable
            ):
                yield expression


@dataclass(kw_only=True, eq=False)
class Match(
    Evaluable,
    HasQueryModifiers[T],
    HasSymbolicOperations[T],
    AbstractMatchExpression[T],
    HasFactoryAndKwargs[T],
    HasExpression,
):
    """
    Construct a query that looks for the pattern provided by the type and the keyword arguments.
    Example usage where we look for an object of type Drawer with body of type Body that has the name"drawer_1":
        >>> @dataclass
        >>> class Body:
        >>>     name: str
        >>> @dataclass
        >>> class Drawer:
        >>>     body: Body
        >>> drawer = a(Drawer)(body=a(Body)(name="drawer_1")).from_(domain=world.views)

    A match reads like an instance of the matched class - ``drawer.body`` is the ``body``
    of the query the match stands for - and like a query, whose modifiers return the
    match.

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

    _where_conditions_: list[ConditionType] = field(init=False, default_factory=list)
    """
    A list of all conditions that have been applied to this instance using the
    `where` method.
    """

    _has_been_called: bool = field(init=False, default=False)
    """
    Flag indicating whether the match instance has been called with keyword arguments.
    """

    _quantifier_type_: type[ResultQuantifier] = field(init=False, default=An)
    """
    The result quantifier applied when this match is materialized into a runnable query.

    Defaults to ``An`` (zero or more results); set to ``The`` when built via
    ``the(...)``.
    """

    _element_quantifier_: ElementQuantifier = field(
        init=False, default=ElementQuantifier.SOME
    )
    """
    How many elements of the collection attribute this pattern is the value of it
    describes.
    """

    _type_union_: UnionType | None = field(init=False, default=None)
    """
    The union of classes the matched object is an instance of one of, when the pattern
    names a union; the matched type is then their nearest common base class.
    """

    _domain_: DomainType | None = field(default=None, init=False)
    """
    The instances the match ranges over.

    ``None`` constructs from scratch (an underspecified, generative request); a domain
    makes it a search over those existing instances.
    """

    def __post_init__(self):
        """
        Take the matched type from :attr:`_factory_` when none was declared.

        A union of classes given as :attr:`_factory_` becomes :attr:`_type_union_`, and
        its nearest common base class the factory.

        :raises MatchTypeCannotBeDetermined: If :attr:`_factory_` names no type it
            builds.
        """
        if is_union_annotation(self._factory_):
            self._type_union_ = self._factory_
            self._factory_ = common_base_class(list(get_args(self._factory_))) or object
        if self._declared_type_ is not None:
            return
        self._declared_type_ = self._type_built_by_(factory=self._factory_)
        if self._declared_type_ is None:
            raise MatchTypeCannotBeDetermined(match=self)

    def __call__(self, *args: Any, **kwargs: Any) -> T | Self:
        """
        Set the pattern the match looks for, the first time; call the matched instance
        symbolically, every time after that.

        The first parentheses after ``a(Drawer)`` state the pattern, so a matched class
        that is itself callable is called through a second pair - ``an(Adder)(offset=1)(2)``,
        or ``an(Adder)()(2)`` where the pattern is empty. The pattern parentheses take
        keyword arguments only, since a pattern names fields.

        A matched class whose instances are not callable has nothing a second call could
        mean, so it still raises: writing the pattern twice is
        :class:`~krrood.entity_query_language.exceptions.CalledMatchMultipleTimes`, and
        writing it after the match was lowered is
        :class:`~krrood.entity_query_language.exceptions.CalledMatchAfterResolution`.

        Setting the pattern eagerly creates the match's subject variable so it can be
        referenced in ``where`` conditions immediately (lowering the pattern into
        conditions stays lazy, tracked by ``_resolved_``). If this match is later nested
        under a parent, the parent overwrites the subject with its own attribute during
        resolution.

        :param args: The positional arguments of a symbolic call; never part of a pattern.
        :param kwargs: The pattern's keyword arguments, or a symbolic call's.
        :return: This match when the pattern was set, otherwise the symbolic call.
        """
        if self._has_been_called or self._resolved_:
            if not self._matched_instances_are_callable_:
                raise (
                    CalledMatchAfterResolution(match=self)
                    if self._resolved_
                    else CalledMatchMultipleTimes(match=self)
                )
            return HasSymbolicOperations.__call__(self, *args, **kwargs)
        if args:
            raise PositionalArgumentsInMatchPattern(match=self, arguments=args)
        self._kwargs_ = kwargs
        self._has_been_called = True
        if self._variable_ is None:
            self._create_or_update_variable_()
        return self

    @property
    def _matched_instances_are_callable_(self) -> bool:
        """
        :return: Whether instances of the matched class can be called, which is what a
            call beyond the pattern's own means.
        """
        return "__call__" in dir(self._type_)

    def _is_own_name_(self, *, name: str) -> bool:
        """
        :param name: A name that this match does not define.
        :return: Whether the name belongs to the match machinery, which keeps its own
            state behind underscore-prefixed names, leaving every other name to the
            matched class.
        """
        return name.startswith("_")

    def _quantifier_operands_(
        self, *, condition: ConditionType
    ) -> tuple[Variable[T], ConditionType]:
        """
        Overrides :meth:`Operand._quantifier_operands_`: a quantifier over a match
        ranges over the variable it describes, and states the pattern and the
        :meth:`where` conditions of each value together with *condition*.
        """
        self.resolve()
        return self._variable_, chained_logic(
            AND, *self._conditions_, *self._where_conditions_, condition
        )

    @property
    def _symbolic_expression_(self) -> Entity[T]:
        """
        :return: The query this match stands for - its pattern lowered to a selection of
            the variable it describes - which every symbolic operation on the match is
            built on, so ``a(Drawer).body`` is that query's ``body`` and carries the
            pattern.

        Operations are built *on* that query rather than read *off* it, so nothing in its
        own namespace - its modifiers, ``build``, ``evaluate`` - can stand in for a
        matched class's field of the same name.
        """
        if self._expression is not None:
            return self._expression

        if not self._resolved_:
            self.resolve()
        entity_ = MatchedEntity(_selected_variables_=(self._variable_,))
        if self._conditions_:
            entity_ = entity_.where(*self._conditions_)
        entity_._quantify_(quantifier_type=self._quantifier_type_)
        self._expression = entity_
        return entity_

    def _get_expression_(self) -> SymbolicExpression:
        return self._symbolic_expression_

    def _resolve(
        self,
        *,
        variable: Selectable | None = None,
        parent: Match | None = None,
    ):
        """
        Resolve the match by creating the variable and conditions expressions in-place.

        A nested match given a domain requires the object it describes to be in that
        domain.

        :param variable: An optional pre-existing variable to use for the match; if not
            provided, a new variable will be created.
        :param parent: The parent match if this is a nested match.
        """
        parent = parent or self
        self._update_fields_(variable=variable, parent=parent)
        self._conditions_.extend(self._type_union_conditions_)
        if variable is not None and self._domain_ is not None:
            # A variable over the domain reads a one-shot source once, and an expression
            # domain per binding, where a membership test would re-read the raw domain.
            domain_member = Variable(
                _type_=self._type_,
                _domain_=_resolve_domain(type_=self._type_, domain=self._domain_),
            )
            self._conditions_.append(self._variable_ == domain_member)
        for attr_name, attr_assigned_value in self._kwargs_.items():
            if isinstance(attr_assigned_value, (list, tuple)) and any(
                isinstance(element, AbstractMatchExpression)
                for element in attr_assigned_value
            ):
                self._resolve_list_like_value(
                    key=attr_name, value=attr_assigned_value, parent=parent
                )
                continue
            self._create_attribute_match_and_resolve(
                parent=parent,
                attribute_name=attr_name,
                assigned_value=attr_assigned_value,
            )

    def _create_attribute_match_and_resolve(
        self,
        *,
        parent: Match,
        attribute_name: str,
        assigned_value: Any,
        index_access: Any | None = None,
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
            _parent_=parent,
            attribute_name=attribute_name,
            index_access=index_access,
            assigned_value=assigned_value,
        )
        attr_match.resolve()
        self._children_.append(attr_match)
        self._conditions_.extend(attr_match._conditions_)
        return attr_match

    def _resolve_list_like_value(self, *, key: str, value: list | tuple, parent: Match):
        """
        Resolves list-like values by iterating over their elements and creating
        attribute matches for the parent match variable; the attribute must hold as many
        elements as the value.

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
        collection = getattr(self._variable_, key)
        # An absent optional collection holds no elements and has no length to compare.
        self._conditions_.append(
            AND(
                left=collection != None,
                right=length(iterable=collection) == len(value),
            )
        )

    def _update_fields_(
        self,
        *,
        variable: Selectable | None = None,
        parent: AbstractMatchExpression | None = None,
    ):
        """
        Update the match variable, and parent.

        :param variable: The variable to use for the match. If None, a new variable will
            be created.
        :param parent: The parent match if this is a nested match.
        """
        if variable is not None:
            self._variable_ = variable
        elif self._variable_ is None:
            self._create_or_update_variable_()

        self._parent_ = parent

    def _create_or_update_variable_(self):
        """
        Create the subject variable from this match's current type and domain.

        If a subject variable already exists (``from_`` re-scoping the domain after
        ``__call__`` eagerly created one), its domain is updated in place instead of
        replacing the variable outright: conditions built earlier against ``self._variable_``
        (for example from an already-recorded ``where``) reference that same object, so
        replacing it would silently orphan them from the re-scoped domain.
        """
        if self._variable_ is None:
            from krrood.entity_query_language.factories import variable

            self._variable_ = variable(self._type_, domain=self._domain_)
            return

        self._variable_._update_domain_(
            domain=_resolve_domain(self._type_, self._domain_)
        )

    def evaluate(
        self, backend: QueryBackend | None = None, state: TermReader | None = None
    ) -> Iterator:
        """
        Overrides :meth:`~krrood.entity_query_language.evaluable.Evaluable.evaluate`: a
        pattern that states how many elements of an attribute match is not evaluated on
        its own.

        :raises ElementQuantifierWithoutAttribute: If this pattern was made with
            :func:`~krrood.entity_query_language.factories.every_is` or
            :func:`~krrood.entity_query_language.factories.none_is`.
        """
        if self._element_quantifier_ is not ElementQuantifier.SOME:
            raise ElementQuantifierWithoutAttribute(pattern=self)
        return super().evaluate(backend=backend, state=state)

    def _evaluate_natively_(self) -> Iterator:
        """
        Select the elements of the match's domain (its variable's domain, or the
        ``SymbolGraph`` for ``Symbol`` types without one) that satisfy the structural
        pattern and ``where`` conditions, in this process.

        .. note::
            Constructing *new* instances from an underspecified match is the job of a
            :class:`~krrood.entity_query_language.backends.GenerativeBackend`, not of
            the default selective evaluation.

        :return: An iterator over the matching elements.
        """
        return self._symbolic_expression_._evaluate_natively_()

    def _select_satisfying_(self, *, instances: Iterable[T]) -> Entity[T]:
        """
        Select, from instances constructed for this match, those that satisfy it.

        Construction passes the factory every value it takes, but a stated value of an
        attribute the factory does not take, such as a property, and a ``where``
        condition hold only if the constructed values happen to imply them.

        :param instances: Instances constructed from this match's pattern.
        :return: The query selecting the instances that have every value the pattern
            states for an attribute its factory does not take, and satisfy every
            ``where`` condition.
        """
        from krrood.entity_query_language.factories import entity

        self._variable_._update_domain_(domain=instances)
        selection = entity(self._variable_)
        conditions = [
            *self._type_union_conditions_,
            *self._conditions_on_stated_values_,
            *self._where_conditions_,
        ]
        if conditions:
            selection = selection.where(*conditions)
        return selection

    def _find_assigned_variable_(self, *, access_path_name: str) -> Variable:
        """
        :param access_path_name: The name of an attribute's access path, e.g.
            ``"Rectangle.width"``, which is also the name of the model variable the
            attribute corresponds to.
        :return: The variable this match's pattern assigns to that attribute.
        :raises NoAttributeWithAccessPathName: If the pattern assigns a variable to no
            attribute with that access path name.
        """
        for attribute_match in self._matches_with_variables_:
            if attribute_match.name_from_variable_access_path == access_path_name:
                return attribute_match.assigned_variable
        raise NoAttributeWithAccessPathName(
            match=self, access_path_name=access_path_name
        )

    def _construct_instance_from_bindings_(self, *, bindings: Bindings) -> T:
        """
        Construct an instance of the matched type from the values the bindings give the
        variables of this match's pattern, leaving the pattern itself unchanged.

        An attribute whose assigned variable the bindings do not bind takes the value
        the pattern states for it.

        :param bindings: The values of the pattern's variables, keyed by the identifier
            of the variable assigned to each attribute.
        :return: The constructed instance.
        :raises UnboundPatternVariable: If the bindings do not bind a variable that
            states no value of its own.
        """
        self.resolve()
        return self._factory_(
            **{
                keyword: self._construct_argument_from_bindings_(
                    keyword=keyword, bindings=bindings
                )
                for keyword in self._factory_keyword_arguments_
            }
        )

    def _construct_argument_from_bindings_(
        self, *, keyword: str, bindings: Bindings
    ) -> Any:
        """
        :param keyword: A keyword argument of the pattern that is passed to the factory.
        :param bindings: The values of the pattern's variables, keyed by the identifier
            of the variable assigned to each attribute.
        :return: The value the keyword's attribute takes under the bindings; a list-like
            value whose elements are matched one by one is rebuilt from its elements.
        """
        attribute_matches = [
            attribute_match
            for attribute_match in self._children_
            if attribute_match.attribute_name == keyword
        ]
        if attribute_matches[0].index_access is None:
            return attribute_matches[0].construct_value_from_bindings(bindings=bindings)
        return type(self._kwargs_[keyword])(
            attribute_match.construct_value_from_bindings(bindings=bindings)
            for attribute_match in attribute_matches
        )

    @property
    def _type_union_conditions_(self) -> list[ConditionType]:
        """
        :return: The condition that the matched object is an instance of a class of
            :attr:`_type_union_`, or none when the pattern names a single class.
        """
        if self._type_union_ is None:
            return []
        return [HasType(variable=self._variable_, types_=self._type_union_)]

    @property
    def _conditions_on_stated_values_(self) -> Iterator[ConditionType]:
        """
        :return: The conditions requiring an attribute to have the plain value the
            pattern states for it, anywhere in the pattern, where the factory building
            the attribute's owner does not take it. A value the factory takes is its
            to keep under any name, and an attribute left open with ``...`` or given a
            symbolic value states no single value.
        """
        for attribute_match in self._matches_with_variables_:
            value = attribute_match.assigned_value
            if isinstance(value, MatchAssignedValue) or self._is_or_contains_ellipsis(
                value=value
            ):
                continue
            if attribute_match._stating_match_._is_factory_parameter_(
                keyword=attribute_match.attribute_name
            ):
                continue
            yield from attribute_match._conditions_

    @property
    def _has_ellipsis_attributes_(self) -> bool:
        """
        :return: Whether any attribute anywhere in this match's pattern (including nested
            matches, and ``...`` elements inside an otherwise-concrete list/tuple attribute)
            is left fully unspecified via ``...``, requiring construction rather than search
            to resolve.
        """
        return any(
            self._is_or_contains_ellipsis(value=attribute_match.assigned_value)
            for attribute_match in self._matches_with_variables_
        )

    @staticmethod
    def _is_or_contains_ellipsis(*, value: Any) -> bool:
        """
        :param value: An attribute match's assigned value.
        :return: Whether ``value`` is ``...`` itself, or a list/tuple/set containing ``...`` as
            one of its elements (such a collection with no nested :class:`Match` element is
            resolved as a single literal attribute match, so its elements are never visited
            individually). Deliberately not any ``Iterable``: a generator would be exhausted by
            this check, and a plain iterable is not a container kind the rest of this class
            resolves or is tested against.
        """
        if isinstance(value, (list, tuple, set)):
            return any(isinstance(element, type(Ellipsis)) for element in value)
        return isinstance(value, type(Ellipsis))

    @property
    def _has_cause_attributes_(self) -> bool:
        """
        :return: Whether any attribute anywhere in this match's pattern (including nested
            matches) is marked with :func:`~krrood.entity_query_language.factories.cause` --
            a ``do()``-intervention target only a causal backend can resolve.
        """
        return any(
            isinstance(attribute_match.assigned_value, Cause)
            for attribute_match in self._matches_with_variables_
        )

    def _is_kept_out_of_construction_(self, *, keyword: str, value_type: type) -> bool:
        """
        A keyword the factory does not take still belongs to the pattern when it names
        an attribute of the matched class, such as a property or a method, since the
        constructed instance provides it.

        A keyword marked with a causal role names an aggregation statistic of the
        matched class, which grounding computes.
        """
        return self._is_pattern_keyword_kept_out_of_construction_(
            factory=self._factory_,
            matched_type=self._type_,
            keyword=keyword,
            value_type=value_type,
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def _is_pattern_keyword_kept_out_of_construction_(
        *,
        factory: Callable[..., Any],
        matched_type: type,
        keyword: str,
        value_type: type,
    ) -> bool:
        """
        :param factory: The factory constructing the matched instance.
        :param matched_type: The matched class.
        :param keyword: A keyword of the pattern.
        :param value_type: The type of the value the pattern gives for that keyword.
        :return: Whether that keyword is left out of construction, decided once per
            factory, matched class, keyword and value type.
        """
        if CallSignature.from_callable(callee=factory).accepts_keyword(keyword=keyword):
            return False
        return keyword in dir(matched_type) or issubclass(value_type, CausalRoleMarker)

    @property
    def _name_(self) -> str:
        type_name = self._type_.__name__ if self._type_ is not None else "?"
        return f"Match({type_name})"

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_

    def where(self, *conditions: ConditionType) -> Self:
        """
        Constrain the matched instance by conditions, on top of the pattern.

        The conditions are also kept on the match itself, since a generative backend
        reads them from the pattern rather than from the lowered query.

        :param conditions: The conditions the matched instance must satisfy.
        :return: This match.
        """
        self._where_conditions_.extend(conditions)
        self._symbolic_expression_.where(*conditions)
        return self

    def having(self, *conditions: ConditionType) -> Self:
        """
        Constrain the grouped results of the lowered query.

        :param conditions: The conditions a group must satisfy.
        :return: This match.
        """
        self._symbolic_expression_.having(*conditions)
        return self

    def ordered_by(
        self,
        *,
        variable: Selectable[T] | Any,
        descending: bool = False,
        key: Callable | None = None,
    ) -> Self:
        """
        Order the matched instances by the given expression.

        :param variable: The expression to order by.
        :param descending: Whether to order the results in descending order.
        :param key: A function to extract the key from the expression's value.
        :return: This match.
        """
        self._symbolic_expression_.ordered_by(
            variable=variable, descending=descending, key=key
        )
        return self

    def distinct(self, *on: Selectable | Any) -> Self:
        """
        Keep only matched instances that differ in the given expressions.

        :param on: The expressions the results must differ in; the matched instance by
            default.
        :return: This match.
        """
        self._symbolic_expression_.distinct(*on)
        return self

    def grouped_by(self, *variables_to_group_by: Selectable | Any) -> Self:
        """
        Group the matched instances by the given expressions.

        :param variables_to_group_by: The expressions to group the results by.
        :return: This match.
        """
        self._symbolic_expression_.grouped_by(*variables_to_group_by)
        return self

    def limit(self, *, n: int) -> Self:
        """
        Return at most ``n`` matched instances.

        :param n: The maximum number of results to return.
        :return: This match.
        """
        self._symbolic_expression_.limit(n=n)
        return self

    @property
    def _limit_(self) -> int | None:
        """
        :return: The most matched instances this match returns, or ``None`` if it was
            given no :meth:`limit`.
        """
        return self._get_expression_()._limit_

    def causes_effect(self, *conditions: ConditionType) -> Match[T]:
        """
        Mark condition(s) as the effect side of a causal query, e.g.
        ``a(Pick)(arm=cause).causes_effect(pick.action.status == SUCCESS)``.

        Sugar for ``self.where(CausesEffect(and_(*conditions), cause_attributes=...))``:
        semantically identical to an ordinary ``.where()`` under every backend except
        :class:`~krrood.parametrization.probabilistic_backend.ProbabilisticBackend`, which reads
        the wrapped condition to find which variable(s) a
        :data:`~krrood.entity_query_language.factories.cause` search should optimize for.
        The ``cause``-marked attribute(s) are also attached to the built
        :class:`~krrood.entity_query_language.operators.causal.CausesEffect` node, so its
        verbalization can name them.

        :param conditions: One literal comparator, or several combined with AND.
        :return: This match, for chaining.
        """
        # `and_` stays a local import: factories.py imports this module, so a module-level
        # import here would be circular.
        from krrood.entity_query_language.factories import and_

        cause_attributes = [
            attribute_match.attribute
            for attribute_match in self._matches_with_variables_
            if isinstance(attribute_match.assigned_value, Cause)
        ]
        return self.where(
            CausesEffect(_child_=and_(*conditions), cause_attributes=cause_attributes)
        )

    def from_(self, *, domain: DomainType) -> Self:
        """
        Range the match over ``domain`` instead of over all instances of its type.

        A domain does not commit the match to selection: the chosen backend decides what to do
        with it (a selective backend finds the matching existing instances, a generative backend
        constructs or completes them), so this stays a :class:`Match`. Use :attr:`expression` to
        get the lowered selection query when you need symbolic attribute access (``.parent`` /
        ``.child``) on a name the match's own methods shadow, ``the(...)`` or ``set_of(...)``.

        .. note::
            ``__call__`` eagerly creates a subject variable before the domain is known (and with
            no domain that is a SymbolGraph-wide variable for Symbol types). ``_create_or_update_variable_``
            re-scopes that same variable's domain in place (see its docstring) rather than
            replacing it, so a ``where`` recorded before this call keeps referencing the correct,
            now domain-scoped, variable.

        An expression as ``domain`` ranges the match, for each binding of the expression,
        over the instances of its type among the elements of the value it evaluates to.

        :param domain: The instances the match ranges over.
        :return: This match, for chaining.
        """
        self._domain_ = domain
        self._create_or_update_variable_()
        return self


@dataclass(kw_only=True, eq=False)
class AttributeMatch(AbstractMatchExpression[T]):
    """
    A class representing an attribute assignment in a Match statement.
    """

    _parent_: AbstractMatchExpression = field(kw_only=True)
    """
    The parent match expression.
    """

    attribute_name: str = field(kw_only=True)
    """
    The name of the attribute to assign the value to.
    """

    index_access: Any | None = None
    """
    The index  that is accessed.

    Is not None if the attribute is an indexable object.
    """

    assigned_value: Literal | Match | None = None
    """
    The value to assign to the attribute, which can be a Match instance or a Literal.
    """

    _variable_: Attribute | Variable = field(default=None, kw_only=True)
    """
    The symbolic variable representing the attribute.
    """

    def __post_init__(self):
        if isinstance(self.assigned_value, Match):
            self._children_ = self.assigned_value._children_

    @cached_property
    def _symbolic_expression_(self) -> CanBehaveLikeAVariable[T] | T:
        """
        :return: The symbolic attribute this match stands for, which is the variable it
            assigns the matched value to.
        """
        if not self._variable_:
            self.resolve()
        return self._variable_

    def _resolve(self):
        """
        Resolve the attribute assignment into conditions on the attribute: on a
        collection, a value must be in it, and as many elements as the pattern's
        :class:`ElementQuantifier` states must match a pattern.

        A pattern evaluated before it was nested keeps its :meth:`Match.where`
        conditions.
        """
        attribute = self.attribute
        if not isinstance(self.assigned_value, AbstractMatchExpression):
            self._conditions_.append(self._build_condition_on_assigned_value_())
            return

        element_quantifier = self.assigned_value._element_quantifier_
        if self.assigned_value._resolved_:
            self._conditions_.append(
                self._condition_on_resolved_pattern_(
                    element_quantifier=element_quantifier
                )
            )
            return

        if not self._states_an_element_of_the_attribute_:
            self._conditions_.extend(
                element_quantifier.conditions_on_value(
                    conditions=self._resolve_pattern_against_(variable=attribute)
                )
            )
            return

        element = Variable(_domain_=attribute)
        # The pattern's own attributes are read off this match's variable, which now
        # stands for one element of the collection rather than the whole collection.
        self._variable_ = element
        self._conditions_.append(
            element_quantifier.condition_on_elements(
                element=element,
                condition=chained_logic(
                    AND, *self._resolve_pattern_against_(variable=element)
                ),
            )
        )

    def _condition_on_resolved_pattern_(
        self, *, element_quantifier: ElementQuantifier
    ) -> SymbolicExpression:
        """
        :param element_quantifier: How many elements of the attribute the pattern
            describes.
        :return: The condition that that many elements of the attribute are values of
            the assigned pattern, which was resolved before it was nested, satisfying its
            conditions and its :meth:`Match.where` conditions.
        """
        pattern_value = self.assigned_variable
        pattern_holds = chained_logic(
            AND,
            *self.assigned_value._conditions_,
            *self.assigned_value._where_conditions_,
        )
        if element_quantifier is ElementQuantifier.EVERY:
            if self._states_an_element_of_the_attribute_:
                element = Variable(_domain_=self.attribute)
                return ForAll(
                    left=element,
                    right=Exists(
                        left=pattern_value,
                        right=AND(left=pattern_value == element, right=pattern_holds),
                    ),
                )
            # The one value of a single-valued attribute is every element there is.
            element_quantifier = ElementQuantifier.SOME
        return element_quantifier.condition_on_elements(
            element=pattern_value,
            condition=AND(
                left=self._build_condition_on_assigned_value_(), right=pattern_holds
            ),
        )

    def _resolve_pattern_against_(
        self, *, variable: CanBehaveLikeAVariable
    ) -> list[ConditionType]:
        """
        Resolve the assigned pattern with *variable* as the object it describes.

        :param variable: The variable the pattern describes.
        :return: The conditions of the pattern on *variable*.
        """
        self.assigned_value.resolve(variable=variable, parent=self)
        conditions = []
        quantifier_states_the_type = (
            self.assigned_value._element_quantifier_ is not ElementQuantifier.SOME
        )
        if self.is_type_filter_needed or quantifier_states_the_type:
            conditions.append(
                HasType(variable=variable, types_=self.assigned_value._type_)
            )
        conditions.extend(self.assigned_value._conditions_)
        return conditions

    def _build_condition_on_assigned_value_(self) -> SymbolicExpression:
        """
        :return: Membership of the assigned value in the attribute when it states one
            element of a collection-valued attribute, else equality.
        """
        if not self._states_an_element_of_the_attribute_:
            return self.attribute == self.assigned_variable
        membership = Comparator(
            left=self.attribute,
            right=self.assigned_variable,
            operation=operator.contains,
        )
        if self._collection_field_.is_optional:
            # An absent optional collection has no elements, so it contains nothing.
            return AND(left=self.attribute != None, right=membership)
        return membership

    @cached_property
    def _states_an_element_of_the_attribute_(self) -> bool:
        """
        :return: Whether the attribute holds a collection and the assigned value stands
            for one of its elements: a pattern, or a plain or symbolic value of the
            element type, any number standing for a number. An indexed attribute names
            one element already.
        :raises ValueFitsNeitherElementNorCollection: If a plain value is neither of the
            element type nor a collection.
        """
        if self.index_access is not None or not self._attribute_holds_a_collection_:
            return False
        value = self.assigned_value
        if isinstance(value, AbstractMatchExpression):
            return True
        element_type = self._collection_element_type_
        if is_union_annotation(element_type):
            # Elements of a union are instances of its members' nearest common base.
            element_type = common_base_class(list(get_args(element_type)))
        element_type = get_origin(element_type) or element_type
        # An element type that is no class, or ``Any``, cannot be checked against.
        element_type_is_known = isclass(element_type) and element_type is not Any
        if isinstance(value, SymbolicExpression):
            return (
                isclass(value._type_)
                and element_type_is_known
                and issubclass(value._type_, element_type)
            )
        if value is None or value is Ellipsis:
            return False
        is_a_collection = isinstance(value, (list, tuple, set, frozenset))
        if not element_type_is_known:
            return not is_a_collection
        if isinstance(value, element_type) or (
            issubclass(element_type, numbers.Number)
            and isinstance(value, numbers.Number)
        ):
            return True
        if is_a_collection:
            return False
        raise ValueFitsNeitherElementNorCollection(
            attribute_match=self, element_type=element_type
        )

    @property
    def _attribute_holds_a_collection_(self) -> bool:
        """
        :return: Whether the attribute is declared as a collection, such as a list, set,
            tuple or sequence of elements, or as an optional one.
        """
        return self._collection_field_ is not None

    @cached_property
    def _collection_field_(self) -> WrappedField | None:
        """
        :return: The attribute's field when it is declared as a collection or an
            optional collection, else None.
        """
        pattern = self._parent_
        if isinstance(pattern, AttributeMatch) and isinstance(
            pattern.assigned_value, AbstractMatchExpression
        ):
            # Nested in an attribute, the class is the one the pattern assigned to that
            # attribute names, as EndEffector in end_effector=an(EndEffector)(...).
            pattern = pattern.assigned_value
        pattern_class = pattern._type_
        wrapped_field = self._field_declared_below_(
            owner_class=pattern_class
        ) or self._field_declared_below_(owner_class=self.attribute._owner_class_)
        if (
            wrapped_field is None
            or self._declared_collection_(wrapped_field=wrapped_field) is None
        ):
            return None
        return wrapped_field

    def _field_declared_below_(self, *, owner_class: Any) -> WrappedField | None:
        """
        :param owner_class: The class the pattern is written for, or the class the
            attribute is read from.
        :return: The attribute's field as the class declares it or, for a class that
            leaves it to its subclasses, such as an abstract end effector whose concrete
            grippers declare their fingers, as the nearest subclass declares it.
        """
        if isinstance(owner_class, TypeVar):
            # A field declared with a type variable holds values of its bound.
            owner_class = owner_class.__bound__
        if not isclass(owner_class):
            return None
        classes_to_search = [owner_class]
        searched = set()
        while classes_to_search:
            candidate = classes_to_search.pop(0)
            if candidate in searched:
                continue
            searched.add(candidate)
            wrapped_field = get_wrapped_field(
                owner_class=candidate, field_name=self.attribute_name
            )
            if wrapped_field is not None:
                return wrapped_field
            classes_to_search.extend(candidate.__subclasses__())
        return None

    @cached_property
    def _collection_element_type_(self) -> type | None:
        """
        :return: The type of the elements of the collection-valued attribute.
        """
        element_types = get_args(
            self._declared_collection_(wrapped_field=self._collection_field_)
        )
        return element_types[0] if element_types else None

    @staticmethod
    def _declared_collection_(*, wrapped_field: WrappedField) -> Any | None:
        """
        :param wrapped_field: The field of an attribute.
        :return: The collection type the field is declared as, unwrapped from optional,
            such as ``set[Course]``, or None when it is not a collection.
        """
        declared_type = (
            wrapped_field.contained_type
            if wrapped_field.is_optional
            else wrapped_field.resolved_type
        )
        container_type = get_origin(declared_type)
        if (
            container_type in wrapped_field.container_types
            and container_type is not type
        ):
            return declared_type
        return None

    @cached_property
    def assigned_variable(self) -> SymbolicExpression:
        """
        :return: The symbolic variable representing the assigned value; a plain value
            stands for nothing symbolic and is wrapped in a literal of the attribute's
            type.
        """
        if not isinstance(self.assigned_value, MatchAssignedValue):
            return Literal(
                _name__=self._variable_._name_,
                _type_=self._type_,
                _value_=self.assigned_value,
            )
        return self.assigned_value._as_assigned_variable_(attribute_type=self._type_)

    @cached_property
    def attribute(self) -> Attribute:
        """
        :return: the attribute of the variable.
        :raises NoneWrappedFieldError: If the attribute does not have a WrappedField.
        """
        if self._variable_ is not None:
            return self._variable_

        attr: Attribute = getattr(self._parent_._variable_, self.attribute_name)
        if self.index_access is not None:
            attr = attr[self.index_access]
        self._variable_ = attr
        return attr

    @property
    def is_assigned_to_ellipsis(self) -> bool:
        """
        Whether the attribute is assigned ``...`` itself; a collection holding ``...``
        as an element does not count.
        """
        return self.assigned_value is ...

    @cached_property
    def is_type_filter_needed(self):
        """
        :return: True if a type filter condition is needed for the attribute assignment, else False.
        """
        attr_type = self._type_
        if not attr_type:
            return True
        if not isinstance(attr_type, type):
            # A type variable or a generic alias cannot be compared with issubclass, so
            # the value's type is checked when the query runs.
            return True
        return (
            self.assigned_value._type_ and self.assigned_value._type_ is not attr_type
        ) and issubclass(self.assigned_value._type_, attr_type)

    @property
    def _name_(self) -> str:
        return f"{self._parent_._name_}.{self.attribute_name}"

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_

    def construct_value_from_bindings(self, *, bindings: Bindings) -> Any:
        """
        :param bindings: The values of the pattern's variables, keyed by the identifier
            of the variable assigned to each attribute.
        :return: The value this attribute takes under the bindings: an instance
            constructed from them for a nested match, otherwise the value of the
            assigned variable evaluated under them.
        :raises UnboundPatternVariable: If the bindings do not bind the assigned
            variable and it states no value of its own.
        """
        if isinstance(self.assigned_value, Match):
            return self.assigned_value._construct_instance_from_bindings_(
                bindings=bindings
            )
        if (
            not isinstance(self.assigned_variable, Literal)
            and self.assigned_variable._id_ not in bindings
        ):
            raise UnboundPatternVariable(attribute_match=self)
        return next(
            self.assigned_variable._evaluate_(
                sources=OperationResult(bindings=bindings)
            )
        ).value

    @property
    def _stating_match_(self) -> Match:
        """
        :return: The match whose pattern states this attribute, and whose factory
            builds the object owning it.
        """
        if isinstance(self._parent_, AttributeMatch):
            return self._parent_.assigned_value
        return self._parent_

    @property
    def name_from_variable_access_path(self):
        """
        :return: The last name from the variables access path. This is similar to `self._name_` but without `Match`
        specific wrappings.
        """
        return self._variable_._access_path_[-1]._name_

    @property
    def _type_(self) -> type[T] | None:
        result = super()._type_
        if result is not None:
            return result

        if not isinstance(self._parent_, AttributeMatch):
            return None

        if isclass(self._parent_.assigned_value._factory_):
            return get_field_type_endpoint(
                self._parent_.assigned_value._type_, self._variable_._attribute_name_
            )
        else:
            return get_type_hints_of_object(self._parent_.assigned_value._factory_)[
                self._variable_._attribute_name_
            ]


def construct_graph_and_get_root(
    node_data: AbstractMatchExpression, graph: rx.PyDAG | None = None
) -> RWXNode:
    """
    Construct a graph representation of the match expression and return the root node.

    :param node_data: The root node of the match expression.
    :param graph: The graph to construct the subgraph in.
    :return: The root node of the constructed subgraph.
    """
    graph = graph or rx.PyDAG()
    node = RWXNode(name=node_data._name_, graph=graph, data=node_data)
    for child in node_data._children_:
        child_node = construct_graph_and_get_root(child, graph=graph)
        child_node.parent = node
    return node
