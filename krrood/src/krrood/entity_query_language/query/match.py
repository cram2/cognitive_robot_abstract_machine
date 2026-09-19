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

import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from inspect import ismethod, isfunction, isclass
from typing import assert_never, Any

import rustworkx as rx
from typing_extensions import (
    Callable,
    Optional,
    Type,
    List,
    Union,
    Generic,
    TYPE_CHECKING,
    Self,
    Iterator,
)

from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.entity_query_language.core.base_expressions import (
    HasExpression,
    Selectable,
    SymbolicExpression,
)
from krrood.entity_query_language.operators.causal import (
    Cause,
    CausesEffect,
    Confounder,
)
from krrood.entity_query_language.core.helpers import _resolve_domain
from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    FlatVariable,
    CanBehaveLikeAVariable,
    HasSymbolicOperations,
    MappedVariable,
    IndexByValue,
)
from krrood.entity_query_language.core.variable import Literal, DomainType, Variable
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.exceptions import (
    CalledMatchAfterResolution,
    CalledMatchMultipleTimes,
    MatchTypeCannotBeDetermined,
    PositionalArgumentsInMatchPattern,
    ReadOnlyMapping,
)
from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.query.quantifiers import An, ResultQuantifier
from krrood.entity_query_language.query.query_modifiers import HasQueryModifiers
from krrood.entity_query_language.utils import T
from krrood.patterns.factory_and_kwargs import HasFactoryAndKwargs
from krrood.rustworkx_utils.rxnode import RWXNode
from krrood.symbol_graph.helpers import get_field_type_endpoint

if TYPE_CHECKING:
    from krrood.entity_query_language.factories import ConditionType
    from krrood.entity_query_language.query.query import Entity, Query


@dataclass
class AbstractMatchExpression(Generic[T], ABC):
    """
    Abstract base class for constructing and handling a match expression.

    This class is intended to provide a framework for defining and managing match
    expressions, which are used to structural pattern matching in the form of nested
    match expressions with keyword arguments.
    """

    _declared_type_: Optional[Type[T]] = field(default=None, kw_only=True)
    """
    The matched type as explicitly given (or inferred from the factory); ``None`` until
    known.
    """

    _variable_: Optional[Variable[T]] = field(default=None, kw_only=True)
    """
    The created variable from the type and keyword arguments.
    """

    _conditions_: List[ConditionType] = field(init=False, default_factory=list)
    """
    The conditions that define the match.
    """

    _parent_: Optional[AbstractMatchExpression] = field(init=False, default=None)
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

    _children_: List[AttributeMatch] = field(init=False, default_factory=list)
    """
    The child matches of this match expression.
    """

    @property
    @abstractmethod
    def _symbolic_expression_(self) -> Union[CanBehaveLikeAVariable[T], T]:
        """
        :return: The expression this match expression stands for, on which every symbolic
            operation written on it is built.
        """
        ...

    def resolve(self, *args, **kwargs) -> Self:
        """
        Resolve the match by creating the variable and conditions expressions.
        """
        if self._resolved_:
            return self
        self._resolve(*args, **kwargs)
        self._resolved_ = True
        return self

    @abstractmethod
    def _resolve(self, *args, **kwargs):
        """
        This method serves as an abstract definition to be implemented by subclasses,
        aimed at handling specific resolution logic for the derived class.

        The method is designed to be flexible in accepting any number and type of input
        parameters through positional (*args) and keyword (**kwargs) arguments.
        Subclasses must extend this method to provide concrete implementations tailored
        to their unique behaviors and requirements.
        """
        ...

    @property
    @abstractmethod
    def _name_(self) -> str: ...

    @property
    def _type_(self) -> Optional[Type[T]]:
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
        :return: All attribute matches where the assigned variable is a variable.
        These matches are typically the leaves of a match expression.
        """
        self.resolve()
        for expression in self._descendants_:
            if isinstance(expression.assigned_variable, Variable):
                yield expression


@dataclass(eq=False)
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
        >>> drawer = a(Drawer)(body=a(Body)(name="drawer_1")).from_(world.views)

    A match reads like an instance of the matched class: every symbolic operation of
    :class:`~krrood.entity_query_language.core.mapped_variable.HasSymbolicOperations` -
    attribute access, indexing, calling, comparison and arithmetic - is built on
    :attr:`_symbolic_expression_`, so ``drawer.body`` is that query's ``body`` and
    carries the pattern. Names of the match's own methods (``where``, ``from_``,
    ``resolve``, ...) are resolved normally and never delegated; a matched-class field
    shadowed by one of them stays reachable through :attr:`_symbolic_expression_`.

    The one operation the match keeps for itself is the *first* call, which states the
    pattern; a matched class that is itself callable is called through a second pair of
    parentheses - see :meth:`__call__`.

    It also reads like a query: the modifiers of
    :class:`~krrood.entity_query_language.query.query_modifiers.HasQueryModifiers` narrow
    the lowered query and return the match, so a chain stays on the match.

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
    A list of all conditions that have been applied to this instance using the
    `where` method.
    """

    _has_been_called: bool = field(init=False, default=False)
    """
    Flag indicating whether the match instance has been called with keyword arguments.
    """

    _quantifier_type_: Type[ResultQuantifier] = field(init=False, default=An)
    """
    The result quantifier applied when this match is materialized into a runnable query.

    Defaults to ``An`` (zero or more results); set to ``The`` when built via
    ``the(...)``.
    """

    _domain_: Optional[DomainType] = field(default=None, init=False)
    """
    The instances the match ranges over.

    ``None`` constructs from scratch (an underspecified, generative request); a domain
    makes it a search over those existing instances.
    """

    def __post_init__(self):
        if self._declared_type_ is None:
            self._initialize_type_()

    def _initialize_type_(self):
        """
        Initialize the type of the match based on the provided information in- place.
        """
        if isclass(self._factory_):
            self._declared_type_ = self._factory_
        elif ismethod(self._factory_):
            self._declared_type_ = self._factory_.__class__
        elif isfunction(self._factory_):
            type_ = get_type_hints_of_object(self._factory_).get("return")
            if type_ is None or not isclass(type_):
                raise MatchTypeCannotBeDetermined(self)
            self._declared_type_ = type_
        else:
            assert_never(self._factory_)

    def __call__(self, *args: Any, **kwargs: Any) -> Union[T, Self]:
        """
        Set the pattern the match looks for, the first time; call the matched instance
        symbolically, every time after that.

        The first parentheses after ``a(Drawer)`` state the pattern, so a matched class
        that is itself callable is called through a second pair - ``a(Adder)(offset=1)(2)``,
        or ``a(Adder)()(2)`` where the pattern is empty. The pattern parentheses take
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
                    CalledMatchAfterResolution(self)
                    if self._resolved_
                    else CalledMatchMultipleTimes(self)
                )
            return HasSymbolicOperations.__call__(self, *args, **kwargs)
        if args:
            raise PositionalArgumentsInMatchPattern(self, args)
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

    def _is_own_name_(self, name: str) -> bool:
        """
        :param name: A name that this match does not define.
        :return: Whether the name belongs to the match machinery, which keeps its own
            state behind underscore-prefixed names, leaving every other name to the
            matched class.
        """
        return name.startswith("_")

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
        from krrood.entity_query_language.factories import entity

        if self._expression is not None:
            return self._expression

        if not self._resolved_:
            self.resolve()
        entity_ = entity(self._variable_)
        if self._conditions_:
            entity_ = entity_.where(*self._conditions_)
        entity_._quantify_(self._quantifier_type_)
        self._expression = entity_
        return entity_

    def _get_expression_(self) -> SymbolicExpression:
        return self._symbolic_expression_

    def _resolve(
        self,
        variable: Optional[Selectable] = None,
        parent: Optional[Match] = None,
    ):
        """
        Resolve the match by creating the variable and conditions expressions in-place.

        :param variable: An optional pre-existing variable to use for the match; if not
            provided, a new variable will be created.
        :param parent: The parent match if this is a nested match.
        """
        parent = parent or self
        self._update_fields_(variable, parent)
        for attr_name, attr_assigned_value in self._kwargs_.items():
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
        parent: Match,
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
            _parent_=parent,
            attribute_name=attribute_name,
            index_access=index_access,
            assigned_value=assigned_value,
        )
        attr_match.resolve()
        self._children_.append(attr_match)
        self._conditions_.extend(attr_match._conditions_)
        return attr_match

    def _resolve_list_like_value(
        self, key: str, value: Union[list, tuple], parent: Match
    ):
        """
        Resolves list-like values by iterating over their elements and creating
        attribute matches for the parent match variable.

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

    def _update_fields_(
        self,
        variable: Optional[Selectable] = None,
        parent: Optional[AbstractMatchExpression] = None,
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

        self._variable_._update_domain_(_resolve_domain(self._type_, self._domain_))

    def _evaluate_natively_(self) -> Iterator:
        """
        Evaluate the match selectively in the current python process: select elements
        from the match's domain (its variable's domain, or the ``SymbolGraph`` for
        ``Symbol`` types when no domain was given) that satisfy the structural pattern
        and ``where`` conditions.

        .. note::
            Constructing *new* instances from an underspecified match is the job of a
            :class:`~krrood.entity_query_language.backends.GenerativeBackend` (for example
            :class:`~krrood.entity_query_language.backends.EntityQueryLanguageGenerativeBackend`
            or :class:`~krrood.entity_query_language.backends.ProbabilisticBackend`), not of the
            default selective evaluation.

        :return: An iterator over the matching elements.
        """
        return self._symbolic_expression_._evaluate_natively_()

    @property
    def _has_ellipsis_attributes_(self) -> bool:
        """
        :return: Whether any attribute anywhere in this match's pattern (including nested
            matches, and ``...`` elements inside an otherwise-concrete list/tuple attribute)
            is left fully unspecified via ``...``, requiring construction rather than search
            to resolve.
        """
        return any(
            self._is_or_contains_ellipsis(attribute_match.assigned_value)
            for attribute_match in self._matches_with_variables_
        )

    @staticmethod
    def _is_or_contains_ellipsis(value: Any) -> bool:
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
        variable: Union[Selectable[T], Any],
        descending: bool = False,
        key: Optional[Callable] = None,
    ) -> Self:
        """
        Order the matched instances by the given expression.

        :param variable: The expression to order by.
        :param descending: Whether to order the results in descending order.
        :param key: A function to extract the key from the expression's value.
        :return: This match.
        """
        self._symbolic_expression_.ordered_by(variable, descending=descending, key=key)
        return self

    def distinct(self, *on: Union[Selectable, Any]) -> Self:
        """
        Keep only matched instances that differ in the given expressions.

        :param on: The expressions the results must differ in; the matched instance by
            default.
        :return: This match.
        """
        self._symbolic_expression_.distinct(*on)
        return self

    def grouped_by(self, *variables_to_group_by: Union[Selectable, Any]) -> Self:
        """
        Group the matched instances by the given expressions.

        :param variables_to_group_by: The expressions to group the results by.
        :return: This match.
        """
        self._symbolic_expression_.grouped_by(*variables_to_group_by)
        return self

    def limit(self, n: int) -> Self:
        """
        Return at most ``n`` matched instances.

        :param n: The maximum number of results to return.
        :return: This match.
        """
        self._symbolic_expression_.limit(n)
        return self

    def causes_effect(self, *conditions: ConditionType) -> Match[T]:
        """
        Mark condition(s) as the effect side of a causal query, e.g.
        ``a(Pick)(arm=cause).causes_effect(pick.action.status == SUCCESS)``.

        Sugar for ``self.where(CausesEffect(and_(*conditions), cause_attributes=...))``:
        semantically identical to an ordinary ``.where()`` under every backend except
        :class:`~krrood.entity_query_language.backends.ProbabilisticBackend`, which reads
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
            CausesEffect(and_(*conditions), cause_attributes=cause_attributes)
        )

    def from_(self, domain: DomainType) -> Self:
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

        :param domain: The instances the match ranges over.
        :return: This match, for chaining.
        """
        self._domain_ = domain
        self._create_or_update_variable_()
        return self

    def _update_kwargs_from_literal_values(self):
        """
        Update the kwargs dictionary with values from this statements leaves.
        """
        for attribute_match in self._matches_with_variables_:
            attribute_match._update_kwargs_from(self)

    def _get_mapped_variable_by_name(self, name: str) -> Optional[MappedVariable]:
        """
        Get a mapped variable by its name in the path.

        :param name: The name
        :return: The mapped variable
        """
        result = [
            attribute_match.assigned_variable
            for attribute_match in self._matches_with_variables_
            if attribute_match.name_from_variable_access_path == name
        ]
        if len(result) == 0:
            return None
        elif len(result) == 1:
            return result[0]
        else:
            raise KeyError(f"Multiple variables with name {name}")


@dataclass(eq=False)
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

    index_access: Optional[Any] = None
    """
    The index  that is accessed.

    Is not None if the attribute is an indexable object.
    """

    assigned_value: Optional[Union[Literal, Match]] = None
    """
    The value to assign to the attribute, which can be a Match instance or a Literal.
    """

    _variable_: Union[Attribute, FlatVariable] = field(default=None, kw_only=True)
    """
    The symbolic variable representing the attribute.
    """

    def __post_init__(self):
        if isinstance(self.assigned_value, Match):
            self._children_ = self.assigned_value._children_

    @cached_property
    def _symbolic_expression_(self) -> Union[CanBehaveLikeAVariable[T], T]:
        """
        :return: The symbolic attribute this match stands for, which is the variable it
            assigns the matched value to.
        """
        if not self._variable_:
            self.resolve()
        return self._variable_

    def _resolve(self):
        """
        Resolve the attribute assignment by creating the conditions and applying the
        necessary mappings to the attribute.
        """
        if (
            not isinstance(self.assigned_value, AbstractMatchExpression)
            or self.assigned_value._resolved_
        ):
            self._conditions_.append(self.attribute == self.assigned_variable)
            return

        self.assigned_value.resolve(self.attribute, self)

        if self.is_type_filter_needed:
            self._conditions_.append(
                HasType(self.attribute, self.assigned_value._type_)
            )

        self._conditions_.extend(self.assigned_value._conditions_)

    @cached_property
    def assigned_variable(self) -> Selectable:
        """
        :return: The symbolic variable representing the assigned value.
        """
        if isinstance(self.assigned_value, AbstractMatchExpression):
            return self.assigned_value._variable_
        if (
            isinstance(self.assigned_value, (Cause, Confounder))
            and self.assigned_value._type_ is None
        ):
            # `cause`/`confounder` are shared instances written directly into every
            # matching kwarg, so unlike a plain literal (whose `Literal` wrapper is
            # created fresh right here, with `_type_=self._type_`), an unresolved one has
            # no declared type of its own yet, and mutating it in place would corrupt
            # every other field also marked `cause`/`confounder`. Return a fresh,
            # per-attribute copy with the type filled in instead, so code reading
            # `assigned_variable._type_` (parametrization, generation) sees the
            # attribute's declared type without touching the shared original.
            return type(self.assigned_value)(_type_=self._type_)
        elif not isinstance(self.assigned_value, SymbolicExpression):
            return Literal(
                _name__=self._variable_._name_,
                _type_=self._type_,
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
        if self._variable_ is not None:
            return self._variable_

        attr: Attribute = getattr(self._parent_._variable_, self.attribute_name)
        if self.index_access is not None:
            attr = attr[self.index_access]
        self._variable_ = attr
        return attr

    @cached_property
    def is_type_filter_needed(self):
        """
        :return: True if a type filter condition is needed for the attribute assignment, else False.
        """
        attr_type = self._type_
        return (not attr_type) or (
            (self.assigned_value._type_ and self.assigned_value._type_ is not attr_type)
            and issubclass(self.assigned_value._type_, attr_type)
        )

    @property
    def _name_(self) -> str:
        return f"{self._parent_._name_}.{self.attribute_name}"

    def __repr__(self):
        return self._name_

    def __str__(self):
        return self._name_

    def _update_kwargs_from(self, match: Match[T]):
        """
        Update the kwargs of the parent match with the values of the assigned variable.

        Only works if this is a variable assignment.
        """
        current_value = match
        for step in self._variable_._access_path_[:-1]:
            if isinstance(step, Attribute):
                current_value = current_value._kwargs_[step._attribute_name_]
            elif isinstance(step, IndexByValue):
                current_value = current_value[step._key_]
            else:
                raise ReadOnlyMapping(step)

        final_step = self._variable_._access_path_[-1]

        if isinstance(final_step, Attribute):
            current_value._kwargs_[final_step._attribute_name_] = (
                self.assigned_variable._value_
            )
        else:
            final_step._set_child_instance_value_(
                current_value, self.assigned_variable._value_
            )

    @property
    def name_from_variable_access_path(self):
        """
        :return: The last name from the variables access path. This is similar to `self._name_` but without `Match`
        specific wrappings.
        """
        return self._variable_._access_path_[-1]._name_

    @property
    def _type_(self) -> Optional[Type[T]]:
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
    node_data: AbstractMatchExpression, graph: Optional[rx.PyDAG] = None
) -> RWXNode:
    """
    Construct a graph representation of the match expression and return the root node.

    :param node_data: The root node of the match expression.
    :param graph: The graph to construct the subgraph in.
    :return: The root node of the constructed subgraph.
    """
    graph = graph or rx.PyDAG()
    node = RWXNode(node_data._name_, graph, data=node_data)
    for child in node_data._children_:
        child_node = construct_graph_and_get_root(child, graph=graph)
        child_node.parent = node
    return node
