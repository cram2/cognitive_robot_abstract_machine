from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing_extensions import Callable, Any, ClassVar, TypeVar

from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.core.variable import Variable as SymbolicVariable
from krrood.parametrization.feature_extraction.exceptions import (
    InvalidAggregatedFieldError,
    MissingFieldNameError,
    OutOfDomainValueError,
)
from krrood.patterns.role import Role
from random_events.variable import Variable, Symbolic
from krrood.entity_query_language.factories import variable
from krrood.utils import recursive_subclasses, get_generic_type_parameters

logger = logging.getLogger(__name__)

TAggregatedObject = TypeVar("TAggregatedObject", bound=object)
"""
The type of the domain object an :class:`AggregationStatistic` is a role of.
"""


@dataclass
class _AggregationStatisticDescriptor:
    """
    Descriptor returned by :func:`aggregation_statistic`.

    When Python constructs the enclosing class, it calls :meth:`__set_name__` with the
    owner class and the attribute name. At that point the function is registered and the
    descriptor replaces itself with the bare function so normal method-call semantics
    are preserved.
    """

    field_name: str
    """
    The exchangeable-part field this statistic aggregates over.
    """

    func: Callable
    """
    The wrapped statistic method.
    """

    def __set_name__(self, owner: type[AggregationStatistic], name: str) -> None:
        if "aggregation_registry" not in owner.__dict__:
            owner.aggregation_registry = defaultdict(list)
        owner.aggregation_registry[self.field_name].append(self.func)
        setattr(owner, name, self.func)


def aggregation_statistic(
    aggregated_field: Attribute,
) -> Callable[[Callable], _AggregationStatisticDescriptor]:
    """
    Marks a method as an aggregation statistic for an exchangeable-part field.

    The containing class is discovered automatically via
    ``__set_name__`` — there is no need to pass it explicitly.

    Example: ``@aggregation_statistic(variable(Shelf).layers)`` marks a statistic over
    the ``layers`` of a ``Shelf``.

    :param aggregated_field: A symbolic attribute of a variable, naming the field this
        statistic aggregates for.
    :raises InvalidAggregatedFieldError: If ``aggregated_field`` is not a direct
        attribute of a variable.
    """
    if not (
        isinstance(aggregated_field, Attribute)
        and isinstance(aggregated_field._child_, SymbolicVariable)
    ):
        raise InvalidAggregatedFieldError(aggregated_field=aggregated_field)

    def decorator(func: Callable) -> _AggregationStatisticDescriptor:
        return _AggregationStatisticDescriptor(
            field_name=aggregated_field._attribute_name_, func=func
        )

    return decorator


@lru_cache(maxsize=None)
def get_aggregation_class(owner: type) -> type[AggregationStatistic] | None:
    """
    Returns the most specific :class:`AggregationStatistic` subclass for ``owner``.

    Walks the MRO of ``owner`` from most specific to least specific, returning the first
    subclass of :class:`AggregationStatistic` whose role taker type matches that ancestor.
    This means that if ``B`` extends ``A`` and only ``AggregationStatistic[A]`` exists,
    a lookup for ``B`` will return it.

    :param owner: The domain class to look up.
    :return: The most specific matching subclass, or ``None`` if none has been defined.
    """
    subclasses = list(recursive_subclasses(AggregationStatistic))
    for ancestor in owner.__mro__:
        for subclass in subclasses:
            bound = get_generic_type_parameters(subclass, AggregationStatistic)
            if bound and bound[0] == ancestor:
                return subclass
    return None


@dataclass(eq=False)
class AggregationStatistic(Role[TAggregatedObject]):
    """
    Base class for aggregation statistics over a domain object's exchangeable-
    part fields.

    An aggregation statistic is a :class:`~krrood.patterns.role.Role` of the domain
    object it describes: it adds statistics to the object without changing its type or
    identity, and the object's own attributes are readable through it.

    Subclasses bind ``TAggregatedObject`` to the type of the role taker and declare one
    or more methods, each annotated with :func:`aggregation_statistic`.  Discovery
    happens automatically via :func:`get_aggregation_class` — no explicit registration
    is required.

    Set :attr:`field_name` to scope :attr:`aggregation_features` and related methods to
    a single exchangeable-part field.

    .. note::
        Each owner class may have at most one ``AggregationStatistic`` subclass, which must
        handle all of its exchangeable-part fields.  Shared logic across owner types should
        be extracted to an intermediate abstract subclass whose concrete children each bind
        their own role taker type.  The intermediate class declares its own bounded
        TypeVar ``S = TypeVar("S", bound=Base)`` and inherits from
        ``AggregationStatistic[S]``; when all owner types share a common supertype
        ``Base``, this lets it access ``self.role_taker`` with the full interface of
        ``Base`` without losing type safety in the concrete subclasses.
    """

    field_name: str | None = None
    """
    The exchangeable-part field this instance is scoped to.

    Must be set before accessing :attr:`aggregation_features`.
    """

    aggregation_registry: ClassVar[dict[str, list[Callable]]] = defaultdict(list)
    """
    Methods marked with :func:`aggregation_statistic` that are defined directly on this
    class.

    Keys are field names; values are the registered callables.  Each subclass receives
    its own registry via :meth:`__init_subclass__` so registrations never bleed across
    unrelated branches of the hierarchy.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "aggregation_registry" not in cls.__dict__:
            cls.aggregation_registry = defaultdict(list)

    @classmethod
    def _registered_names_for_field(cls, field_name: str) -> set[str]:
        """
        Collect the names of all statistics registered for ``field_name`` across the MRO
        so inherited registrations are included.

        :param field_name: The exchangeable-part field to look up.
        :return: Names of all callables registered under ``field_name``.
        """
        names: set[str] = set()
        for ancestor in cls.__mro__:
            if not issubclass(ancestor, AggregationStatistic):
                continue
            for func in ancestor.aggregation_registry[field_name]:
                names.add(func.__name__)
        return names

    @classmethod
    def aggregation_features_of_field(cls, field_name: str) -> list[Callable]:
        """
        All methods on this class marked with :func:`aggregation_statistic` for
        ``field_name``.

        :param field_name: The exchangeable-part field to look up.
        :return: The marked callable methods for the given field, sorted alphabetically
            by name.
        """
        registered = cls._registered_names_for_field(field_name)
        return [
            func
            for _, func in inspect.getmembers(cls, predicate=inspect.isfunction)
            if func.__name__ in registered
        ]

    @classmethod
    def symbolic_features_of_field(cls, field_name: str) -> list[MappedVariable]:
        """
        Symbolic variables for the statistic methods that aggregate ``field_name``.

        Requires no domain object, so the features of an already fitted model can be
        recreated from the class alone.

        :param field_name: The exchangeable-part field to look up.
        :return: One :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`
            per matching statistic method, in alphabetical order.
        """
        aggregation_variable = variable(cls, [])
        return [
            getattr(aggregation_variable, func.__name__)()
            for func in cls.aggregation_features_of_field(field_name)
        ]

    @property
    def aggregation_features(self) -> list[Callable]:
        """
        All methods on this class marked with :func:`aggregation_statistic` for
        :attr:`field_name`.

        :return: The marked callable methods for the scoped field, sorted alphabetically
            by name.
        :raises MissingFieldNameError: If :attr:`field_name` was not provided.
        """
        if self.field_name is None:
            raise MissingFieldNameError()
        return type(self).aggregation_features_of_field(self.field_name)

    def symbolic_aggregation_features(self) -> list[MappedVariable]:
        """
        Symbolic variables for statistic methods that aggregate
        :attr:`field_name`.

        :return: One :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`
            per matching statistic method, in alphabetical order.
        :raises MissingFieldNameError: If :attr:`field_name` was not provided.
        """
        if self.field_name is None:
            raise MissingFieldNameError()
        return type(self).symbolic_features_of_field(self.field_name)

    def apply_mapping(self) -> dict[str, Any]:
        """
        Evaluates every statistic for :attr:`field_name` against this instance.

        :return: A mapping from each statistic method name to its computed value.
        """
        return {
            func.__name__: feature.apply_mapping_on_external_root(self)
            for func, feature in zip(
                self.aggregation_features, self.symbolic_aggregation_features()
            )
        }


def compute_aggregation_statistics(
    domain_object,
    field_name: str,
    latent_variables: list[Variable],
) -> dict[Variable, Any]:
    """
    Evaluate the aggregation statistics of one exchangeable-part field against a domain
    object and map the results to latent variables.

    Each statistic is evaluated only if its name matches a latent variable.

    :param domain_object: The domain object whose aggregation statistics are computed.
    :param field_name: The exchangeable-part field whose statistics are evaluated.
    :param latent_variables: Latent variables that define which statistics are relevant.
    :return: A mapping from matched latent variables to their observed values.
    """
    latent_variable_by_name = {
        latent_variable.name: latent_variable for latent_variable in latent_variables
    }
    aggregation_class = get_aggregation_class(type(domain_object))
    if aggregation_class is None:
        return {}
    aggregation_instance = aggregation_class(role_taker=domain_object)
    statistics = {}
    for feature_function in aggregation_class.symbolic_features_of_field(field_name):
        feature_name = feature_function._name_
        if feature_name not in latent_variable_by_name:
            continue
        value = feature_function.apply_mapping_on_external_root(aggregation_instance)
        latent_variable = latent_variable_by_name[feature_name]
        try:
            latent_variable.make_value(value)
        except (ValueError, TypeError):
            logger.info(
                "Could not determine value for the aggregation statistic. Falling back to Monte-Carlo integration",
            )
        statistics[latent_variable] = value
    return statistics
