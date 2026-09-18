from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Type, Optional, Any, TYPE_CHECKING
from typing_extensions import get_origin


from krrood.ormatic.exceptions import NoGenericError, NoDAOFoundError
from krrood.utils import recursive_subclasses

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.alternative_mappings import (
        AlternativeMapping,
    )
    from krrood.ormatic.data_access_objects.dao import DataAccessObject
    from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState


def _same_domain_class(candidate: Type, original_clazz: Type) -> bool:
    """
    Whether *candidate* is the domain class a DAO subclass was mapped against, allowing
    for one of the two to have been loaded as a process's entry point.

    A module run as a script's or ``-m``'s entry point executes under the name
    ``__main__``. The ``spawn`` multiprocessing start method then re-executes that same
    module in every worker it starts, under the name ``__mp_main__`` -- a synthetic name
    CPython uses precisely to keep this reload from colliding with a worker's own,
    separate ``__main__``. Either way, a class the module defines is a distinct object
    there from the one obtained by importing the module normally elsewhere in the same
    process, even though both come from the same class body. Plain identity comparison
    would then treat whichever one is not *original_clazz* as an unmapped class, so this
    falls back to matching by defining file and qualified name -- but only when one side
    was loaded under one of those two entry-point names, so two classes that merely
    share a name in genuinely different modules are never conflated.

    :param candidate: A candidate class found while searching DAO subclasses.
    :param original_clazz: The domain class being resolved.
    :return: Whether they are the same domain class.
    """
    if candidate == original_clazz:
        return True
    if not {"__main__", "__mp_main__"} & {
        candidate.__module__,
        original_clazz.__module__,
    }:
        return False
    if candidate.__qualname__ != original_clazz.__qualname__:
        return False
    try:
        return inspect.getfile(candidate) == inspect.getfile(original_clazz)
    except TypeError:
        return False


@lru_cache(maxsize=None)
def _get_clazz_by_original_clazz(
    base_clazz: Type, original_clazz: Type
) -> Optional[Type]:
    """
    Find a subclass that maps to a specific domain class.

    :param base_clazz: The base class to search from.
    :param original_clazz: The domain class to match.
    :return: The matching subclass or None.
    """
    for subclass in recursive_subclasses(base_clazz):
        try:
            if _same_domain_class(subclass.original_class(), original_clazz):
                return subclass
        except (AttributeError, TypeError, NoGenericError):
            continue
    return None


def _maps_parametrization_of(dao_clazz: Type, generic_clazz: Type) -> bool:
    """
    Check whether a DAO class maps a parametrization of a bare generic class.

    :param dao_clazz: The candidate DAO class.
    :param generic_clazz: The bare generic domain class.
    :return: True if the DAO maps a parametrization (e.g. ``C[float]``) of
        ``generic_clazz``.
    """
    try:
        mapped_class = dao_clazz.original_class()
    except (AttributeError, TypeError, NoGenericError):
        return False
    return get_origin(mapped_class) is generic_clazz


@lru_cache(maxsize=None)
def _get_concrete_generic_subclass(
    base_dao: Type[DataAccessObject], original_clazz: Type
) -> Optional[Type[DataAccessObject]]:
    """
    Find the unique concrete DAO subclass for a bare generic domain class.

    A bare generic domain class (for example ``DerivativeMap``) maps to an empty
    polymorphic base DAO, while its parametrizations (for example
    ``DerivativeMap[float]``) map to concrete data-bearing leaf DAOs. A runtime instance
    of the bare generic carries no type argument, so it must be persisted through such a
    concrete leaf.

    :param base_dao: The DAO resolved for the bare generic class.
    :param original_clazz: The bare generic domain class.
    :return: The unique concrete leaf subclass DAO, or None when ``original_clazz`` is
        not a bare generic, ``base_dao`` already maps a parametrization, or the leaf is
        ambiguous.
    """
    if not getattr(original_clazz, "__parameters__", ()):
        return None
    if get_origin(base_dao.original_class()) is not None:
        return None

    parametrized_subclasses = [
        subclass
        for subclass in recursive_subclasses(base_dao)
        if _maps_parametrization_of(subclass, original_clazz)
    ]
    leaf_subclasses = [
        subclass
        for subclass in parametrized_subclasses
        if not any(
            other is not subclass and issubclass(other, subclass)
            for other in parametrized_subclasses
        )
    ]
    if len(leaf_subclasses) == 1:
        return leaf_subclasses[0]
    return None


@lru_cache(maxsize=None)
def get_dao_class(
    original_clazz: Type, expected_type: Optional[Type] = None
) -> Optional[Type[DataAccessObject]]:
    """
    Retrieve the DAO class for a domain class.

    :param original_clazz: The domain class.
    :param expected_type: The expected domain type (from relationship).
    :return: The corresponding DAO class or None.
    """
    from krrood.ormatic.data_access_objects.dao import DataAccessObject

    if issubclass(original_clazz, DataAccessObject):
        return original_clazz

    alternative_mapping = get_alternative_mapping(original_clazz)
    if alternative_mapping is not None:
        original_clazz = alternative_mapping

    # If the actual class is the same as the origin of the expected type,
    # the expected type is more specific (likely a parametrized generic)
    # and we should prefer it.
    if expected_type is not None and original_clazz == get_origin(expected_type):
        dao = _get_clazz_by_original_clazz(DataAccessObject, expected_type)
        if dao is not None:
            return dao

    # Try the actual class first.
    # This is important for polymorphic inheritance to get the most specific DAO.
    dao = _get_clazz_by_original_clazz(DataAccessObject, original_clazz)
    if dao is not None:
        concrete_dao = _get_concrete_generic_subclass(dao, original_clazz)
        return concrete_dao if concrete_dao is not None else dao

    # Fallback to the expected type if provided.
    if expected_type is not None:
        dao = _get_clazz_by_original_clazz(DataAccessObject, expected_type)
        if dao is not None:
            return dao

    return None


def clear_dao_lookup_caches() -> None:
    """
    Clear all caches that map domain classes to DAO classes.

    This has to be called whenever a new DataAccessObject or AlternativeMapping subclass
    is created, since previously failed lookups (cached as None) would otherwise stay
    stale forever.
    """
    _get_clazz_by_original_clazz.cache_clear()
    _get_concrete_generic_subclass.cache_clear()
    get_dao_class.cache_clear()
    get_alternative_mapping.cache_clear()


@lru_cache(maxsize=None)
def get_alternative_mapping(
    original_clazz: Type,
) -> Optional[Type[AlternativeMapping]]:
    """
    Retrieve the alternative mapping for a domain class.

    :param original_clazz: The domain class.
    :return: The corresponding alternative mapping or None.
    """
    from krrood.ormatic.data_access_objects.alternative_mappings import (
        AlternativeMapping,
    )

    return _get_clazz_by_original_clazz(AlternativeMapping, original_clazz)


def to_dao(
    source_object: Any, state: Optional[ToDataAccessObjectState] = None
) -> DataAccessObject:
    """
    Convert an object to its corresponding DAO.

    :param source_object: The object to convert.
    :param state: The conversion state.
    :return: The converted DAO instance.
    """
    from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState

    dao_clazz = get_dao_class(type(source_object))
    if dao_clazz is None:
        raise NoDAOFoundError(source_object)
    state = state or ToDataAccessObjectState()
    return dao_clazz.to_dao(source_object, state)
