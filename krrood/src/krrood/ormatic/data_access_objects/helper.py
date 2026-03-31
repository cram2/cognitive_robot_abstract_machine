from __future__ import annotations

from functools import lru_cache
from typing import Type, Optional, Any, TYPE_CHECKING
from typing_extensions import get_origin


from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping

from krrood.ormatic.exceptions import NoGenericError, NoDAOFoundError
from krrood.utils import recursive_subclasses

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import DataAccessObject
    from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState


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
            if subclass.original_class() == original_clazz:
                return subclass
        except (AttributeError, TypeError, NoGenericError):
            continue
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
        return dao

    # Fallback to the expected type if provided.
    if expected_type is not None:
        dao = _get_clazz_by_original_clazz(DataAccessObject, expected_type)
        if dao is not None:
            return dao

    return None


@lru_cache(maxsize=None)
def get_alternative_mapping(
    original_clazz: Type,
) -> Optional[Type[AlternativeMapping]]:
    """
    Retrieve the alternative mapping for a domain class.

    :param original_clazz: The domain class.
    :return: The corresponding alternative mapping or None.
    """
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
