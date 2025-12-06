from __future__ import annotations

from typing import Optional, Union, Type

from .utils import T
from .match import EntityType, Match
from .result_quantification_constraint import (
    ResultQuantificationConstraint,
)
from .symbolic import An, The, ResultQuantifier


def an(
    entity_: EntityType,
    quantification: Optional[ResultQuantificationConstraint] = None,
) -> Union[An[T], T]:
    """
    Select a single element satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :param quantification: Optional quantification constraint.
    :return: A quantifier representing "an" element.
    :rtype: An[T]
    """
    return _quantify_entity(An, entity_, _quantification_constraint_=quantification)


a = an
"""
This is an alias to accommodate for words not starting with vowels.
"""


def the(
    entity_: EntityType,
) -> Union[The[T], T]:
    """
    Select the unique element satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :return: A quantifier representing "an" element.
    :rtype: The[T]
    """
    return _quantify_entity(The, entity_)


def _quantify_entity(
    quantifier: Type[ResultQuantifier], entity_: EntityType, **quantifier_kwargs
) -> Union[ResultQuantifier[T], T]:
    """
    Apply the given quantifier to the given entity.

    :param quantifier: The quantifier to apply.
    :param entity_: The entity to quantify.
    :param quantifier_kwargs: Keyword arguments to pass to the quantifier.
    :return: The quantified entity.
    """
    if isinstance(entity_, Match) and not entity_.variable:
        entity_ = entity_.expression
    return quantifier(entity_, **quantifier_kwargs)
