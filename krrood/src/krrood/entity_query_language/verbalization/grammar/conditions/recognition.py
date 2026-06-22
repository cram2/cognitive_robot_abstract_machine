from __future__ import annotations

import dataclasses
import operator
from dataclasses import dataclass

from typing_extensions import List, Optional

from krrood.symbol_graph.symbol_graph import Symbol
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.core.variable import Literal, Variable
from krrood.entity_query_language.operators.aggregators import Aggregator, Extreme
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.core.expression_structure import (
    chain_ends_in_boolean_attribute,
    chain_root,
    walk_chain,
)
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.subquery import (
    aggregation_source_root,
    is_collapsible_aggregation_subquery,
    selected_aggregator,
    unwrap_result_quantifiers,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    ENGLISH_PREPOSITIONS,
)

#: The prepositions that, as the final token of a snake_case attribute name, mark a *relational*
#: field (*assigned_to*, *owned_by*, *located_in*): the English preposition inventory from the
#: lexicon minus ``of`` — ``<noun>_of`` is a genitive (``number_of``, ``type_of``), never a relation.
_RELATIONAL_PREPOSITIONS = ENGLISH_PREPOSITIONS - {"of"}


@dataclass(frozen=True)
class RelationVerb:
    """A relational field's verb, split into its parts — so callers can either strand the
    preposition (*"assigned to"*, used by absence) or pied-pipe it (*"to which … is assigned"*,
    used by relational navigation)."""

    participle: str
    """The past-participle verb word(s) (*"assigned"*, *"cross referenced"*)."""

    preposition: str
    """The trailing preposition (*"to"*, *"by"*, *"with"*)."""

    @property
    def phrase(self) -> str:
        """:return: The stranded phrase *"<participle> <preposition>"* (*"assigned to"*)."""
        return f"{self.participle} {self.preposition}"


def relational_verb(attribute_name: str) -> Optional[RelationVerb]:
    """
    :param attribute_name: An attribute identifier (snake_case).
    :return: The relation's verb, split into participle + preposition, when *attribute_name* names a
        relation as *past-participle + preposition* (*assigned_to* → *"assigned"* + *"to"*); else
        ``None`` for a plain noun attribute.

    The participle check (:func:`morphology.is_past_participle`) is what distinguishes a relation
    from a noun that merely ends in a preposition: *color_in* / *price_at* / *name_to* are rejected
    because *color* / *price* / *name* are not participles.
    """
    tokens = attribute_name.split("_")
    if len(tokens) < 2 or tokens[-1] not in _RELATIONAL_PREPOSITIONS:
        return None
    if not morphology.is_past_participle(tokens[-2]):
        return None
    return RelationVerb(participle=" ".join(tokens[:-1]), preposition=tokens[-1])


def relational_verb_phrase(attribute_name: str) -> Optional[str]:
    """
    :param attribute_name: An attribute identifier (snake_case).
    :return: The humanized stranded verb phrase (*"assigned to"*) for a relational field, else
        ``None`` — the form absence uses (*"has not been assigned to any Robot"*). Thin wrapper over
        :func:`relational_verb` so the recognition lives in one place.
    """
    verb = relational_verb(attribute_name)
    return verb.phrase if verb is not None else None


def attribute_names(left: SymbolicExpression) -> List[str]:
    """
    :param left: A ``MappedVariable`` chain (or any expression).
    :return: The attribute names along the chain, outermost last (``[]`` if none).
    """
    names: List[str] = []
    current = left
    while isinstance(current, MappedVariable):
        if isinstance(current, Attribute):
            names.append(current._attribute_name_)
        current = current._child_
    return names


def single_hop_attribute(
    expression: SymbolicExpression, subject_variable: Optional[Variable]
) -> Optional[Attribute]:
    """
    :param expression: Candidate expression.
    :param subject_variable: The variable the attribute must be on.
    :return: The attribute node when *expression* is exactly ``subject_variable.<attribute>``, else
        ``None``.
    """
    if subject_variable is None or not isinstance(expression, MappedVariable):
        return None
    chain, root = walk_chain(expression)
    if not (isinstance(root, Variable) and root._id_ == subject_variable._id_):
        return None
    if len(chain) != 1 or not isinstance(chain[0], Attribute):
        return None
    return chain[0]


def is_none_literal(expression: SymbolicExpression) -> bool:
    """
    :param expression: Candidate expression (typically a comparator's right side).
    :return: ``True`` when *expression* is a literal whose value is ``None`` — the marker for an
        absence comparison (``<chain> == None``) rendered as *"has no <attribute>"* / *"does not
        exist"* rather than a value predicate.
    """
    return isinstance(expression, Literal) and expression._value_ is None


def is_concrete_object_literal(expression: SymbolicExpression) -> bool:
    """
    :param expression: Candidate expression.
    :return: ``True`` when *expression* is a literal holding a concrete domain-object instance (a
        dataclass or :class:`Symbol` instance) — rendered by identity (*"a specific Body"*) rather
        than its (possibly huge) ``repr``. A class object, primitive, ``enum`` member, ``datetime``,
        or ``None`` is not.
    """
    if not isinstance(expression, Literal):
        return False
    value = expression._value_
    if isinstance(value, type):
        return False
    return dataclasses.is_dataclass(value) or isinstance(value, Symbol)


def is_atomic_value(expression: SymbolicExpression) -> bool:
    """
    :param expression: Candidate value expression (a comparator right side / assignment value).
    :return: ``True`` when the value renders as a single short token — a plain scalar / ``enum`` /
        type-object literal — so it may sit in a *"… are X, Y, and Z respectively"* coordination.
        A concrete-object literal (*"a specific Body"*), a domain-listing variable (*"one of …"*), a
        sub-query, or a range each renders as a phrase and is *not* atomic, so it is said on its own.
    """
    if not isinstance(expression, Literal):
        return False
    if expression._value_ is None:
        return False
    return not is_concrete_object_literal(expression)


def references(expression: SymbolicExpression, subject_variable: Variable) -> bool:
    """
    :param expression: Candidate expression.
    :param subject_variable: The variable to look for.
    :return: ``True`` when *expression* mentions *subject_variable* (so it is not a clean right-hand side
        value).
    """
    try:
        return any(
            getattr(variable, "_id_", None) == subject_variable._id_
            for variable in expression._unique_variables_
        )
    except AttributeError:
        return chain_root(expression) is subject_variable


def is_boolean_attribute_chain(expression: SymbolicExpression) -> bool:
    """
    :param expression: Candidate expression.
    :return: ``True`` when *expression* is a ``MappedVariable`` chain ending in a ``bool``-typed
        attribute.
    """
    if not isinstance(expression, MappedVariable):
        return False
    chain, _ = walk_chain(expression)
    return chain_ends_in_boolean_attribute(chain)


@dataclass(frozen=True)
class SuperlativeFold:
    """
    A subject restriction of the form ``subject.<chain> == max/min(<same-type>.<same chain>)``
    folded to a superlative selection modifier — *"with the maximum <leaf>"*.

    English's superlative is the meaning of "equal to the extreme value over the whole
    population", so this fold is meaning-preserving, not an optimisation.
    """

    aggregator: Aggregator
    """The ``Max`` / ``Min`` aggregator — supplies the superlative word and the leaf attribute
    for the *"… <leaf>"* tail."""


def superlative_aggregation(
    comparator: SymbolicExpression, subject: Optional[Variable]
) -> Optional[SuperlativeFold]:
    """Recognise a superlative restriction on *subject*.  The guard is strict so it never fires
    on a self-join, a constrained sub-query, a different chain, or a non-extreme aggregation.

    :param comparator: Candidate ``==`` comparator.
    :param subject: The variable the restriction is on.
    :return: The fold when *comparator* is ``subject.<chain> == <unconstrained Max/Min over a
        different same-type variable's identical chain>``, else ``None``.
    """
    if subject is None or not isinstance(comparator, Comparator):
        return None
    if comparator.operation is not operator.eq:
        return None
    left_root = chain_root(comparator.left)
    if getattr(left_root, "_id_", None) != getattr(subject, "_id_", object()):
        return None

    inner = unwrap_result_quantifiers(comparator.right)
    if not (isinstance(inner, Entity) and is_collapsible_aggregation_subquery(inner)):
        return None
    aggregator = selected_aggregator(inner)
    if not isinstance(aggregator, Extreme) or aggregator._leaf_attribute_ is None:
        return None

    source = aggregation_source_root(inner)
    if source is None or source is subject:
        return None  # the population must be a distinct variable of the same type
    if getattr(source, "_type_", None) is not getattr(subject, "_type_", None):
        return None
    if attribute_names(comparator.left) != attribute_names(aggregator._child_):
        return None
    return SuperlativeFold(aggregator=aggregator)
