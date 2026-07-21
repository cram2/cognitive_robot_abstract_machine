"""
Assemble a boolean attribute's predicate clause.

A boolean attribute reads as a predicate whose form is declared per field
(:class:`~krrood.entity_query_language.verbalization.boolean_predicate.BooleanPredicate`) or inferred
from the attribute name's shape. Each predicate builds its own head and object; this module assembles
them into the positive clause, derives the open-boolean *"either … or not"* coordination, and resolves
which predicate a field uses. The negation is *derived*, not re-templated — the head carries the
``negated`` flag and the morphology pass realises do-support (*"does not have milk"*) or copula
suppletion (*"is not operational"*).

It depends only on the parts-of-speech vocabulary, the predicate types, and morphology (no ``grammar``
import), so the chain assembler can call it without a cycle.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from typing_extensions import List

from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.boolean_predicate import (
    AdjectivalPredicate,
    BooleanPredicate,
    PossessivePredicate,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    Clause,
    RoleFragment,
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar_metadata import GrammarMetadata
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    Logicals,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    ClauseConstituent,
    clause,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.core.mapped_variable import Attribute


def default_boolean_predicate(attribute_name: str) -> BooleanPredicate:
    """:param attribute_name: The boolean attribute's name.

    :return: The predicate inferred from *attribute_name*'s shape — adjectival for a
        participle/adjective-shaped name (*"completed"*, *"operational"*), else possessive (*"milk"* →
        *"has milk"*).

    A best-effort fallback for a field with no explicit
    :class:`~krrood.entity_query_language.verbalization.grammar_metadata.GrammarMetadata`. The
    participle test is deterministic; the adjective test is a suffix heuristic, so a suffix-less
    adjective (*"airborne"*) falls through to possessive. A verbal reading (*"produces milk"*) is
    never inferred — a lexical verb cannot be told from a noun by shape — so it requires an explicit
    :class:`~krrood.entity_query_language.verbalization.boolean_predicate.VerbalPredicate`. The
    reliable source of the classification is an explicit definition through the field's grammar
    metadata.
    """
    last = attribute_name.split("_")[-1]
    if morphology.is_past_participle(last) or morphology.is_likely_adjective(last):
        return AdjectivalPredicate()
    return PossessivePredicate()


def resolve_boolean_predicate(attribute: Attribute) -> BooleanPredicate:
    """:param attribute: The boolean attribute node.

    :return: The :class:`~krrood.entity_query_language.verbalization.boolean_predicate.BooleanPredicate`
        the field declares via
        :class:`~krrood.entity_query_language.verbalization.grammar_metadata.GrammarMetadata`, or the
        :func:`default_boolean_predicate` heuristic when the field declares none.
    """
    metadata = GrammarMetadata.of_field(
        attribute._owner_class_, attribute._attribute_name_
    )
    if metadata is not None and metadata.boolean_predicate is not None:
        return metadata.boolean_predicate
    return default_boolean_predicate(attribute._attribute_name_)


def boolean_predicate_clause(
    subject: VerbalizationFragment,
    predicate: BooleanPredicate,
    attribute: Attribute,
    negated: bool = False,
    number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
) -> Clause:
    """
    :param subject: The already-rendered subject phrase (the navigation to the attribute owner).
    :param predicate: How the attribute reads as a predicate.
    :param attribute: The boolean attribute node (its owner and name back the default surface).
    :param negated: Whether the predicate is negated — realised as do-support / copula suppletion.
    :param number: The subject number the head agrees with (coreference re-agrees an in-scope subject).
    :return: The subject-led predicate clause (*"the Animal has milk"* / *"is not operational"*).
    """
    head = replace(predicate.head(), negated=negated, number=number)
    predicate_object = predicate.predicate_object(attribute)
    constituents: List[ClauseConstituent] = [subject, head]
    if predicate_object is not None:
        constituents.append(predicate_object)
    return clause(*constituents)


def boolean_alternative_clause(
    subject: VerbalizationFragment,
    predicate: BooleanPredicate,
    attribute: Attribute,
) -> Clause:
    """:param subject: The already-rendered subject phrase.
    :param predicate: How the attribute reads as a predicate.
    :param attribute: The boolean attribute node.

    :return: The open-boolean predicate — for a boolean attribute left open (compared to a
        both-``True``-and-``False`` domain), so neither polarity is asserted (*"is either operational
        or not"*, *"either has milk or not"*).

    Where *either* attaches follows how the head negates — the same fact the morphology pass uses. A
    head that negates in place (copula, *"is"* → *"is not"*) is shared across both polarities, so only
    the object is coordinated and *either* sits after it: *"is either operational or not"*. A head that
    negates via do-support (a verb, *"has"* → *"does not have"*) is not shared, so the whole verb
    phrase is coordinated and *either* fronts the head: *"either has milk or not"*, *"either breathes
    or not"*.
    """
    head = predicate.head()
    predicate_object = predicate.predicate_object(attribute)
    if _head_negates_in_place(head):
        return clause(
            subject,
            head,
            Logicals.EITHER,
            predicate_object,
            Conjunctions.OR,
            Logicals.NOT,
        )
    constituents: List[ClauseConstituent] = [subject, Logicals.EITHER, head]
    if predicate_object is not None:
        constituents.append(predicate_object)
    constituents += [Conjunctions.OR, Logicals.NOT]
    return clause(*constituents)


def _head_negates_in_place(head: VerbalizationFragment) -> bool:
    """:param head: A predicate's finite head leaf.

    :return: Whether *head* realises negation on itself (copula suppletion, *"is not"*) rather than
        via do-support (*"does not have"*).

    Mirrors the ``OPERATOR``-vs-``VERB`` split the
    :class:`~krrood.entity_query_language.verbalization.rendering.morphology_processor.MorphologyProcessor`
    uses — the one fact that also decides where *either* attaches in an open-boolean coordination.
    """
    return isinstance(head, RoleFragment) and head.role is SemanticRole.OPERATOR
