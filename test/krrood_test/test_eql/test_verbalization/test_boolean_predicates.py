"""
Tests for boolean-attribute predicate verbalization:
:mod:`krrood.entity_query_language.verbalization.attribute_predicates`.

A boolean attribute reads as a predicate whose form is declared per field
(:class:`~krrood.entity_query_language.verbalization.boolean_predicate.BooleanPredicate`) or inferred
from the attribute name's shape: possessive (*"has milk"*), adjectival (*"is operational"*), or verbal
(*"produces milk"*). Negation is derived — do-support / copula suppletion — never re-templated.

The mimic dataclasses are named after the predicate shape they exercise, not any external class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood.entity_query_language.factories import for_all, variable
from krrood.entity_query_language.verbalization.attribute_predicates import (
    default_boolean_predicate,
    resolve_boolean_predicate,
)
from krrood.entity_query_language.verbalization.boolean_predicate import (
    AdjectivalPredicate,
    BooleanPredicate,
    PossessivePredicate,
    VerbalPredicate,
)
from krrood.entity_query_language.verbalization.fragments.features import Definiteness
from krrood.entity_query_language.verbalization.grammar_metadata import GrammarMetadata
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.patterns.field_metadata import FieldMetadata


def _predicate(predicate: BooleanPredicate) -> object:
    """
    A boolean dataclass field declaring *predicate* as its predicate form.
    """
    return field(
        default=False,
        metadata=FieldMetadata(
            other_metadata=[GrammarMetadata(boolean_predicate=predicate)]
        ).as_dict(),
    )


@dataclass
class _InferredForms:
    """
    Boolean fields with no metadata — each resolves by name-shape heuristic.
    """

    milk: bool = False
    """A noun-shaped name — inferred as possessive (*"has milk"*)."""

    backbone: bool = False
    """A noun-shaped name — inferred as possessive (*"has backbone"*)."""

    completed: bool = False
    """A past-participle name — inferred as adjectival (*"is completed"*)."""

    operational: bool = False
    """An adjective-suffix name — inferred as adjectival (*"is operational"*)."""


@dataclass
class _DeclaredForms:
    """
    Boolean fields each declaring an explicit predicate form.
    """

    milk: bool = _predicate(PossessivePredicate())
    """A bare-noun possessive (*"has milk"*)."""

    backbone: bool = _predicate(
        PossessivePredicate(definiteness=Definiteness.INDEFINITE)
    )
    """A count-noun possessive taking an article (*"has a backbone"*)."""

    glands: bool = _predicate(PossessivePredicate(noun="mammary glands"))
    """A possessive overriding the noun (*"has mammary glands"*)."""

    reachable: bool = _predicate(AdjectivalPredicate(adjective="within reach"))
    """An adjectival predicate overriding the adjective (*"is within reach"*)."""

    secretes_milk: bool = _predicate(
        VerbalPredicate(verb="secrete", object_noun="milk")
    )
    """A transitive verbal predicate (*"secretes milk"*)."""

    breathes: bool = _predicate(VerbalPredicate(verb="breathe"))
    """An intransitive verbal predicate (*"breathes"*)."""


# %% Heuristic default resolution


def test_noun_name_defaults_to_possessive():
    assert isinstance(default_boolean_predicate("milk"), PossessivePredicate)


def test_participle_name_defaults_to_adjectival():
    assert isinstance(default_boolean_predicate("completed"), AdjectivalPredicate)


def test_adjective_suffix_name_defaults_to_adjectival():
    assert isinstance(default_boolean_predicate("operational"), AdjectivalPredicate)


def test_declared_predicate_overrides_the_heuristic():
    animal = variable(_DeclaredForms, [])
    assert resolve_boolean_predicate(animal.milk) == PossessivePredicate()


def test_verbs_are_never_inferred_only_declared():
    # A verb cannot be told from a noun by shape, so the heuristic never guesses a verbal reading.
    assert not isinstance(default_boolean_predicate("secretes_milk"), VerbalPredicate)
    # A verbal reading is reachable only when the field declares it.
    animal = variable(_DeclaredForms, [])
    assert isinstance(resolve_boolean_predicate(animal.secretes_milk), VerbalPredicate)


# %% Inferred surfaces (no metadata)


def test_inferred_noun_reads_as_possession():
    animal = variable(_InferredForms, [])
    assert verbalize_expression(animal.milk == True) == "a _InferredForms has milk"


def test_inferred_participle_reads_as_state():
    animal = variable(_InferredForms, [])
    assert (
        verbalize_expression(animal.completed == True)
        == "a _InferredForms is completed"
    )


def test_inferred_adjective_reads_as_state():
    animal = variable(_InferredForms, [])
    assert (
        verbalize_expression(animal.operational == True)
        == "a _InferredForms is operational"
    )


# %% Declared surfaces


def test_possessive_bare_noun():
    animal = variable(_DeclaredForms, [])
    assert verbalize_expression(animal.milk == True) == "a _DeclaredForms has milk"


def test_possessive_count_noun_takes_an_article():
    animal = variable(_DeclaredForms, [])
    assert (
        verbalize_expression(animal.backbone == True)
        == "a _DeclaredForms has a backbone"
    )


def test_possessive_noun_override():
    animal = variable(_DeclaredForms, [])
    assert (
        verbalize_expression(animal.glands == True)
        == "a _DeclaredForms has mammary glands"
    )


def test_adjectival_override():
    animal = variable(_DeclaredForms, [])
    assert (
        verbalize_expression(animal.reachable == True)
        == "a _DeclaredForms is within reach"
    )


def test_verbal_with_object():
    animal = variable(_DeclaredForms, [])
    assert (
        verbalize_expression(animal.secretes_milk == True)
        == "a _DeclaredForms secretes milk"
    )


def test_verbal_intransitive():
    animal = variable(_DeclaredForms, [])
    assert verbalize_expression(animal.breathes == True) == "a _DeclaredForms breathes"


# %% Negation is derived from the positive form


def test_possessive_negation_uses_do_support():
    animal = variable(_DeclaredForms, [])
    assert (
        verbalize_expression(animal.milk == False)
        == "a _DeclaredForms does not have milk"
    )


def test_adjectival_negation_uses_copula_suppletion():
    animal = variable(_InferredForms, [])
    assert (
        verbalize_expression(animal.operational == False)
        == "a _InferredForms is not operational"
    )


def test_verbal_negation_uses_do_support():
    animal = variable(_DeclaredForms, [])
    assert (
        verbalize_expression(animal.secretes_milk == False)
        == "a _DeclaredForms does not secrete milk"
    )


# %% Number agreement (plural subject)


def test_possessive_agrees_with_a_plural_subject():
    animal = variable(_DeclaredForms, [])
    assert "they have milk" in verbalize_expression(
        for_all(animal, animal.milk == True)
    )


def test_adjectival_agrees_with_a_plural_subject():
    animal = variable(_InferredForms, [])
    assert "they are operational" in verbalize_expression(
        for_all(animal, animal.operational == True)
    )


# %% Open boolean domain (either / or not)


def test_open_domain_possessive_alternative():
    # Do-support fronts "either" before the whole verb phrase ("either has milk or not"), because the
    # auxiliary differs between the two polarities ("has" / "does not have").
    animal = variable(_DeclaredForms, [])
    assert (
        verbalize_expression(animal.milk == variable(bool, [True, False]))
        == "a _DeclaredForms either has milk or not"
    )


def test_open_domain_adjectival_alternative():
    # The copula is shared across both polarities ("is" / "is not"), so "either" sits after it and
    # only the complement is coordinated.
    animal = variable(_InferredForms, [])
    assert (
        verbalize_expression(animal.operational == variable(bool, [True, False]))
        == "a _InferredForms is either operational or not"
    )


def test_open_domain_verbal_alternative():
    animal = variable(_DeclaredForms, [])
    assert (
        verbalize_expression(animal.secretes_milk == variable(bool, [True, False]))
        == "a _DeclaredForms either secretes milk or not"
    )


def test_open_domain_intransitive_verb_alternative():
    animal = variable(_DeclaredForms, [])
    assert (
        verbalize_expression(animal.breathes == variable(bool, [True, False]))
        == "a _DeclaredForms either breathes or not"
    )
