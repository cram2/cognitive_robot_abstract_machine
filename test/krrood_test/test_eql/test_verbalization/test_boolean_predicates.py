"""
Tests for boolean-attribute predicate verbalization:
:mod:`krrood.entity_query_language.verbalization.attribute_predicates`.

A boolean attribute reads as a predicate whose form is declared per field
(:class:`~krrood.patterns.boolean_predicate.BooleanPredicateSpec`) or inferred from the attribute
name's shape: possessive (*"has milk"*), adjectival (*"is operational"*), or verbal (*"produces
milk"*). Negation is derived — do-support / copula suppletion — never re-templated.

The mimic dataclasses are named after the predicate shape they exercise, not any external class.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field

import pytest

from krrood.entity_query_language.factories import for_all, variable
from krrood.entity_query_language.verbalization.attribute_predicates import (
    boolean_predicate_clause,
    default_boolean_predicate,
    resolve_boolean_predicate,
)
from krrood.entity_query_language.verbalization.exceptions import (
    UnknownBooleanPredicateError,
)
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.patterns.boolean_predicate import (
    AdjectivalPredicate,
    Article,
    BooleanPredicateSpec,
    PossessivePredicate,
    VerbalPredicate,
)
from krrood.patterns.field_metadata import FieldMetadata, GrammarMetadata


def _predicate(spec: BooleanPredicateSpec) -> object:
    """
    A boolean dataclass field declaring *spec* as its predicate form.
    """
    return field(
        default=False,
        metadata=FieldMetadata(
            other_metadata=[GrammarMetadata(boolean_predicate=spec)]
        ).as_dict(),
    )


@dataclass
class _InferredForms:
    """
    Boolean fields with no metadata — each resolves by name-shape heuristic.
    """

    milk: bool = False
    backbone: bool = False
    completed: bool = False
    operational: bool = False


@dataclass
class _DeclaredForms:
    """
    Boolean fields each declaring an explicit predicate form.
    """

    milk: bool = _predicate(PossessivePredicate())
    backbone: bool = _predicate(PossessivePredicate(article=Article.INDEFINITE))
    glands: bool = _predicate(PossessivePredicate(noun="mammary glands"))
    reachable: bool = _predicate(AdjectivalPredicate(adjective="within reach"))
    secretes_milk: bool = _predicate(
        VerbalPredicate(verb="secrete", object_noun="milk")
    )
    breathes: bool = _predicate(VerbalPredicate(verb="breathe"))


# %% Heuristic default resolution


def test_noun_name_defaults_to_possessive():
    assert isinstance(default_boolean_predicate("milk"), PossessivePredicate)


def test_participle_name_defaults_to_adjectival():
    assert isinstance(default_boolean_predicate("completed"), AdjectivalPredicate)


def test_adjective_suffix_name_defaults_to_adjectival():
    assert isinstance(default_boolean_predicate("operational"), AdjectivalPredicate)


def test_declared_spec_overrides_the_heuristic():
    assert resolve_boolean_predicate(_DeclaredForms, "milk") == PossessivePredicate()


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
    animal = variable(_DeclaredForms, [])
    assert (
        verbalize_expression(animal.milk == variable(bool, [True, False]))
        == "a _DeclaredForms has either milk or not"
    )


def test_open_domain_adjectival_alternative():
    animal = variable(_InferredForms, [])
    assert (
        verbalize_expression(animal.operational == variable(bool, [True, False]))
        == "a _InferredForms is either operational or not"
    )


# %% Fail-loud on an unrealizable spec


@dataclass(frozen=True)
class _UnrealizablePredicate(BooleanPredicateSpec):
    """
    A spec with no registered realizer, to prove the coverage gap fails loudly.
    """


class _SubjectStub:
    """
    A minimal clause subject stand-in (the raise happens before it is used).
    """

    def as_fragment(self):
        return self


def test_unknown_spec_raises():
    with pytest.raises(UnknownBooleanPredicateError):
        boolean_predicate_clause(
            _SubjectStub(), _UnrealizablePredicate(), _DeclaredForms, "milk"
        )


# %% Layering: the patterns spec must not pull in the verbalization subsystem


def test_patterns_spec_has_no_verbalization_dependency():
    code = (
        "import sys, krrood.patterns.boolean_predicate;"
        "leaked = [m for m in sys.modules "
        "if m.startswith('krrood.entity_query_language.verbalization')];"
        "assert not leaked, leaked"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ})
