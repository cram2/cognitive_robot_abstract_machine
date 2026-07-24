"""
Tests for operand referring: how a predicate / function names the operands filling its
fields, and how same-noun operands are told apart.

The head noun is resolved in order of decreasing specificity: the operand's own type
when informative (a concrete class always wins, so a genuinely typed operand is never
hidden behind its field's role), else the owning field's declared grammatical metadata,
else the field name itself, else a generic "object". Same-noun operands are
disambiguated by the referring service (through the ordinary coreference machinery,
keyed on referent identity) rather than inside the predicate: a fresh pair reads "a
point ... another point"; a larger group reads "a point, a second point, and a third
point". A variable reused elsewhere (a query subject) keeps its ordinary referring
rendering, including pronominalisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood.entity_query_language.factories import an, entity, for_all, variable
from krrood.entity_query_language.predicate import Predicate, SymbolicFunction
from krrood.entity_query_language.verbalization.microplanning.referring import (
    operand_head_noun,
    ParentEdge,
)
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.entity_query_language.verbalization.vocabulary.english import Prepositions
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    Adjective,
    clause,
    ConjunctivePhrase,
    Copula,
    FunctionVerbalizationTemplates,
    Noun,
    Verb,
)
from krrood.entity_query_language.verbalization.grammar_metadata import (
    GrammarMetadata,
)
from krrood.patterns.field_metadata import FieldMetadata

# %% mimic domain


class Marker:
    """
    A stand-in operand type whose class name is unremarkable, so it still identifies the
    operand plainly when its type is informative.
    """


class Worker:
    """
    A stand-in operand type for the distinct-role predicate.
    """


@dataclass(eq=False)
class SameTypePair(Predicate):
    """
    Two operands of the same type, told apart by the referring service rather than
    inside the predicate.
    """

    point_1: Marker
    point_2: Marker

    def __call__(self) -> bool:
        return True

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(Noun(fields["point_1"]), Verb("face"), Noun(fields["point_2"]))


@dataclass(eq=False)
class SameTypeTriple(Predicate):
    """
    Three operands of the same type, disambiguated by determiner then ordinals.
    """

    point_1: Marker
    point_2: Marker
    point_3: Marker

    def __call__(self) -> bool:
        return True

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(
            ConjunctivePhrase(
                [
                    Noun(fields["point_1"]),
                    Noun(fields["point_2"]),
                    Noun(fields["point_3"]),
                ]
            ),
            Copula(),
            Adjective("collinear"),
        )


@dataclass(eq=False)
class MetadataNamed(Predicate):
    """
    An operand whose type is uninformative (``object``), named by a grammatical-metadata
    display name overriding the field name.
    """

    burning_thing: object = field(
        metadata=FieldMetadata(
            other_metadata=[GrammarMetadata(display_name="torch")]
        ).as_dict()
    )

    def __call__(self) -> bool:
        return True

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(Noun(fields["burning_thing"]), Copula(), Adjective("lit"))


@dataclass(eq=False)
class DistinctRoles(Predicate):
    """
    Two same-type operands with distinct role names, kept distinct — no determiner
    disambiguation, because they resolve different nouns.
    """

    subordinate: Worker
    supervisor: Worker

    def __call__(self) -> bool:
        return True

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(
            Noun(fields["subordinate"]),
            Verb("report"),
            Prepositions.TO,
            Noun(fields["supervisor"]),
        )


@dataclass(eq=False)
class SingleRole(Predicate):
    """
    A single-operand predicate used both standalone and as a query subject.
    """

    surface: Marker

    def __call__(self) -> bool:
        return True

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(Noun(fields["surface"]), Copula(), Adjective("warm"))


@dataclass(eq=False)
class UntypedRole(Predicate):
    """
    A single-operand predicate whose field is typed ``object`` (uninformative) and
    carries no metadata, so the field name is the fallback noun.
    """

    location: object

    def __call__(self) -> bool:
        return True

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(Noun(fields["location"]), Copula(), Adjective("reachable"))


@dataclass(eq=False)
class RoleNamedReading(SymbolicFunction):
    """
    A value function whose operand is field-named inside a possessive noun phrase.
    """

    sensor: object

    def __call__(self) -> float:
        return 0.0

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return FunctionVerbalizationTemplates.possessive(cls, *fields.values())


# %% head-noun resolution (operand_head_noun)


def test_informative_type_is_the_default_head_noun():
    assert operand_head_noun(variable(Marker, []), []) == "Marker"


def test_informative_type_wins_over_field_metadata():
    """
    A concrete type always wins, even when the sole owning field declares metadata --
    metadata is a fallback for an uninformative type, not an override.
    """
    parent = MetadataNamed(variable(Marker, []))
    edges = [ParentEdge(parent, "burning_thing")]
    assert operand_head_noun(variable(Marker, []), edges) == "Marker"


def test_uninformative_type_falls_through_to_field_metadata():
    parent = MetadataNamed(variable(object, []))
    edges = [ParentEdge(parent, "burning_thing")]
    assert operand_head_noun(variable(object, []), edges) == "torch"


def test_uninformative_type_falls_through_to_field_name():
    parent = UntypedRole(variable(object, []))
    edges = [ParentEdge(parent, "location")]
    assert operand_head_noun(variable(object, []), edges) == "location"


def test_uninformative_type_with_no_sole_field_falls_through_to_object():
    assert operand_head_noun(variable(object, []), []) == "object"


def test_reused_operand_ignores_field_context():
    """
    A variable reachable through more than one edge is not a sole predicate operand, so
    it never reads a field name or metadata -- only its type (or "object" as the last
    resort) applies.
    """
    parent = UntypedRole(variable(object, []))
    edges = [ParentEdge(parent, "location"), ParentEdge(parent, "location")]
    assert operand_head_noun(variable(object, []), edges) == "object"


# %% same-noun disambiguation (end-to-end)


def test_same_type_pair_uses_the_indefinite_alternative_on_first_mention():
    assert verbalize_expression(
        SameTypePair(variable(Marker, []), variable(Marker, []))
    ) == ("a Marker faces another Marker")


def test_same_type_triple_uses_ordinals_with_a_plural_copula():
    assert verbalize_expression(
        SameTypeTriple(variable(Marker, []), variable(Marker, []), variable(Marker, []))
    ) == ("a Marker, a second Marker, and a third Marker are collinear")


def test_distinct_role_names_are_kept_distinct():
    assert verbalize_expression(
        DistinctRoles(variable(Worker, []), variable(Worker, []))
    ) == ("a Worker reports to another Worker")


def test_metadata_named_pair_shares_one_noun_and_is_disambiguated():
    assert verbalize_expression(MetadataNamed(variable(object, []))) == "a torch is lit"


# %% reused operand keeps ordinary referring rendering


def test_single_operand_is_named_by_its_type_when_informative():
    assert verbalize_expression(SingleRole(variable(Marker, []))) == "a Marker is warm"


def test_single_operand_falls_back_to_field_name_when_untyped():
    assert verbalize_expression(UntypedRole(variable(object, []))) == (
        "a location is reachable"
    )


def test_reused_operand_pronominalises_as_the_query_subject():
    """A variable that is also the query subject keeps its ordinary referring rendering and
    pronominalises -- it is not treated as an anonymous operand."""
    subject = variable(Marker, [])
    assert verbalize_expression(an(entity(subject).where(SingleRole(subject)))) == (
        "Find a Marker such that it is warm"
    )


def test_reused_operand_agrees_with_a_plural_population():
    subject = variable(Marker, [])
    assert verbalize_expression(for_all(subject, SingleRole(subject))) == (
        "for all Markers, they are warm"
    )


def test_function_operand_is_named_by_its_type_when_informative():
    assert verbalize_expression(RoleNamedReading(variable(Marker, []))) == (
        "the role named reading of a Marker"
    )


def test_function_operand_falls_back_to_field_name_when_untyped():
    assert verbalize_expression(RoleNamedReading(variable(object, []))) == (
        "the role named reading of a sensor"
    )
