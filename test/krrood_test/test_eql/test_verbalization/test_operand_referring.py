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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from krrood.entity_query_language.factories import an, and_, entity, for_all, variable
from krrood.entity_query_language.predicate import Predicate, SymbolicFunction
from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
)
from krrood.entity_query_language.verbalization.grammar_metadata import GrammarMetadata
from krrood.entity_query_language.verbalization.microplanning.referring import (
    disjunctive_type_head,
    operand_head_noun,
    operand_type_alternatives,
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
        metadata=GrammarMetadata(display_name="torch").as_dict()
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


@dataclass
class Shape(ABC):
    """
    A stand-in abstract operand type with a small, nameable family of concrete
    subclasses -- never a valid direct referent itself.
    """

    @abstractmethod
    def area(self) -> float: ...


@dataclass
class Circle(Shape):
    """
    One of two concrete alternatives naming a :class:`Shape`-typed operand.
    """

    def area(self) -> float:
        return 0.0


@dataclass
class Square(Shape):
    """
    The other of two concrete alternatives naming a :class:`Shape`-typed operand.
    """

    def area(self) -> float:
        return 0.0


@dataclass
class Instrument(ABC):
    """
    A stand-in abstract operand type with a three-member family -- exercises the Oxford-
    comma joining an over-two-item disjunction takes (*"Drum, Flute, or Harp"*), where a
    naive ``" or "``-only join would silently disagree with the linked fragment's own
    joining.
    """

    @abstractmethod
    def play(self) -> None: ...


@dataclass
class Drum(Instrument):
    def play(self) -> None:
        pass


@dataclass
class Flute(Instrument):
    def play(self) -> None:
        pass


@dataclass
class Harp(Instrument):
    def play(self) -> None:
        pass


@dataclass
class Polygon(ABC):
    """
    A stand-in abstract operand type whose concrete family is too large to spell out --
    named directly, like an uninformative-but-concrete type.
    """

    @abstractmethod
    def sides(self) -> int: ...


@dataclass
class Triangle(Polygon):
    def sides(self) -> int:
        return 3


@dataclass
class Quadrilateral(Polygon):
    def sides(self) -> int:
        return 4


@dataclass
class Pentagon(Polygon):
    def sides(self) -> int:
        return 5


@dataclass
class Hexagon(Polygon):
    def sides(self) -> int:
        return 6


@dataclass
class Heptagon(Polygon):
    def sides(self) -> int:
        return 7


@dataclass
class Octagon(Polygon):
    def sides(self) -> int:
        return 8


@dataclass
class Nonagon(Polygon):
    def sides(self) -> int:
        return 9


@dataclass
class ConcreteBase:
    """
    A stand-in *concrete* (non-abstract) operand type that nonetheless has subclasses --
    still a valid referent in its own right, unlike an abstract base with no valid
    direct instance.
    """


@dataclass
class ConcreteBaseVariant(ConcreteBase):
    """
    A subclass of :class:`ConcreteBase`, present only to prove a concrete base with
    subclasses is never expanded.
    """


@dataclass
class Sensor:
    """
    A stand-in operand type for the concrete counterpart alongside an abstract-typed
    one.
    """


@dataclass(eq=False)
class AbstractOperandRole(Predicate):
    """
    A single-operand predicate whose field is typed with an abstract base -- named by
    disjunction over its concrete subclasses rather than by its own (unnameable) type.
    """

    surface: Shape

    def __call__(self) -> bool:
        return True

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(Noun(fields["surface"]), Copula(), Adjective("warm"))


@dataclass(eq=False)
class InstrumentRole(Predicate):
    """
    A single-operand predicate whose field is typed with a three-member abstract family
    -- exercises the Oxford-comma-joined disjunctive head end-to-end.
    """

    surface: Instrument

    def __call__(self) -> bool:
        return True

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(Noun(fields["surface"]), Copula(), Adjective("warm"))


@dataclass(eq=False)
class VisibleFromSensor(Predicate):
    """
    Mirrors a real two-operand predicate whose one field is typed with an abstract base
    and the other with a concrete one (e.g. ``Visible(camera,
    KinematicStructureEntity)``).
    """

    target: Shape
    sensor: Sensor

    def __call__(self) -> bool:
        return True

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(
            Noun(fields["target"]),
            Copula(),
            Adjective("visible"),
            Prepositions.FROM,
            Noun(fields["sensor"]),
        )


@dataclass(eq=False)
class LargeFamilyRole(Predicate):
    """
    A single-operand predicate whose field is typed with an abstract base whose concrete
    family exceeds the disjunction cap.
    """

    surface: Polygon

    def __call__(self) -> bool:
        return True

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(Noun(fields["surface"]), Copula(), Adjective("warm"))


@dataclass(eq=False)
class ConcreteBaseRole(Predicate):
    """
    A single-operand predicate whose field is typed with a concrete (non-abstract) base.
    """

    surface: ConcreteBase

    def __call__(self) -> bool:
        return True

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(Noun(fields["surface"]), Copula(), Adjective("warm"))


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


# %% abstract operand disjunction (concrete-subclass expansion)


def test_abstract_type_alternatives_are_its_concrete_subclasses():
    assert operand_type_alternatives(variable(Shape, [])) == (Circle, Square)


def test_concrete_type_has_no_alternatives_even_with_subclasses():
    """
    A concrete base is a valid referent in its own right, even with subclasses of its
    own -- only an abstract base (never itself a valid instance) is expanded.
    """
    assert operand_type_alternatives(variable(ConcreteBase, [])) == ()


def test_abstract_type_beyond_the_cap_has_no_alternatives():
    """
    A concrete family too large to spell out falls back to naming the abstract type
    directly, the same bounded-listing trade-off `one_of` makes for a value domain.
    """
    assert operand_type_alternatives(variable(Polygon, [])) == ()


def test_uninformative_type_has_no_alternatives():
    assert operand_type_alternatives(variable(object, [])) == ()


def test_disjunctive_type_head_renders_a_bare_or_joined_noun():
    assert (
        flatten_fragment_to_plain_text(disjunctive_type_head((Circle, Square)))
        == "Circle or Square"
    )


def test_abstract_type_head_noun_is_the_disjunctive_label():
    assert operand_head_noun(variable(Shape, []), []) == "Circle or Square"


def test_three_member_family_head_noun_matches_the_oxford_comma_joined_fragment():
    """
    The head-noun label reuses `disjunctive_type_head`'s own joining rather than a plain
    `" or "` join, so a three-or-more-member family agrees with the Oxford-comma-joined
    fragment `VariableRule.build` actually renders (`"Drum, Flute, or Harp"`, not the
    comma-less `"Drum or Flute or Harp"` a naive join would produce).
    """
    alternatives = (Drum, Flute, Harp)
    assert operand_head_noun(
        variable(Instrument, []), []
    ) == flatten_fragment_to_plain_text(disjunctive_type_head(alternatives))
    assert operand_head_noun(variable(Instrument, []), []) == "Drum, Flute, or Harp"


def test_abstract_type_beyond_the_cap_is_named_by_its_own_type():
    assert operand_head_noun(variable(Polygon, []), []) == "Polygon"


def test_concrete_base_with_subclasses_is_named_directly_by_head_noun():
    assert operand_head_noun(variable(ConcreteBase, []), []) == "ConcreteBase"


def test_abstract_operand_reads_as_a_disjunctive_noun_phrase():
    assert verbalize_expression(AbstractOperandRole(variable(Shape, []))) == (
        "a Circle or a Square is warm"
    )


def test_three_member_family_reads_with_an_oxford_comma():
    assert verbalize_expression(InstrumentRole(variable(Instrument, []))) == (
        "a Drum, a Flute, or a Harp is warm"
    )


def test_abstract_operand_alongside_a_concrete_one_reads_naturally():
    assert (
        verbalize_expression(
            VisibleFromSensor(variable(Shape, []), variable(Sensor, []))
        )
        == "a Circle or a Square is visible from a Sensor"
    )


def test_large_concrete_family_falls_back_to_naming_the_abstract_type():
    assert verbalize_expression(LargeFamilyRole(variable(Polygon, []))) == (
        "a Polygon is warm"
    )


def test_concrete_base_with_subclasses_is_named_directly():
    assert verbalize_expression(ConcreteBaseRole(variable(ConcreteBase, []))) == (
        "a ConcreteBase is warm"
    )


def test_reused_abstract_operand_pronominalises_as_the_query_subject():
    """
    A variable typed with an abstract base is disambiguated by disjunction wherever it
    is resolved, including as a query subject -- the trigger is the type alone, not
    whether the variable is treated as an anonymous operand.
    """
    subject = variable(Shape, [])
    assert (
        verbalize_expression(an(entity(subject).where(AbstractOperandRole(subject))))
        == "Find a Circle or a Square such that it is warm"
    )


def test_reused_abstract_operand_reads_as_a_definite_disjunction_on_repeat_mention():
    """
    A repeat mention that is not the current discourse subject (so it is not
    pronominalised) still carries every disjunctive alternative through to its definite
    form -- "the Circle or the Square", not just "the Circle" with the other alternative
    silently dropped.
    """
    shape, sensor = variable(Shape, []), variable(Sensor, [])
    assert verbalize_expression(
        and_(AbstractOperandRole(shape), VisibleFromSensor(shape, sensor))
    ) == (
        "a Circle or a Square is warm, and the Circle or the Square is visible from a Sensor"
    )


def test_reused_abstract_operand_pronominalises_on_every_mention_within_its_scope():
    """
    A discourse scope (here, the query's WHERE) makes its subject pronoun-eligible for
    every mention inside it, not just the first repeat -- both conjuncts naming the same
    disjunctively-typed query subject read "it", each keeping the full first-mention
    disjunction intact for the one spelled-out occurrence.
    """
    subject, sensor = variable(Shape, []), variable(Sensor, [])
    assert verbalize_expression(
        an(
            entity(subject).where(
                and_(AbstractOperandRole(subject), VisibleFromSensor(subject, sensor))
            )
        )
    ) == (
        "Find a Circle or a Square such that it is warm, and it is visible from a Sensor"
    )
