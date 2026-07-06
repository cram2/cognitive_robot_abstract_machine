"""
Tests for verbalizing EQL *match* expressions built with ``an``.

A match verbalized on its own opens with *"Generate"* — it is a construction description, and
whether a backend *finds* (selective) or *generates* (generative) is the backend's concern. A
match has two condition parts: the construction-pattern equalities (the ``kwargs``) → *"given
that"*, and the ``.where(...)`` conditions → *"where"*.  Each condition is its own point; equality
assignments on the same object are grouped (*"x, y, and z of the Position are 1, 2, and 3
respectively"*); an ``Ellipsis`` value is a value to generate → folded into the header (*"… and
predict its x, y, and z values"*).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from krrood.entity_query_language.factories import (
    an,
    variable,
)
from krrood.entity_query_language.verbalization.pipeline import (
    VerbalizationPipeline,
    verbalize_expression,
)
from krrood.patterns.field_metadata import FieldMetadata, GrammarMetadata
from krrood.entity_query_language.verbalization.rendering.formatter import (
    PlainFormatter,
)
from krrood.entity_query_language.verbalization.rendering.renderer import (
    HierarchicalRenderer,
)


@dataclass
class Position:
    """A 3-D position — three scalar attributes that group into one *"given that"* point."""

    x: float
    y: float
    z: float


@dataclass
class Pose:
    """A pose with two nested sub-objects — a nested match groups attributes per sub-object."""

    position: Position
    orientation: Position


def _hierarchical(expression) -> str:
    """:return: *expression* rendered as a plain-text indented bullet list (point per condition)."""
    return VerbalizationPipeline(HierarchicalRenderer(PlainFormatter())).verbalize(
        expression
    )


# ── Generate vs Find ─────────────────────────────────────────────────────────


def test_match_opens_with_generate():
    """A match verbalized on its own (no backend) opens with *"Generate"* — the default reading is
    generative (a construction description). The query alone no longer encodes find-vs-generate;
    that is backend-driven (see :mod:`...test_backend_performative`)."""
    assert verbalize_expression(an(Position)(x=1)).startswith("Generate")


def test_domain_carrying_match_also_opens_with_generate():
    """A domain-carrying ``an(...).from_(...)`` still opens with *"Generate"* by default: the domain
    does not by itself make it a search — that is the backend's call."""
    search = an(Position)(x=1, y=2).from_([Position(1, 2, 3)])
    assert verbalize_expression(search).startswith("Generate")


# ── given that: grouped equality conditions ──────────────────────────────────


def test_grouped_attributes_say_respectively():
    """Several equalities on one object aggregate into one *"… respectively"* point."""
    text = verbalize_expression(an(Position)(x=1, y=2, z=3))
    assert text == (
        "Generate a Position given that the x, y, and z of the Position are 1, 2, and 3 respectively"
    )


def test_single_attribute_pronominalises_and_uses_is_not_respectively():
    """A single equality is said through the shared comparator path, so it pronominalises to *"its
    x is …"* and never uses *"respectively"*."""
    text = verbalize_expression(an(Position)(x=5))
    assert text == "Generate a Position given that its x is 5"
    assert "respectively" not in text


def test_given_that_is_its_own_block_with_one_point_per_group():
    """The *"given that"* part is a sub-header block; each group is its own point."""
    text = _hierarchical(an(Position)(x=1, y=2, z=3))
    assert text == (
        "Generate a Position\n"
        "  given that\n"
        "    - the x, y, and z of the Position are 1, 2, and 3 respectively"
    )


def test_over_cap_singles_pronominalise_and_coordinate_with_and():
    """Beyond the grouping cap the singles are said through the shared comparator path — each
    pronominalised (*"its x is …"*) and Oxford-coordinated with a closing *"and"*."""

    @dataclass
    class _Quad:
        x: int
        w: int
        y: int
        t: int

    text = verbalize_expression(an(_Quad)(x=1, w=45, y=2, t=4))
    assert text == (
        "Generate a _Quad given that its x is 1, its w is 45, its y is 2, and its t is 4"
    )


# ── predict: Ellipsis values folded into the header ──────────────────────────


def test_all_ellipsis_predicts_in_header():
    """All-``Ellipsis`` values are generated → *"and predict its … values"* in the header."""
    text = verbalize_expression(an(Position)(x=..., y=..., z=...))
    assert text == "Generate a Position and predict its x, y, and z values"


def test_single_ellipsis_predicts_singular_value():
    """A single predicted attribute uses the singular *"value"*."""
    text = verbalize_expression(an(Position)(x=...))
    assert text == "Generate a Position and predict its x value"


def test_mixed_concrete_and_ellipsis():
    """Concrete kwargs go to *"given that"*; ``Ellipsis`` kwargs are predicted in the header."""
    text = verbalize_expression(an(Position)(x=1, y=...))
    assert text == ("Generate a Position and predict its y value given that its x is 1")


# ── where: free conditions as points ─────────────────────────────────────────


def test_where_conditions_are_their_own_block():
    """``.where(...)`` conditions form a *"where"* block, one point each, distinct from
    *"given that"*."""
    match = an(Position)(x=1)
    match.resolve()
    match.where(match.variable.y > 2)
    text = _hierarchical(match)
    assert text == (
        "Generate a Position\n"
        "  given that\n"
        "    - its x is 1\n"
        "  where\n"
        "    - its y is greater than 2"
    )


def test_where_only_match_has_no_given_that_block():
    """A match with only ``where`` conditions renders just the *"where"* block."""
    match = an(Position)()
    match.resolve()
    match.where(match.variable.x > 0)
    text = _hierarchical(match)
    assert text == "Generate a Position\n  where\n    - its x is greater than 0"


def test_where_folds_a_range_pair_into_one_between_point():
    """Complementary bounds on one chain fold into a single *"is between …"* point — the same
    conjunction reduction the ``AND`` / restriction assemblers apply, invoked over the flat
    ``where`` list."""
    match = an(Position)()
    match.resolve()
    match.where(match.variable.x > 0.0, match.variable.x < 5.0)
    text = _hierarchical(match)
    assert text == "Generate a Position\n  where\n    - its x is between 0.0 and 5.0"


# ── nested matches: per-sub-object grouping ──────────────────────────────────


def _nested_pose():
    """:return: An underspecified pose whose position and orientation are themselves
    underspecified (all-``Ellipsis``) matches."""
    return an(Pose)(
        position=an(Position)(x=..., y=..., z=...),
        orientation=an(Position)(x=..., y=..., z=...),
    )


def test_nested_predict_groups_per_sub_object():
    """Predicted attributes of a nested match group per sub-object into a *"predict"* block —
    *"x, y, and z of its position"* — never the raw ``Ellipsis`` literal."""
    text = _hierarchical(_nested_pose())
    assert text == (
        "Generate a Pose\n"
        "  and predict\n"
        "    - the x, y, and z of its position\n"
        "    - the x, y, and z of its orientation"
    )
    assert "Ellipsis" not in text


def test_nested_predict_with_where_range_on_sub_object():
    """A nested predict block coexists with a ``where`` block that folds a sub-object range."""
    pose = an(Pose)(position=an(Position)(x=..., y=..., z=...))
    pose.expression
    pose.where(pose.variable.position.x > 0.0, pose.variable.position.x < 5.0)
    text = _hierarchical(pose)
    assert text == (
        "Generate a Pose\n"
        "  and predict\n"
        "    - the x, y, and z of its position\n"
        "  where\n"
        "    - the x of its position is between 0.0 and 5.0"
    )


# ── leaf-value rendering: None, domains, concrete objects ────────────────────


class _Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclass
class _Widget:
    color: _Color
    size: int
    owner: object


def test_none_assignment_reads_as_has_no_separate_from_group():
    """An attribute set to ``None`` is a *"<object> has no <attr>"* point, pulled out of the
    *"… respectively"* group of the present attributes — never the raw ``None`` value.
    """
    text = verbalize_expression(
        an(Pose)(position=an(Position)(x=0.1), orientation=None)
    )
    assert "the Pose has no orientation" in text
    assert "None" not in text


def test_multiple_none_assignments_coordinate_under_has_no():
    """Several ``None`` attributes of one object coordinate under a single *"has no"*."""
    text = verbalize_expression(an(_Widget)(color=None, size=None, owner=object()))
    assert "has no color and size" in text


def test_domain_value_variable_lists_candidates():
    """A bounded enum-domain variable assigned to an attribute lists its candidates."""
    text = verbalize_expression(
        an(_Widget)(color=variable(_Color, [_Color.RED, _Color.GREEN, _Color.BLUE]))
    )
    assert "one of RED, GREEN, or BLUE" in text


def test_concrete_object_reads_as_a_specific_type():
    """A concrete object assignment reads *"a specific <Type>"* — identity, not its repr."""
    text = verbalize_expression(an(_Widget)(owner=Position(x=1.0, y=2.0, z=3.0)))
    assert "a specific Position" in text
    assert "Position(" not in text  # no repr leak


# ── object identity: identifying field qualifies "a specific <Type>" ──────────


@dataclass
class _NamedThing:
    name: str
    payload: object


@dataclass
class _Coded:
    serial: int = field(
        metadata=FieldMetadata(
            other_metadata=[GrammarMetadata(is_identifying_field=True)]
        ).as_dict()
    )
    name: str = ""


def test_concrete_object_qualified_by_conventional_name_field():
    """A concrete object with a conventional ``name`` field reads *"a specific X with name '…'"*."""
    text = verbalize_expression(
        an(_Widget)(owner=_NamedThing(name="door", payload=object()))
    )
    assert "a specific _NamedThing with name 'door'" in text


def test_concrete_object_uses_declared_identifying_attributes():
    """A field marked ``is_identifying_field`` controls which field identifies the object."""
    text = verbalize_expression(an(_Widget)(owner=_Coded(serial=7, name="x")))
    assert "a specific _Coded with serial 7" in text
    assert (
        "name" not in text.split("specific _Coded")[1]
    )  # declared field wins over 'name'


# ── "respectively" grouping: atomic scalars only, capped ─────────────────────


@dataclass
class _Trio:
    a: float
    b: float
    c: float


@dataclass
class _Quint:
    a: int
    b: int
    c: int
    d: int
    e: int


def test_atomic_scalars_group_under_respectively():
    """Up to three atomic scalar values coordinate under one *"… respectively"* point."""
    text = verbalize_expression(an(_Trio)(a=1.0, b=2.0, c=3.0))
    assert "the a, b, and c of the _Trio are 1.0, 2.0, and 3.0 respectively" in text


def test_over_cap_scalars_are_said_separately():
    """Beyond the cap, each assignment is its own point — no unreadable many-way zip."""
    text = verbalize_expression(an(_Quint)(a=1, b=2, c=3, d=4, e=5))
    assert "respectively" not in text
    assert "its a is 1" in text
    assert "its e is 5" in text


def test_compound_value_pulled_out_of_respectively_group():
    """A phrase-valued assignment (*"one of …"*) is said on its own; atomic ones still group."""
    text = verbalize_expression(an(_Trio)(a=1.0, b=2.0, c=variable(float, [7.0, 8.0])))
    assert "the a and b of the _Trio are 1.0 and 2.0 respectively" in text
    assert "its c is one of 7.0 or 8.0" in text
