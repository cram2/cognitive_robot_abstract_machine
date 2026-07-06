"""
Tests for the fragment model, formatters, renderers, and VerbalizationPipeline.

Coverage:
- VerbalizationFragment tree structure: SemanticRole tagging for variables, aggregations, keywords, operators
- PlainFormatter: identity pass-through
- ANSIFormatter: wraps text in ANSI escape sequences; plain for PLAIN role
- HTMLFormatter: wraps text in <span> tags; plain for PLAIN role; uses &nbsp; / <br>
- BulletStyle / IndentSize enums
- ParagraphRenderer: flattens block structure to prose
- HierarchicalRenderer: indents block items as bullet points; no isinstance coupling
- VerbalizationPipeline: end-to-end with each factory
"""

from __future__ import annotations

from dataclasses import dataclass

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.factories import (
    an,
    and_,
    or_,
    entity,
    variable,
    inference,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    PhraseFragment,
    RoleFragment,
    VerbalizationFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline
from krrood.entity_query_language.verbalization.rendering.formatter import (
    ANSIFormatter,
    BulletStyle,
    HTMLFormatter,
    IndentSize,
    PlainFormatter,
)
from krrood.entity_query_language.verbalization.rendering.renderer import (
    HierarchicalRenderer,
    ParagraphRenderer,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
)
from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
from ...dataset.semantic_world_like_classes import (
    Drawer,
    FixedConnection,
    Handle,
    PrismaticConnection,
)


@dataclass
class _Robot:
    battery: int
    name: str


@dataclass
class _Task:
    completed: bool


# ── VerbalizationFragment helpers ───────────────────────────────────────────────────────────


def _collect_roles(fragment: VerbalizationFragment) -> list[SemanticRole]:
    """Recursively collect all SemanticRole values from a fragment tree."""
    match fragment:
        case RoleFragment(role=role):
            return [role]
        case PhraseFragment(parts=parts):
            return [r for p in parts for r in _collect_roles(p)]
        case BlockFragment(header=header, items=items):
            result = _collect_roles(header) if header else []
            return result + [r for item in items for r in _collect_roles(item)]
        case _:
            return []


def _collect_role_texts(
    fragment: VerbalizationFragment, role: SemanticRole
) -> list[str]:
    """Return all text values in the tree that carry *role*."""
    match fragment:
        case RoleFragment(text=text, role=r) if r == role:
            return [text]
        case PhraseFragment(parts=parts):
            return [t for p in parts for t in _collect_role_texts(p, role)]
        case BlockFragment(header=header, items=items):
            result = _collect_role_texts(header, role) if header else []
            return result + [
                t for item in items for t in _collect_role_texts(item, role)
            ]
        case _:
            return []


# ── VerbalizationFragment structure tests ───────────────────────────────────────────────────


def test_variable_fragment_carries_variable_role():
    x = variable(_Robot, [])
    frag = EQLVerbalizer().build(x)
    assert SemanticRole.VARIABLE in _collect_roles(frag)


def test_aggregation_fragment_carries_aggregation_role():
    x = variable(_Robot, [])
    frag = EQLVerbalizer().build(eql.count(x))
    assert SemanticRole.AGGREGATION in _collect_roles(frag)


def test_aggregation_role_text_is_keyword_phrase():
    x = variable(_Robot, [])
    frag = EQLVerbalizer().build(eql.sum(x))
    agg_texts = _collect_role_texts(frag, SemanticRole.AGGREGATION)
    assert any("sum" in t for t in agg_texts)


def test_comparator_fragment_carries_operator_role():
    x = variable(_Robot, [])
    frag = EQLVerbalizer().build(x.battery > 50)
    assert SemanticRole.OPERATOR in _collect_roles(frag)


def test_query_find_carries_keyword_role():
    r = variable(_Robot, [])
    frag = EQLVerbalizer().build(an(entity(r)))
    keyword_texts = _collect_role_texts(frag, SemanticRole.KEYWORD)
    assert any("Find" in t for t in keyword_texts)


def test_query_where_carries_keyword_role():
    r = variable(_Robot, [])
    frag = EQLVerbalizer().build(an(entity(r).where(r.battery > 50)))
    keyword_texts = _collect_role_texts(frag, SemanticRole.KEYWORD)
    assert any("whose" in t for t in keyword_texts)


def test_rule_if_then_carry_keyword_role(doors_and_drawers_world):
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    pc = variable(PrismaticConnection, world.connections)
    fc = an(FixedConnection).from_(world.connections)(parent=pc.child, child=handle)
    drawer_var = inference(Drawer)(
        container=fc.expression.parent, handle=fc.expression.child
    )
    frag = EQLVerbalizer().build(entity(drawer_var))
    keyword_texts = _collect_role_texts(frag, SemanticRole.KEYWORD)
    assert any("If" in t for t in keyword_texts)
    assert any("then" in t for t in keyword_texts)


def test_logical_for_all_carries_logical_role():
    x = variable(_Robot, [])
    frag = EQLVerbalizer().build(eql.for_all(x, x.battery > 0))
    assert SemanticRole.LOGICAL in _collect_roles(frag)


def test_literal_carries_literal_role():
    from krrood.entity_query_language.core.variable import Literal

    lit = Literal(_value_=42)
    frag = EQLVerbalizer().build(lit)
    assert SemanticRole.LITERAL in _collect_roles(frag)


def test_where_clause_condition_preserves_semantic_roles():
    """Regression: where-clause condition must keep RoleFragment roles in the fragment tree.

    _verbalize_query_body_ was calling delegate.verbalize() (which flattens to plain string)
    then re-wrapping the result in _word() — stripping all OPERATOR/ATTRIBUTE/LITERAL roles
    from the condition.  Only 'Find', the selected variable, and 'such that' were colored;
    the rest of the sentence was one uncolored WordFragment.

    Reproduces: VerbalizationPipeline.ansi().verbalize(query) output where only
    'Find', 'Robot', and 'such that' appear in color.
    """

    @dataclass
    class _Mission:
        priority: int
        assigned_to: _Robot

    robot = variable(_Robot, [])
    mission = variable(_Mission, [])
    q = an(
        entity(robot).where(
            and_(
                mission.assigned_to == robot,
                mission.priority > 2,
            )
        )
    )
    frag = EQLVerbalizer().build(q)

    op_texts = _collect_role_texts(frag, SemanticRole.OPERATOR)
    assert any("greater than" in t for t in op_texts), (
        f"OPERATOR role ('greater than') missing from where clause — "
        f"all OPERATOR texts found: {op_texts}"
    )

    attr_texts = _collect_role_texts(frag, SemanticRole.ATTRIBUTE)
    assert any("priority" in t for t in attr_texts), (
        f"ATTRIBUTE role ('priority') missing from where clause — "
        f"all ATTRIBUTE texts found: {attr_texts}"
    )

    lit_texts = _collect_role_texts(frag, SemanticRole.LITERAL)
    assert any("2" in t for t in lit_texts), (
        f"LITERAL role ('2') missing from where clause — "
        f"all LITERAL texts found: {lit_texts}"
    )


def test_query_is_block_fragment():
    r = variable(_Robot, [])
    frag = EQLVerbalizer().build(an(entity(r).where(r.battery > 50)))
    assert isinstance(frag, BlockFragment)


def test_rule_is_block_fragment(doors_and_drawers_world):
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    pc = variable(PrismaticConnection, world.connections)
    fc = an(FixedConnection).from_(world.connections)(parent=pc.child, child=handle)
    drawer_var = inference(Drawer)(
        container=fc.expression.parent, handle=fc.expression.child
    )
    frag = EQLVerbalizer().build(entity(drawer_var))
    assert isinstance(frag, BlockFragment)


# ── PlainFormatter ─────────────────────────────────────────────────────────────


def test_plain_formatter_returns_text_unchanged():
    f = PlainFormatter()
    assert f.colorize("hello", SemanticRole.KEYWORD) == "hello"
    assert f.colorize("hello", SemanticRole.AGGREGATION) == "hello"
    assert f.colorize("hello", SemanticRole.PLAIN) == "hello"


def test_plain_formatter_space_is_space():
    assert PlainFormatter().space == " "


def test_plain_formatter_newline_is_unix_newline():
    assert PlainFormatter().newline == "\n"


# ── ANSIFormatter ──────────────────────────────────────────────────────────────


def test_ansi_formatter_wraps_keyword():
    f = ANSIFormatter()
    result = f.colorize("If", SemanticRole.KEYWORD)
    assert result.startswith("\033[38;2;")
    assert "If" in result
    assert result.endswith("\033[0m")


def test_ansi_formatter_plain_role_no_escape():
    f = ANSIFormatter()
    result = f.colorize("the", SemanticRole.PLAIN)
    assert result == "the"


def test_ansi_formatter_aggregation_uses_red_orange():
    f = ANSIFormatter()
    result = f.colorize("sum of", SemanticRole.AGGREGATION)
    # #F54927 → R=245, G=73, B=39
    assert "245;73;39" in result


def test_ansi_formatter_variable_uses_cornflowerblue():
    f = ANSIFormatter()
    result = f.colorize("Robot", SemanticRole.VARIABLE)
    # cornflowerblue → R=100, G=149, B=237
    assert "100;149;237" in result


def test_ansi_formatter_space_is_space():
    assert ANSIFormatter().space == " "


def test_ansi_formatter_newline_is_unix_newline():
    assert ANSIFormatter().newline == "\n"


# ── HTMLFormatter ──────────────────────────────────────────────────────────────


def test_html_formatter_wraps_keyword_in_span():
    f = HTMLFormatter()
    result = f.colorize("If", SemanticRole.KEYWORD)
    assert result == '<span style="color:#eded18">If</span>'


def test_html_formatter_plain_role_no_span():
    f = HTMLFormatter()
    result = f.colorize("the", SemanticRole.PLAIN)
    assert result == "the"


def test_html_formatter_aggregation():
    f = HTMLFormatter()
    result = f.colorize("number of", SemanticRole.AGGREGATION)
    assert "#F54927" in result
    assert "number of" in result


def test_html_formatter_variable():
    f = HTMLFormatter()
    result = f.colorize("Robot", SemanticRole.VARIABLE)
    assert "cornflowerblue" in result
    assert "Robot" in result


def test_html_formatter_space_is_nbsp():
    assert HTMLFormatter().space == "&nbsp;"


def test_html_formatter_newline_is_br():
    assert HTMLFormatter().newline == "<br>"


# ── BulletStyle / IndentSize enums ─────────────────────────────────────────────


def test_bullet_style_dash_value():
    assert BulletStyle.DASH.value == "-"


def test_bullet_style_dot_value():
    assert BulletStyle.DOT.value == "•"


def test_bullet_style_asterisk_value():
    assert BulletStyle.ASTERISK.value == "*"


def test_indent_size_two_spaces_value():
    assert IndentSize.TWO_SPACES.value == "  "


def test_indent_size_four_spaces_value():
    assert IndentSize.FOUR_SPACES.value == "    "


def test_indent_size_tab_value():
    assert IndentSize.TAB.value == "\t"


# ── ParagraphRenderer ──────────────────────────────────────────────────────────


def test_paragraph_renderer_word():
    r = ParagraphRenderer()
    assert r.render(WordFragment("hello")) == "hello"


def test_paragraph_renderer_role_fragment_plain():
    r = ParagraphRenderer(PlainFormatter())
    assert r.render(RoleFragment("Robot", SemanticRole.VARIABLE)) == "Robot"


def test_paragraph_renderer_phrase():
    r = ParagraphRenderer()
    frag = PhraseFragment([WordFragment("a"), WordFragment("Robot")])
    assert r.render(frag) == "a Robot"


def test_paragraph_renderer_block_flattens_to_prose():
    r = ParagraphRenderer(PlainFormatter())
    block = BlockFragment(
        header=RoleFragment("Find", SemanticRole.KEYWORD),
        items=[
            PhraseFragment(
                [RoleFragment("such that", SemanticRole.KEYWORD), WordFragment("x > 5")]
            ),
        ],
    )
    result = r.render(block)
    assert "Find" in result
    assert "such that" in result
    assert "x > 5" in result


def test_paragraph_renderer_block_no_header():
    r = ParagraphRenderer()
    block = BlockFragment(header=None, items=[WordFragment("a"), WordFragment("b")])
    result = r.render(block)
    assert "a" in result and "b" in result


def test_paragraph_html_formatter_uses_nbsp_in_block_header_join():
    """ParagraphRenderer uses formatter.space to join the header to its prose."""
    r = ParagraphRenderer(HTMLFormatter())
    block = BlockFragment(
        header=RoleFragment("Find", SemanticRole.PLAIN),
        items=[WordFragment("Robot")],
    )
    result = r.render(block)
    assert "&nbsp;" in result


# ── HierarchicalRenderer ───────────────────────────────────────────────────────


def test_hierarchical_renderer_block_has_header_line():
    r = HierarchicalRenderer(PlainFormatter())
    block = BlockFragment(
        header=RoleFragment("If", SemanticRole.KEYWORD),
        items=[WordFragment("there's a Handle")],
    )
    result = r.render(block)
    lines = result.splitlines()
    assert lines[0] == "If"
    assert any("Handle" in line for line in lines[1:])


def test_hierarchical_renderer_items_are_indented():
    r = HierarchicalRenderer(
        PlainFormatter(), indent_size=IndentSize.TWO_SPACES, bullet=BulletStyle.DASH
    )
    block = BlockFragment(
        header=RoleFragment("Find", SemanticRole.KEYWORD),
        items=[WordFragment("a Robot"), WordFragment("b Something")],
    )
    result = r.render(block)
    lines = result.splitlines()
    item_lines = [l for l in lines if "Robot" in l or "Something" in l]
    for line in item_lines:
        assert line.startswith("  - ")


def test_hierarchical_renderer_nested_block_deepens_indent():
    r = HierarchicalRenderer(
        PlainFormatter(), indent_size=IndentSize.TWO_SPACES, bullet=BulletStyle.DASH
    )
    inner = BlockFragment(
        header=RoleFragment("such that", SemanticRole.KEYWORD),
        items=[WordFragment("battery > 50")],
    )
    outer = BlockFragment(
        header=RoleFragment("Find", SemanticRole.KEYWORD),
        items=[WordFragment("a Robot"), inner],
    )
    result = r.render(outer)
    lines = result.splitlines()
    battery_line = next(l for l in lines if "battery" in l)
    # Inner block items are at depth 2 → "    - battery > 50"
    assert battery_line.startswith("    ")


def test_hierarchical_renderer_custom_bullet():
    r = HierarchicalRenderer(PlainFormatter(), bullet=BulletStyle.DOT)
    block = BlockFragment(
        header=None,
        items=[WordFragment("item one"), WordFragment("item two")],
    )
    result = r.render(block)
    assert "• item one" in result
    assert "• item two" in result


def test_hierarchical_bullet_style_dot_produces_dot_bullet():
    r = HierarchicalRenderer(PlainFormatter(), bullet=BulletStyle.DOT)
    block = BlockFragment(header=None, items=[WordFragment("x")])
    assert "•" in r.render(block)


def test_hierarchical_bullet_style_asterisk_produces_asterisk():
    r = HierarchicalRenderer(PlainFormatter(), bullet=BulletStyle.ASTERISK)
    block = BlockFragment(header=None, items=[WordFragment("x")])
    assert "*" in r.render(block)


def test_hierarchical_indent_four_spaces_indents_items_by_four():
    r = HierarchicalRenderer(PlainFormatter(), indent_size=IndentSize.FOUR_SPACES)
    block = BlockFragment(
        header=WordFragment("H"),
        items=[WordFragment("item")],
    )
    result = r.render(block)
    item_line = next(l for l in result.split("\n") if "item" in l)
    assert item_line.startswith("    ")


# ── HierarchicalRenderer formatter decoupling ─────────────────────────────────


def test_hierarchical_html_formatter_joins_with_br():
    """HTMLFormatter.newline drives line separation — no hardcoded \\n."""
    r = HierarchicalRenderer(HTMLFormatter())
    block = BlockFragment(header=None, items=[WordFragment("a"), WordFragment("b")])
    result = r.render(block)
    assert "<br>" in result


def test_hierarchical_ansi_formatter_joins_with_newline_not_br():
    r = HierarchicalRenderer(ANSIFormatter())
    block = BlockFragment(header=None, items=[WordFragment("a"), WordFragment("b")])
    result = r.render(block)
    assert "\n" in result
    assert "<br>" not in result


def test_hierarchical_custom_formatter_newline_controls_line_separation():
    """Proves no isinstance coupling: any Formatter subclass controls the separator."""

    class PipeFormatter(PlainFormatter):
        @property
        def newline(self) -> str:
            return "|"

    r = HierarchicalRenderer(PipeFormatter())
    block = BlockFragment(header=None, items=[WordFragment("a"), WordFragment("b")])
    result = r.render(block)
    assert "|" in result
    assert "\n" not in result


# ── ParagraphRenderer with HTMLFormatter ──────────────────────────────────────


def test_paragraph_html_query_contains_find_span():
    r = variable(_Robot, [])
    text = VerbalizationPipeline.html().verbalize(an(entity(r).where(r.battery > 50)))
    assert '<span style="color' in text
    assert "Find" in text


def test_paragraph_html_aggregation_is_colored():
    r = variable(_Robot, [])
    text = VerbalizationPipeline.html().verbalize(an(entity(eql.count(r))))
    assert "#F54927" in text


# ── HierarchicalRenderer end-to-end ───────────────────────────────────────────


def test_hierarchical_plain_query_structure():
    r = variable(_Robot, [])
    # An OR condition stays a residual "such that" bullet (it is not a groupable
    # single-hop subject predicate), so the hierarchical indentation is exercised.
    text = VerbalizationPipeline(HierarchicalRenderer(PlainFormatter())).verbalize(
        an(entity(r).where(or_(r.battery > 50, r.battery < 10)))
    )
    lines = text.splitlines()
    assert any("Find" in l for l in lines)
    assert any("such that" in l for l in lines)
    assert any("battery" in l for l in lines)
    # where clause must be indented relative to Find
    find_line = next(l for l in lines if "Find" in l)
    where_line = next(l for l in lines if "such that" in l)
    assert len(where_line) - len(where_line.lstrip()) > len(find_line) - len(
        find_line.lstrip()
    )


def test_hierarchical_plain_rule_structure(doors_and_drawers_world):
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    pc = variable(PrismaticConnection, world.connections)
    fc = an(FixedConnection).from_(world.connections)(parent=pc.child, child=handle)
    drawer_var = inference(Drawer)(
        container=fc.expression.parent, handle=fc.expression.child
    )

    text = VerbalizationPipeline(HierarchicalRenderer(PlainFormatter())).verbalize(
        entity(drawer_var)
    )
    lines = text.splitlines()
    assert any("If" in l for l in lines)
    assert any("then" in l for l in lines)
    assert any("Handle" in l for l in lines)
    assert any("Drawer" in l for l in lines)


# ── Rule fragment structure tests ─────────────────────────────────────────────


def _find_block_with_keyword(
    fragment: VerbalizationFragment, keyword: str
) -> BlockFragment | None:
    """Return the first BlockFragment whose header text contains keyword."""
    if not isinstance(fragment, BlockFragment):
        return None
    if fragment.header is not None and keyword in flatten_fragment_to_plain_text(
        fragment.header
    ):
        return fragment
    for item in fragment.items:
        found = _find_block_with_keyword(item, keyword)
        if found is not None:
            return found
    return None


def _drawer_rule_fragment(doors_and_drawers_world) -> VerbalizationFragment:
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    pc = variable(PrismaticConnection, world.connections)
    fc = an(FixedConnection).from_(world.connections)(parent=pc.child, child=handle)
    drawer_var = inference(Drawer)(
        container=fc.expression.parent, handle=fc.expression.child
    )
    return EQLVerbalizer().build(entity(drawer_var))


def test_rule_condition_whose_is_keyword_role(doors_and_drawers_world):
    frag = _drawer_rule_fragment(doors_and_drawers_world)
    keyword_texts = _collect_role_texts(frag, SemanticRole.KEYWORD)
    assert "whose" in keyword_texts


def test_rule_condition_attribute_carries_attribute_role(doors_and_drawers_world):
    frag = _drawer_rule_fragment(doors_and_drawers_world)
    attr_texts = _collect_role_texts(frag, SemanticRole.ATTRIBUTE)
    assert any("parent" in t for t in attr_texts)
    assert any("child" in t for t in attr_texts)


def test_attribute_chain_owner_carries_variable_role(doors_and_drawers_world):
    frag = _drawer_rule_fragment(doors_and_drawers_world)
    variable_texts = _collect_role_texts(frag, SemanticRole.VARIABLE)
    assert any("PrismaticConnection" in t for t in variable_texts)
    assert any("FixedConnection" in t for t in variable_texts)


def test_rule_if_antecedent_repeats_whose_per_condition(doors_and_drawers_world):
    """An antecedent renders as one phrase — the existential intro woven with its conditions, each
    condition prefixed with its own *"whose"* and joined *"whose …, and whose …"* (the query
    restriction form)."""
    frag = _drawer_rule_fragment(doors_and_drawers_world)
    if_block = _find_block_with_keyword(frag, "If")
    assert if_block is not None
    assert if_block.items, "IF clause must contain at least one antecedent"
    texts = [flatten_fragment_to_plain_text(item) for item in if_block.items]
    antecedent = next(t for t in texts if "FixedConnection" in t)
    assert antecedent.startswith("there's a FixedConnection whose")
    assert ", and whose " in antecedent
    # "whose" is repeated per condition, not shared
    assert antecedent.count("whose") == 2


def test_rule_then_consequent_repeats_whose_per_binding(doors_and_drawers_world):
    """The THEN clause is a single consequent phrase: the existential intro with each field binding
    prefixed by its own *"whose"* and joined *"whose …, and whose …"*."""
    frag = _drawer_rule_fragment(doors_and_drawers_world)
    then_block = _find_block_with_keyword(frag, "then")
    assert then_block is not None
    assert len(then_block.items) == 1
    text = flatten_fragment_to_plain_text(then_block.items[0])
    assert text.startswith("there's a Drawer whose")
    assert ", and whose " in text
    assert text.count("whose") == 2


# ── VerbalizationPipeline factories ───────────────────────────────────────────


def test_pipeline_plain_matches_verbalize_expression():
    from krrood.entity_query_language.verbalization.pipeline import (
        verbalize_expression,
    )

    r = variable(_Robot, [])
    q = an(entity(r).where(r.battery > 50))
    assert VerbalizationPipeline.plain().verbalize(q) == verbalize_expression(q)


def test_pipeline_ansi_contains_escape_codes():
    r = variable(_Robot, [])
    text = VerbalizationPipeline.ansi().verbalize(an(entity(r)))
    assert "\033[" in text


def test_pipeline_html_contains_span():
    r = variable(_Robot, [])
    text = VerbalizationPipeline.html().verbalize(an(entity(r)))
    assert "<span" in text


def test_pipeline_html_hierarchical_has_br():
    r = variable(_Robot, [])
    text = VerbalizationPipeline.html(hierarchical=True).verbalize(
        an(entity(r).where(r.battery > 50))
    )
    assert "<br>" in text


def test_pipeline_html_hierarchical_has_br_between_items_on_rule(
    doors_and_drawers_world,
):
    drawer_fragment = _drawer_rule_fragment(doors_and_drawers_world)
    text = VerbalizationPipeline.html(hierarchical=True).verbalize_fragment(
        drawer_fragment
    )
    assert "<br>" in text


def test_pipeline_ansi_hierarchical_has_newlines_on_rule(doors_and_drawers_world):
    drawer_fragment = _drawer_rule_fragment(doors_and_drawers_world)
    text = VerbalizationPipeline.ansi(hierarchical=True).verbalize_fragment(
        drawer_fragment
    )
    assert "\n" in text
    assert "<br>" not in text
