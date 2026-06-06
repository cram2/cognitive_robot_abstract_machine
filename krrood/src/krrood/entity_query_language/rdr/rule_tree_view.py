"""
Textual, rule-level visualization of an EQL RDR rule tree.

This renders the rule tree the way an RDR expert thinks about it — one line per rule
(``if <conditions> then <conclusion>``), nested by refinement depth — rather than the
per-operation expression graph that :mod:`~krrood.entity_query_language.query_graph`
draws. It is deliberately dependency-light (just ``colorama``), matching the style of
:mod:`~krrood.entity_query_language.rdr.case_table`.

Each rule is coloured by what happened to it during the classification that is being
explained:

* **green**  — the rule *fired* (its condition was satisfied),
* **red**    — the rule was *evaluated* but its condition did not hold,
* **grey**   — the rule was *not evaluated* (a branch the evaluation short-circuited).

The status is read straight from the evaluation observers' id-sets (``satisfied`` /
``evaluated``); see :class:`~krrood.entity_query_language.rdr.observer.ClassificationTrace`.

To keep a large tree readable the rendered rows are *elided*: the first few rules are
shown, then a vertical-dots row, then the few rules ending at the rule that fired (so the
firing rule is always the last visible row).
"""

from __future__ import annotations

import enum

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Any, List, Optional
from uuid import UUID

from colorama import Fore, Style
from ordered_set import OrderedSet

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Literal, Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    OR,
    LogicalOperator,
    Not,
)
from krrood.entity_query_language.rdr.utils import _conclusions_of
from krrood.entity_query_language.rules.conclusion import Add, Conclusion
from krrood.entity_query_language.rules.conclusion_selector import (
    Alternative,
    ConclusionSelector,
    Next,
    Refinement,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.rdr.observer import ClassificationTrace

#: How many rules to show at the top before eliding the middle.
DEFAULT_HEAD = 3

#: How many rules to show at the bottom (ending on the firing rule) after the elision.
DEFAULT_TAIL = 3

#: Glyph used for the elided-rows marker.
_VERTICAL_DOTS = "⋮"

# Branch glyphs for the nesting connectors.
_GUIDE = "│  "  # "│  "  an ancestor branch continues below
_GAP = "   "  # the ancestor branch ended above
_BRANCH = "├─ "  # "├─ " this node has a younger sibling
_BRANCH_LAST = "└─ "  # "└─ " this node is the last of its siblings


class RuleStatus(enum.Enum):
    """What happened to a rule during the classification being explained."""

    FIRED = "fired"
    EVALUATED_NOT_FIRED = "evaluated"
    NOT_EVALUATED = "skipped"

    @property
    def color(self) -> str:
        """:return: The ANSI colour (colorama) this status is drawn in."""
        return {
            RuleStatus.FIRED: Fore.GREEN,
            RuleStatus.EVALUATED_NOT_FIRED: Fore.RED,
            RuleStatus.NOT_EVALUATED: Fore.LIGHTBLACK_EX,
        }[self]


@dataclass
class RuleView:
    """One rule (a condition plus its conclusion(s)) at a place in the rule tree.

    A flat, render-ready projection of a leaf condition node in the
    ``Refinement`` / ``Alternative`` / ``Next`` selector DAG. The ``condition`` is the
    node whose ``_id_`` the evaluation trackers key on, so status resolution is a plain
    membership test.
    """

    condition: SymbolicExpression
    """The leaf condition node carrying the rule's conclusion(s)."""
    conclusions: List[Add]
    """The ``Add`` conclusion(s) attached to :attr:`condition` (one for single-class)."""
    depth: int
    """Refinement-nesting depth (0 = a top-level rule)."""
    kind: str
    """How the rule relates to its predecessor: ``"if"`` / ``"else if"`` / ``"except if"``."""


def walk_rules(conditions_root: SymbolicExpression) -> List[RuleView]:
    """
    Flatten a rule-tree selector DAG into rules in classic RDR display order.

    A ``Refinement``'s right branch nests one level deeper (an *except-if*); an
    ``Alternative``'s right branch is a same-level sibling (an *else-if*). Leaf condition
    nodes (everything that is not a :class:`ConclusionSelector`) become :class:`RuleView`
    rows.

    :param conditions_root: The root of the rule tree's condition DAG.
    :return: The rules in pre-order, each tagged with its depth and kind.
    """
    rules: List[RuleView] = []

    def visit(node: SymbolicExpression, depth: int, kind: str) -> None:
        if isinstance(node, Refinement):
            visit(node.left, depth, kind)
            visit(node.right, depth + 1, "except if")
        elif isinstance(node, Alternative):
            visit(node.left, depth, kind)
            visit(node.right, depth, "else if")
        elif isinstance(node, Next):
            for child in node._operation_children_:
                visit(child, depth, kind)
        else:
            rules.append(
                RuleView(
                    condition=node,
                    conclusions=_conclusions_of(node),
                    depth=depth,
                    kind=kind,
                )
            )

    visit(conditions_root, 0, "if")
    return rules


def resolve_status(
    rule: RuleView,
    satisfied_ids: Optional[OrderedSet[UUID]],
    evaluated_ids: Optional[OrderedSet[UUID]],
) -> RuleStatus:
    """
    Classify a rule as fired / evaluated-not-fired / not-evaluated from the observer id-sets.

    :param rule: The rule whose ``condition`` node id is looked up.
    :param satisfied_ids: Condition ids whose truth value was True (``None`` ⇒ none).
    :param evaluated_ids: Expression ids that were evaluated at all (``None`` ⇒ none).
    :return: The :class:`RuleStatus` for the rule.
    """
    cid = rule.condition._id_
    if satisfied_ids is not None and cid in satisfied_ids:
        return RuleStatus.FIRED
    if evaluated_ids is not None and cid in evaluated_ids:
        return RuleStatus.EVALUATED_NOT_FIRED
    return RuleStatus.NOT_EVALUATED


# ---------------------------------------------------------------------------
# Compact symbolic formatting of conditions and conclusions.
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, enum.Enum):
        return value.name
    return str(value)


def _attribute_path(attr: Attribute) -> str:
    """:return: The attribute access path with the root subject variable dropped."""
    child = attr._child_
    if isinstance(child, Variable):
        return attr._attribute_name_
    return f"{format_condition(child)}.{attr._attribute_name_}"


def _format_conclusion_selector(expr: ConclusionSelector) -> str:
    """Render a :class:`ConclusionSelector` in a compact, readable form.

    ConclusionSelectors (Alternative, Refinement, Next) are control-flow nodes, not
    conditions. When they appear as guard expressions in backward-inference output
    the default dataclass ``repr`` is unreadable — it dumps all internal fields
    including ``_conclusions_``, parent references, and evaluation flags.

    This helper renders them via their child expressions instead.
    """
    match expr:
        case Alternative():
            return f"({format_condition(expr.left)} else {format_condition(expr.right)})"
        case Refinement():
            return f"({format_condition(expr.left)} except if {format_condition(expr.right)})"
        case Next():
            children = ", ".join(format_condition(c) for c in expr._operation_children_)
            return f"next ({children})"
        case _:
            return expr.__class__.__name__


def format_condition(expr: Any) -> str:
    """
    Render a condition expression as a compact, prefix-stripped string.

    e.g. ``case_variable.legs == 4`` becomes ``legs == 4``; an ``AND`` of comparators is
    joined with ``and``. Anything unrecognised falls back to its ``repr``.
    """
    match expr:
        case ConclusionSelector():
            return _format_conclusion_selector(expr)
        case Comparator():
            return f"{format_condition(expr.left)} {expr._name_} {format_condition(expr.right)}"
        case AND():
            return f"{format_condition(expr.left)} and {format_condition(expr.right)}"
        case OR():
            return f"{format_condition(expr.left)} or {format_condition(expr.right)}"
        case Not():
            return f"not {format_condition(expr._child_)}"
        case Attribute():
            return _attribute_path(expr)
        case Literal():
            return _format_value(expr._value_)
        case Variable():
            return expr._name_
        case _:
            return repr(expr)


def format_conclusion(add: Add) -> str:
    """:return: A compact ``attribute = value`` rendering of an ``Add`` conclusion."""
    variable = add.variable
    name = (
        variable._attribute_name_
        if isinstance(variable, Attribute)
        else format_condition(variable)
    )
    value = add.value
    if isinstance(value, Literal):
        value_str = _format_value(value._value_)
    elif isinstance(value, Variable):
        value_str = value._name_
    else:
        value_str = _format_value(value)
    return f"{name} = {value_str}"


def _format_conclusions(rule: RuleView) -> str:
    if not rule.conclusions:
        return "?"
    return ", ".join(format_conclusion(add) for add in rule.conclusions)


# ---------------------------------------------------------------------------
# Tree connectors, elision, and the renderer.
# ---------------------------------------------------------------------------


def _continues_at(depths: List[int], index: int, level: int) -> bool:
    """:return: True if the ancestor at ``level`` has a later sibling (draw a guide)."""
    for j in range(index + 1, len(depths)):
        if depths[j] < level:
            return False
        if depths[j] == level:
            return True
    return False


def _is_last_at(depths: List[int], index: int, level: int) -> bool:
    """:return: True if the node at ``index`` is the last of its siblings at ``level``."""
    for j in range(index + 1, len(depths)):
        if depths[j] < level:
            return True
        if depths[j] == level:
            return False
    return True


def _connector(depths: List[int], index: int) -> str:
    """Build the ``│ ├─ └─`` prefix for a node from the flat list of depths."""
    depth = depths[index]
    if depth == 0:
        return ""
    parts = [
        _GUIDE if _continues_at(depths, index, level) else _GAP
        for level in range(1, depth)
    ]
    parts.append(_BRANCH_LAST if _is_last_at(depths, index, depth) else _BRANCH)
    return "".join(parts)


@dataclass
class RuleTreeRenderer:
    """Renders a flat list of :class:`RuleView` rows as a coloured, elided text tree."""

    head: int = DEFAULT_HEAD
    """How many rules to show before the elision marker."""
    tail: int = DEFAULT_TAIL
    """How many rules (ending on the firing rule) to show after the elision marker."""
    use_color: bool = True
    """Whether to wrap each rule line in its status colour."""

    def render(
        self,
        rules: List[RuleView],
        satisfied_ids: Optional[OrderedSet[UUID]],
        evaluated_ids: Optional[OrderedSet[UUID]],
        fired_index: Optional[int],
    ) -> str:
        """
        :param rules: The rules in display order (from :func:`walk_rules`).
        :param satisfied_ids: Satisfied condition ids (for green).
        :param evaluated_ids: Evaluated expression ids (for red vs grey).
        :param fired_index: Index of the rule that fired; the elided tail ends here.
        :return: The multi-line rendered tree.
        """
        if not rules:
            return ""
        depths = [r.depth for r in rules]
        lines = [
            self._render_row(rule, _connector(depths, i), satisfied_ids, evaluated_ids)
            for i, rule in enumerate(rules)
        ]
        return "\n".join(self._elide(lines, len(rules), fired_index))

    def _render_row(
        self,
        rule: RuleView,
        connector: str,
        satisfied_ids: Optional[OrderedSet[UUID]],
        evaluated_ids: Optional[OrderedSet[UUID]],
    ) -> str:
        status = resolve_status(rule, satisfied_ids, evaluated_ids)
        text = f"{rule.kind} {format_condition(rule.condition)}  →  {_format_conclusions(rule)}"
        if self.use_color:
            text = f"{status.color}{text}{Style.RESET_ALL}"
        return f"{connector}{text}"

    def _elide(
        self, lines: List[str], total: int, fired_index: Optional[int]
    ) -> List[str]:
        """Keep the first :attr:`head` rows + the :attr:`tail` rows ending on the fired row."""
        anchor = fired_index if fired_index is not None else total - 1
        tail_start = max(0, anchor - self.tail + 1)
        # Contiguous (head reaches the tail window): show straight through to the anchor.
        if tail_start <= self.head:
            return lines[: anchor + 1]
        hidden = tail_start - self.head
        marker = f"{Fore.LIGHTBLACK_EX}{_GAP}{_VERTICAL_DOTS}  ({hidden} hidden){Style.RESET_ALL}"
        return lines[: self.head] + [marker] + lines[tail_start : anchor + 1]


def _fired_index(
    rules: List[RuleView], firing_anchor_id: Optional[UUID]
) -> Optional[int]:
    if firing_anchor_id is None:
        return None
    for i, rule in enumerate(rules):
        if rule.condition._id_ == firing_anchor_id:
            return i
    return None


def render_rule_tree(
    trace: "ClassificationTrace",
    *,
    head: int = DEFAULT_HEAD,
    tail: int = DEFAULT_TAIL,
    use_color: bool = True,
) -> str:
    """
    Render the rule tree described by a :class:`ClassificationTrace` as coloured text.

    :param trace: The classification trace carrying the rule-tree root and the observer
        id-sets that drive the colours and the elision anchor.
    :param head: Rules to show before the elision marker.
    :param tail: Rules to show after it (ending on the firing rule).
    :param use_color: Whether to colour rows by status.
    :return: The rendered tree, or ``""`` when the tree is empty.
    """
    if trace.rule_tree_root is None:
        return ""
    rules = walk_rules(trace.rule_tree_root)
    fired_index = _fired_index(rules, trace.firing_anchor_id)
    renderer = RuleTreeRenderer(head=head, tail=tail, use_color=use_color)
    return renderer.render(
        rules,
        trace.satisfied_condition_ids,
        trace.evaluated_expression_ids,
        fired_index,
    )
