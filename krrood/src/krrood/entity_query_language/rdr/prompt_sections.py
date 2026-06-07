"""Declarative prompt sections for the EQL-RDR interactive expert shell.

Each :class:`PromptSection` pairs a predicate (:attr:`~PromptSection.applicable`) with a
line producer (:attr:`~PromptSection.lines`).  The full prompt is assembled by iterating
:data:`PROMPT_SECTIONS` and collecting the output of every applicable section — a
*Composite / Pipeline of Specifications* rather than a nested ``if``-cascade.

New prompt situations become new :class:`PromptSection` entries in :data:`PROMPT_SECTIONS`;
existing sections are never modified to accommodate new cases (open/closed principle).
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Callable, List

from krrood.entity_query_language.rdr.expert import ANSWER_NAME, CONCLUSION_NAME
from krrood.entity_query_language.rdr.interface import AnswerRequest, CaseContext
from krrood.entity_query_language.rdr.prompt_examples import (
    build_conclusion_example,
    build_conditions_example,
)
from krrood.entity_query_language.rdr.rule_tree_view import format_condition

if TYPE_CHECKING:
    from krrood.entity_query_language.rdr.interactive import Palette

#: IPython line magic that re-displays the rule tree.
SHOW_TREE_MAGIC = "show_tree"

#: IPython line magic that re-displays the how-to-answer guidance.
HELP_MAGIC = "help"

#: IPython line magic that re-displays task-specific aid output.
AID_MAGIC = "aid"


@dataclass
class RenderContext:
    """Everything a :class:`PromptSection` needs to decide applicability and produce lines.

    :param case: The case context for the current interaction.
    :param requests: The answer requests for the current interaction.
    :param palette: The colour/styling palette for the current shell.
    """

    case: CaseContext
    requests: List[AnswerRequest]
    palette: Palette

    @property
    def has_target(self) -> bool:
        """:return: True when a ground-truth target conclusion was supplied."""
        return self.case.has_target

    @property
    def has_current_conclusion(self) -> bool:
        """:return: True when the RDR currently concludes something for this case."""
        return self.case.has_current_conclusion

    @property
    def is_conclusion_request(self) -> bool:
        """:return: True when the interaction is asking for a conclusion (no-target path)."""
        return any(r.name == CONCLUSION_NAME for r in self.requests)

    @property
    def is_conditions_request(self) -> bool:
        """:return: True when the interaction is asking for conditions."""
        return any(r.name == ANSWER_NAME for r in self.requests)

    @property
    def has_suggested_condition(self) -> bool:
        """:return: True when a suggested condition hint is available from the resolver."""
        return self.case.suggested_condition is not None


@dataclass(frozen=True)
class PromptSection:
    """One declarative, open-closed unit of prompt content.

    :param name: Identifier for debugging and documentation.
    :param applicable: Returns ``True`` when this section should appear in the prompt.
    :param lines: Produces the text lines when :attr:`applicable` returns ``True``.
    """

    name: str
    applicable: Callable[[RenderContext], bool]
    lines: Callable[[RenderContext], List[str]]


# ---------------------------------------------------------------------------
# Section line producers — one private function per section.
# ---------------------------------------------------------------------------


def _ground_truth_conclusion(ctx: RenderContext) -> List[str]:
    return [
        ctx.palette.label("Ground-truth conclusion: ")
        + ctx.palette.good(repr(ctx.case.target_conclusion))
    ]


def _current_conclusion_vs_target(ctx: RenderContext) -> List[str]:
    value = repr(ctx.case.current_conclusion)
    styled = (
        ctx.palette.good(value)
        if ctx.case.current_conclusion == ctx.case.target_conclusion
        else ctx.palette.wrong(value)
    )
    return [ctx.palette.label("Current conclusion: ") + styled]


def _no_rule_fired_known_target(ctx: RenderContext) -> List[str]:
    return [
        ctx.palette.label("No rule fired for this case."),
        ctx.palette.label("Write a ")
        + ctx.palette.keyword("condition")
        + ctx.palette.label(" that fires for it."),
    ]


def _conflict_resolution(ctx: RenderContext) -> List[str]:
    target = repr(ctx.case.target_conclusion)
    current = repr(ctx.case.current_conclusion)
    lines: List[str] = []
    if ctx.case.trace is not None:
        lines.append(
            ctx.palette.label("The condition ")
            + ctx.palette.code(format_condition(ctx.case.trace.firing_anchor))
            + ctx.palette.label(" concluded ")
            + ctx.palette.strong_wrong(current)
            + ctx.palette.label(" for this case while it should be ")
            + ctx.palette.good(target)
            + ctx.palette.label(".")
        )
    else:
        lines.append(
            ctx.palette.label("The RDR concluded ")
            + ctx.palette.strong_wrong(current)
            + ctx.palette.label(" while it should be ")
            + ctx.palette.good(target)
            + ctx.palette.label(".")
        )
    lines.append(
        ctx.palette.label(
            "Provide a condition that helps us detect this exceptional case"
        )
        + ctx.palette.label(" (and similar ones) such that we conclude ")
        + ctx.palette.good(target)
        + ctx.palette.label(" instead.")
    )
    return lines


def _labelling_has_current(ctx: RenderContext) -> List[str]:
    return [
        ctx.palette.label("The RDR currently concludes ")
        + ctx.palette.neutral(repr(ctx.case.current_conclusion))
        + ctx.palette.label(" — is that correct?")
        + ctx.palette.label(
            " If so, skip (press CTRL+D), else provide the correct conclusion."
        )
    ]


def _labelling_fired_anchor(ctx: RenderContext) -> List[str]:
    return [
        ctx.palette.label("It fired on ")
        + ctx.palette.code(format_condition(ctx.case.trace.firing_anchor))
        + ctx.palette.label(".")
    ]


def _labelling_no_rule(ctx: RenderContext) -> List[str]:
    return [ctx.palette.label("No rule fired — what should this case conclude?")]


def _allowed_values(ctx: RenderContext) -> List[str]:
    domain = ctx.case.conclusion_domain
    if domain.is_enumerable:
        return [
            ctx.palette.label("Choose one of: ") + ctx.palette.code(domain.display())
        ]
    return [
        ctx.palette.label("Conclusion type: ") + ctx.palette.code(domain.type_display)
    ]


def _contextual_example(ctx: RenderContext) -> List[str]:
    if ctx.is_conclusion_request:
        example = build_conclusion_example(ctx)
    else:
        example = build_conditions_example(ctx)
    return [ctx.palette.hint(example)]


def _help_hint(ctx: RenderContext) -> List[str]:
    magics = f"%{HELP_MAGIC}"
    if ctx.case.aids:
        magics += f" / %{AID_MAGIC}"
    return [ctx.palette.hint(f"Type {magics} for help with this case.")]


def _auto_resolution_hint(ctx: RenderContext) -> List[str]:
    return [
        ctx.palette.hint("Suggested condition (auto-resolved): ")
        + ctx.palette.code(format_condition(ctx.case.suggested_condition))
    ]


# ---------------------------------------------------------------------------
# Registry — the single extension point for new prompt situations.
# New prompt situations = append a PromptSection; never modify existing ones.
# ---------------------------------------------------------------------------

PROMPT_SECTIONS: List[PromptSection] = [
    PromptSection(
        name="ground_truth_conclusion",
        applicable=lambda ctx: ctx.has_target,
        lines=_ground_truth_conclusion,
    ),
    PromptSection(
        name="current_conclusion_vs_target",
        applicable=lambda ctx: ctx.has_target,
        lines=_current_conclusion_vs_target,
    ),
    PromptSection(
        name="no_rule_fired_known_target",
        applicable=lambda ctx: ctx.has_target and not ctx.has_current_conclusion,
        lines=_no_rule_fired_known_target,
    ),
    PromptSection(
        name="conflict_resolution",
        applicable=lambda ctx: (
            ctx.has_target
            and ctx.has_current_conclusion
            and ctx.case.current_conclusion != ctx.case.target_conclusion
        ),
        lines=_conflict_resolution,
    ),
    PromptSection(
        name="labelling_has_current",
        applicable=lambda ctx: not ctx.has_target and ctx.has_current_conclusion,
        lines=_labelling_has_current,
    ),
    PromptSection(
        name="labelling_fired_anchor",
        applicable=lambda ctx: (
            not ctx.has_target
            and ctx.has_current_conclusion
            and ctx.case.trace is not None
            and ctx.case.trace.firing_anchor is not None
        ),
        lines=_labelling_fired_anchor,
    ),
    PromptSection(
        name="labelling_no_rule",
        applicable=lambda ctx: not ctx.has_target and not ctx.has_current_conclusion,
        lines=_labelling_no_rule,
    ),
    PromptSection(
        name="allowed_values",
        applicable=lambda ctx: (
            not ctx.has_target and ctx.case.conclusion_domain is not None
        ),
        lines=_allowed_values,
    ),
    PromptSection(
        name="auto_resolution_hint",
        applicable=lambda ctx: ctx.has_suggested_condition,
        lines=_auto_resolution_hint,
    ),
    PromptSection(
        name="contextual_example",
        applicable=lambda ctx: True,
        lines=_contextual_example,
    ),
    PromptSection(
        name="help_hint",
        applicable=lambda ctx: True,
        lines=_help_hint,
    ),
]
