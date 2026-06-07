"""
Tests for ``ResolutionMode`` (enum surface), SILENT-mode regression, and HINT-mode
expert-call / suggestion-flow in :class:`~krrood.entity_query_language.rdr.single_class.EQLSingleClassRDR`.

Four test classes:

  TestResolutionModeEnum     — enum surface (members, count, distinctness)
  TestSilentModeDefault      — freshly constructed RDR is SILENT; resolver suppresses expert call
  TestHintMode               — HINT calls expert with suggestion in default+context, accept/overwrite
  TestPromptSectionHint      — prompt section applicable iff suggested_condition is set
"""

from __future__ import annotations

import dataclasses
from typing_extensions import Any, Dict, List, Optional

import pytest

from .animal import Animal, Species, make_animal as _make_animal
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.rdr.condition_resolver import (
    ChainConditionResolver,
    ResolutionMode,
)
from krrood.entity_query_language.rdr.expert import (
    ANSWER_NAME,
    Expert,
    _validate_conditions,
)
from krrood.entity_query_language.rdr.interface import (
    AnswerRequest,
    CaseContext,
    FunctionInterface,
)
from krrood.entity_query_language.rdr.interactive import Palette
from krrood.entity_query_language.rdr.prompt_sections import (
    PROMPT_SECTIONS,
    RenderContext,
)
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rdr.utils import UNSET

# ---------------------------------------------------------------------------
# Three-rule RDR scenario
#
# Rule 1 (root):          milk == True      -> mammal
# Rule 2 (alternative):   venomous == True  -> reptile
# Rule 3 (alternative):   feathers == True  -> bird
#
# bird2 has feathers=True AND venomous=True, so rule 2 fires first and
# misclassifies it as reptile.  The backward-inference resolver finds
# ``feathers==True`` from Species.bird knowledge — a discriminating condition
# because bird2 has feathers=True but the reptile corner case has feathers=False.
# ---------------------------------------------------------------------------


def _three_rule_answer_fn(context, requests):
    """Answer function that maps each target to a simple discriminating condition.

    * mammal  -> milk == True
    * reptile -> venomous == True
    * bird    -> feathers == True
    """
    v = context.case_variable
    target = context.target_conclusion
    if target is Species.mammal:
        return {"conditions": v.milk == True}
    if target is Species.reptile:
        return {"conditions": v.venomous == True}
    if target is Species.bird:
        return {"conditions": v.feathers == True}
    raise ValueError(f"Unexpected target: {target!r}")


def _three_rule_rdr(*, resolution_mode: ResolutionMode = ResolutionMode.SILENT):
    """Build a three-rule RDR (mammal / reptile / bird1) and return supporting objects.

    Returns ``(rdr, bird2, reptile_case, iface, expert)`` where:
    - ``rdr`` has ``condition_resolver`` set to the default backward-inference chain
      and ``resolution_mode`` set to the provided value.
    - ``bird2`` is a new bird that has venomous=True and is initially misclassified
      as reptile (the refinement path is triggered on fit_case(bird2, bird, expert)).
    - ``reptile_case`` is the corner case for the reptile rule.
    - ``iface`` is the CountingFunctionInterface used for the fourth fit call.
    - ``expert`` wraps ``iface``.
    """
    mammal = _make_animal("mammal", milk=True, hair=True)
    reptile = _make_animal("reptile", venomous=True, eggs=True, toothed=True)
    bird1 = _make_animal("bird1", feathers=True, eggs=True, airborne=True, legs=2)
    bird2 = _make_animal("bird2", feathers=True, venomous=True, eggs=True, legs=2)

    rdr = EQLSingleClassRDR(
        Animal,
        "species",
        condition_resolver=ChainConditionResolver.backward_inference_default(),
        resolution_mode=resolution_mode,
    )

    # Fit the first three cases with a plain FunctionInterface (not counting).
    setup_expert = Expert(interface=FunctionInterface(answer_fn=_three_rule_answer_fn))
    rdr.fit_case(mammal, Species.mammal, setup_expert)
    rdr.fit_case(reptile, Species.reptile, setup_expert)
    rdr.fit_case(bird1, Species.bird, setup_expert)

    # Verify the pre-condition: bird2 is misclassified before we fit it.
    assert (
        rdr.classify(bird2) is Species.reptile
    ), "Pre-condition: bird2 must be misclassified as reptile before the hint test."

    iface = CountingFunctionInterface(answer_fn=_three_rule_answer_fn)
    expert = Expert(interface=iface)
    return rdr, bird2, reptile, iface, expert


# ---------------------------------------------------------------------------
# CountingFunctionInterface spy — defined locally (not imported from peer tests)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CountingFunctionInterface(FunctionInterface):
    """A :class:`FunctionInterface` that counts every ``interact`` call.

    :attr:`interact_count` increments on every invocation so a test can assert
    without any mocking framework whether the expert was (or was not) consulted.
    """

    interact_count: int = dataclasses.field(default=0, init=False)

    def interact(
        self, context: CaseContext, requests: List[AnswerRequest]
    ) -> Dict[str, Any]:
        """Forward to the parent and record the call."""
        self.interact_count += 1
        return super().interact(context, requests)


# ---------------------------------------------------------------------------
# Prompt-section lookup helper
# ---------------------------------------------------------------------------

_SECTION_NAMES = {s.name: s for s in PROMPT_SECTIONS}


def _section(name: str):
    """Return the PromptSection with ``name``, raising KeyError with a clear message."""
    if name not in _SECTION_NAMES:
        raise KeyError(f"Section '{name}' not found. Available: {list(_SECTION_NAMES)}")
    return _SECTION_NAMES[name]


# ---------------------------------------------------------------------------
# TestResolutionModeEnum
# ---------------------------------------------------------------------------


class TestResolutionModeEnum:
    """ResolutionMode exposes exactly the two expected members."""

    def test_silent_member_exists(self):
        """ResolutionMode.SILENT must exist as a valid enum member.

        Guarantee: the attribute resolves without AttributeError.
        """
        member = ResolutionMode.SILENT
        assert isinstance(member, ResolutionMode)

    def test_hint_member_exists(self):
        """ResolutionMode.HINT must exist as a valid enum member.

        Guarantee: the attribute resolves without AttributeError.
        """
        member = ResolutionMode.HINT
        assert isinstance(member, ResolutionMode)

    def test_two_members_total(self):
        """Exactly two members exist in ResolutionMode.

        Guarantee: no accidental extras; future additions require an explicit update.
        """
        assert len(list(ResolutionMode)) == 2

    def test_members_are_distinct(self):
        """SILENT and HINT are not equal to each other.

        Guarantee: mode-switching code can rely on identity / equality comparisons.
        """
        assert ResolutionMode.SILENT != ResolutionMode.HINT


# ---------------------------------------------------------------------------
# TestSilentModeDefault
# ---------------------------------------------------------------------------


class TestSilentModeDefault:
    """A freshly constructed EQLSingleClassRDR defaults to SILENT resolution mode."""

    def test_default_mode_is_silent(self):
        """EQLSingleClassRDR() with no arguments has resolution_mode == SILENT.

        Guarantee: existing callers that do not set resolution_mode are unaffected
        by the introduction of HINT mode.
        """
        rdr = EQLSingleClassRDR(Animal, "species")
        assert rdr.resolution_mode is ResolutionMode.SILENT

    def test_silent_mode_expert_not_called_when_resolver_succeeds(self):
        """In SILENT mode the expert is not consulted when the resolver finds a condition.

        Scenario: three-rule RDR (mammal, reptile, bird1). bird2 has feathers=True and
        venomous=True, so it is initially misclassified as reptile. The backward-inference
        resolver finds ``feathers==True`` as a discriminating condition for Species.bird.
        In SILENT mode the condition is inserted directly without calling the expert.

        Guarantee: CountingFunctionInterface.interact_count == 0 for the auto-resolved step.
        """
        rdr, bird2, _reptile, iface, expert = _three_rule_rdr(
            resolution_mode=ResolutionMode.SILENT
        )
        count_before = iface.interact_count
        rdr.fit_case(bird2, Species.bird, expert)
        calls_made = iface.interact_count - count_before

        assert calls_made == 0, (
            f"Expert must not be called in SILENT mode when resolver succeeds, "
            f"but interact() was called {calls_made} time(s)."
        )


# ---------------------------------------------------------------------------
# TestHintMode
# ---------------------------------------------------------------------------


class TestHintMode:
    """HINT mode passes the resolver suggestion to the expert."""

    def test_hint_mode_expert_called_when_resolver_succeeds(self):
        """In HINT mode the expert IS called even when the resolver finds a condition.

        Scenario: same three-rule RDR as the SILENT regression test. The resolver
        finds ``feathers==True`` for bird2. In HINT mode the condition is passed as a
        suggestion to the expert, so interact() must be called exactly once.

        Guarantee: CountingFunctionInterface.interact_count increments by 1 for the step.
        """
        rdr, bird2, _reptile, iface, expert = _three_rule_rdr(
            resolution_mode=ResolutionMode.HINT
        )
        count_before = iface.interact_count
        rdr.fit_case(bird2, Species.bird, expert)
        calls_made = iface.interact_count - count_before

        assert calls_made == 1, (
            f"Expert must be called exactly once in HINT mode when resolver succeeds, "
            f"but interact() was called {calls_made} time(s)."
        )

    def test_hint_mode_request_default_equals_suggestion(self):
        """In HINT mode, requests[0].default is the SymbolicExpression resolved by the resolver.

        Guarantee: the AnswerRequest that the expert receives has its ``default`` field
        pre-seeded with the resolved condition (a SymbolicExpression), so the expert can
        accept it by returning an empty dict or overwrite it.
        """
        captured_default: List[Any] = []

        def _capture(context, requests):
            captured_default.append(requests[0].default)
            return {"conditions": requests[0].default}

        rdr, bird2, _reptile, iface, expert = _three_rule_rdr(
            resolution_mode=ResolutionMode.HINT
        )
        iface.answer_fn = _capture
        rdr.fit_case(bird2, Species.bird, expert)

        assert len(captured_default) == 1, "answer_fn must be called exactly once"
        default = captured_default[0]
        assert isinstance(default, SymbolicExpression), (
            f"requests[0].default must be a SymbolicExpression (the resolver's suggestion), "
            f"got {type(default).__name__!r}."
        )

    def test_hint_mode_context_suggested_condition_is_set(self):
        """In HINT mode, context.suggested_condition is the SymbolicExpression suggestion.

        Guarantee: CaseContext.suggested_condition is populated before the expert is
        called so the expert shell can display the hint to the user.
        """
        captured_suggestion: List[Any] = []

        def _capture(context, requests):
            captured_suggestion.append(context.suggested_condition)
            return {"conditions": context.suggested_condition}

        rdr, bird2, _reptile, iface, expert = _three_rule_rdr(
            resolution_mode=ResolutionMode.HINT
        )
        iface.answer_fn = _capture
        rdr.fit_case(bird2, Species.bird, expert)

        assert len(captured_suggestion) == 1
        suggestion = captured_suggestion[0]
        assert isinstance(suggestion, SymbolicExpression), (
            f"context.suggested_condition must be a SymbolicExpression in HINT mode, "
            f"got {type(suggestion).__name__!r}."
        )

    def test_hint_mode_context_suggested_condition_same_object_as_request_default(self):
        """context.suggested_condition and requests[0].default are the same expression object.

        Guarantee: no copy or transformation is applied between setting the suggestion
        on CaseContext and seeding it as the AnswerRequest default.
        """
        captured: List[tuple] = []

        def _capture(context, requests):
            captured.append((context.suggested_condition, requests[0].default))
            return {"conditions": requests[0].default}

        rdr, bird2, _reptile, iface, expert = _three_rule_rdr(
            resolution_mode=ResolutionMode.HINT
        )
        iface.answer_fn = _capture
        rdr.fit_case(bird2, Species.bird, expert)

        assert len(captured) == 1
        ctx_suggestion, req_default = captured[0]
        assert (
            ctx_suggestion is req_default
        ), "context.suggested_condition and requests[0].default must be the same object."

    def test_hint_accept_uses_suggested_condition(self):
        """When the expert returns {} (no override), the seeded default condition is used.

        The namespace is pre-seeded with the suggestion as ``conditions``; returning {}
        leaves it in place so the validation loop picks it up. bird2 must be correctly
        classified as Species.bird after fitting with the accepted suggestion.

        Guarantee: an expert that does not overwrite the suggestion produces a correct rule.
        """

        def _accept_default(context, requests):
            # Return nothing — the namespace already has the suggestion seeded as default.
            return {}

        rdr, bird2, _reptile, iface, expert = _three_rule_rdr(
            resolution_mode=ResolutionMode.HINT
        )
        iface.answer_fn = _accept_default
        rdr.fit_case(bird2, Species.bird, expert)

        classification = rdr.classify(bird2)
        assert classification is Species.bird, (
            f"bird2 must be classified as Species.bird after accepting the hint, "
            f"got {classification!r}."
        )

    def test_hint_overwrite_uses_expert_condition(self):
        """When the expert returns a different expression, that overwrite is used as the rule.

        The expert returns ``feathers == True`` (same truth value as the resolver's
        suggestion for bird2). This verifies that the expert's returned condition
        replaces the suggestion end-to-end and the case is still classified correctly.

        Guarantee: the expert's returned condition replaces the suggestion end-to-end.
        """

        def _overwrite(context, requests):
            # Return an alternative condition that also holds for bird2 (feathers=True).
            return {"conditions": context.case_variable.feathers == True}

        rdr, bird2, _reptile, iface, expert = _three_rule_rdr(
            resolution_mode=ResolutionMode.HINT
        )
        iface.answer_fn = _overwrite
        rdr.fit_case(bird2, Species.bird, expert)

        classification = rdr.classify(bird2)
        assert classification is Species.bird, (
            f"bird2 must be classified as Species.bird after expert overwrites the hint, "
            f"got {classification!r}."
        )

    def test_hint_mode_no_resolver_still_calls_expert(self):
        """In HINT mode with condition_resolver=None the expert is called once.

        Guarantee: HINT mode never suppresses the expert call; with no resolver the call
        happens unconditionally, and context.suggested_condition is None (no hint).
        """
        captured_suggestion: List[Any] = ["_not_called_"]

        def _answer(context, requests):
            captured_suggestion[0] = context.suggested_condition
            return {"conditions": context.case_variable.feathers == True}

        mammal = _make_animal("mammal", milk=True, hair=True)
        reptile = _make_animal("reptile", venomous=True, eggs=True, toothed=True)
        bird1 = _make_animal("bird1", feathers=True, eggs=True, airborne=True, legs=2)
        bird2 = _make_animal("bird2", feathers=True, venomous=True, eggs=True, legs=2)

        rdr = EQLSingleClassRDR(
            Animal,
            "species",
            condition_resolver=None,
            resolution_mode=ResolutionMode.HINT,
        )

        setup_expert = Expert(
            interface=FunctionInterface(answer_fn=_three_rule_answer_fn)
        )
        rdr.fit_case(mammal, Species.mammal, setup_expert)
        rdr.fit_case(reptile, Species.reptile, setup_expert)
        rdr.fit_case(bird1, Species.bird, setup_expert)

        assert (
            rdr.classify(bird2) is Species.reptile
        ), "Pre-condition: bird2 must be misclassified as reptile."

        iface = CountingFunctionInterface(answer_fn=_answer)
        expert = Expert(interface=iface)
        count_before = iface.interact_count
        rdr.fit_case(bird2, Species.bird, expert)
        calls_made = iface.interact_count - count_before

        assert calls_made == 1, (
            f"Expert must be called once when resolver=None in HINT mode, "
            f"got {calls_made} call(s)."
        )
        assert (
            captured_suggestion[0] is None
        ), "context.suggested_condition must be None when no resolver is set."


# ---------------------------------------------------------------------------
# TestPromptSectionHint
# ---------------------------------------------------------------------------


class TestPromptSectionHint:
    """The ``auto_resolution_hint`` prompt section fires iff suggested_condition is set."""

    def _palette(self) -> Palette:
        return Palette(use_color=False)

    def _minimal_rdr(self) -> EQLSingleClassRDR:
        """Return an empty RDR for building a CaseContext (provides case_variable)."""
        return EQLSingleClassRDR(Animal, "species")

    def _three_rule_suggestion(self):
        """Build a three-rule RDR and return ``(rdr, bird2, reptile, suggestion)``.

        ``suggestion`` is the auto-resolved expression for ``bird2`` (the condition
        the backward-inference resolver would insert silently in SILENT mode).
        """
        mammal = _make_animal("mammal", milk=True, hair=True)
        reptile = _make_animal("reptile", venomous=True, eggs=True, toothed=True)
        bird1 = _make_animal("bird1", feathers=True, eggs=True, airborne=True, legs=2)
        bird2 = _make_animal("bird2", feathers=True, venomous=True, eggs=True, legs=2)

        rdr = EQLSingleClassRDR(
            Animal,
            "species",
            condition_resolver=ChainConditionResolver.backward_inference_default(),
        )
        setup_expert = Expert(
            interface=FunctionInterface(answer_fn=_three_rule_answer_fn)
        )
        rdr.fit_case(mammal, Species.mammal, setup_expert)
        rdr.fit_case(reptile, Species.reptile, setup_expert)
        rdr.fit_case(bird1, Species.bird, setup_expert)

        suggestion = rdr._try_auto_resolve(
            bird2, Species.bird, Species.reptile, reptile
        )
        assert suggestion is not None, "Pre-condition: resolver must find a suggestion"
        return rdr, bird2, reptile, suggestion

    def _render_ctx(
        self,
        suggested_condition: Optional[SymbolicExpression],
        rdr: Optional[EQLSingleClassRDR] = None,
        case: Optional[Animal] = None,
    ) -> RenderContext:
        """Build a minimal RenderContext with the given suggested_condition."""
        if rdr is None:
            rdr = self._minimal_rdr()
        if case is None:
            case = _make_animal("hint_test_bird", feathers=True)
        ctx = CaseContext(
            case_instance=case,
            case_variable=rdr.case_variable,
            current_conclusion=UNSET,
            target_conclusion=Species.bird,
            suggested_condition=suggested_condition,
        )
        req = AnswerRequest(
            name=ANSWER_NAME,
            validate=_validate_conditions,
            example=f"{ANSWER_NAME} = case_variable.some_attr == True",
        )
        return RenderContext(case=ctx, requests=[req], palette=self._palette())

    def test_auto_resolution_hint_section_exists(self):
        """PROMPT_SECTIONS contains a section named 'auto_resolution_hint'.

        Guarantee: the section was registered and can be looked up by name.
        """
        section = _section("auto_resolution_hint")
        assert section.name == "auto_resolution_hint"

    def test_auto_resolution_hint_applicable_when_suggested_condition_set(self):
        """auto_resolution_hint.applicable returns True when CaseContext.suggested_condition is set.

        Guarantee: the section fires whenever a hint is available for the expert.
        """
        rdr, bird2, _reptile, suggestion = self._three_rule_suggestion()
        render_ctx = self._render_ctx(
            suggested_condition=suggestion, rdr=rdr, case=bird2
        )
        section = _section("auto_resolution_hint")
        assert section.applicable(render_ctx) is True

    def test_auto_resolution_hint_not_applicable_when_no_suggestion(self):
        """auto_resolution_hint.applicable returns False when suggested_condition is None.

        Guarantee: the section is suppressed when no auto-resolution hint is available
        (SILENT mode, no resolver, or resolver returned nothing).
        """
        render_ctx = self._render_ctx(suggested_condition=None)
        section = _section("auto_resolution_hint")
        assert section.applicable(render_ctx) is False

    def test_auto_resolution_hint_lines_contain_formatted_condition(self):
        """auto_resolution_hint.lines returns non-empty text that includes the condition.

        Guarantee: when applicable, the section emits at least one line with a
        non-empty string so the expert shell displays the hint.
        """
        rdr, bird2, _reptile, suggestion = self._three_rule_suggestion()
        render_ctx = self._render_ctx(
            suggested_condition=suggestion, rdr=rdr, case=bird2
        )
        section = _section("auto_resolution_hint")
        lines = section.lines(render_ctx)

        assert isinstance(lines, list), "lines() must return a list"
        assert len(lines) >= 1, "lines() must return at least one line"
        joined = "".join(lines)
        assert joined.strip(), "The joined hint text must be non-empty"
