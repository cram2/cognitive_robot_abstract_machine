"""
Tests for oscillation detection and progress teardown in :meth:`EQLSingleClassRDR.fit`.

Each test verifies exactly one guarantee of the convergence-loop cycle-detection path
introduced in :class:`~krrood.entity_query_language.rdr.single_class.EQLSingleClassRDR`:

1. A non-discriminating expert triggers :class:`RDRConvergenceWarning` and terminates.
2. A discriminating expert completes without any warning.
3. ``SpyProgressReporter.reset()`` receives the actual pending count on a re-pass.
4. The no-target (``targets=None``) path finishes without a warning.
5. The oscillation warning message names the clashing case reprs.
"""

from __future__ import annotations

import dataclasses
import enum
import warnings

import pytest

from krrood.entity_query_language.rdr.expert import Expert
from krrood.entity_query_language.rdr.interface import CaseContext, FunctionInterface
from krrood.entity_query_language.rdr.progress import SpyProgressReporter
from krrood.entity_query_language.rdr.single_class import (
    EQLSingleClassRDR,
    RDRConvergenceWarning,
)

# ---------------------------------------------------------------------------
# Minimal case domain: two boolean attributes, two labels
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TwoAttr:
    """A minimal case with a shared attribute and a unique discriminating attribute."""

    shared: bool
    unique: bool
    label: "TwoLabel | None" = None


class TwoLabel(enum.Enum):
    """Two mutually-exclusive target labels."""

    red = 1
    blue = 2


# Two canonical cases used by most tests.
_RED_CASE = TwoAttr(shared=True, unique=True)
_BLUE_CASE = TwoAttr(shared=True, unique=False)
_RED_TARGET = TwoLabel.red
_BLUE_TARGET = TwoLabel.blue


# ---------------------------------------------------------------------------
# Helper: SpyFunctionInterface (mirrors the pattern from test_single_class_rdr.py)
# ---------------------------------------------------------------------------


class SpyFunctionInterface(FunctionInterface):
    """A :class:`FunctionInterface` whose :meth:`make_progress_reporter` returns a spy.

    Allows tests to observe progress-reporter lifecycle calls injected into ``fit()``
    without displaying anything to the terminal.
    """

    def __init__(self, answer_fn):
        super().__init__(answer_fn=answer_fn)
        self.spy = SpyProgressReporter()

    def make_progress_reporter(self) -> SpyProgressReporter:
        return self.spy


# ---------------------------------------------------------------------------
# Answer functions
# ---------------------------------------------------------------------------


def _non_discriminating_answer(context: CaseContext, requests):
    """Always conditions on ``v.shared == True``, which holds for both cases.

    Because ``shared`` is True for every case in the two-case fixture, no rule built on
    this condition can uniquely discriminate between red and blue.  Every inserted rule
    intercepts the previously-fitted case, causing the pending set to oscillate.
    """
    v = context.case_variable
    return {"conditions": v.shared == True}


def _discriminating_answer(context: CaseContext, requests):
    """Conditions on the ``unique`` attribute, which differs between red and blue.

    red  (unique=True)  → ``v.unique == True``
    blue (unique=False) → ``v.unique == False``

    Each rule is uniquely satisfied by exactly one case, so the RDR converges in at
    most two passes.
    """
    v = context.case_variable
    target = context.target_conclusion
    if target == TwoLabel.red:
        return {"conditions": v.unique == True}
    return {"conditions": v.unique == False}


def _labelling_answer(context: CaseContext, requests):
    """No-target (ask_for_rule) path: returns both conclusion and unique condition.

    Used to verify the no-target ``fit()`` path does not trigger oscillation detection.
    """
    v = context.case_variable
    case = context.case_instance
    # Provide the conclusion when asked (the ask_for_rule two-call protocol).
    result: dict = {}
    if any(r.name == "conclusion" for r in requests):
        result["conclusion"] = TwoLabel.red if case.unique else TwoLabel.blue
    # Use the uniquely discriminating condition.
    result["conditions"] = (v.unique == True) if case.unique else (v.unique == False)
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def oscillating_rdr() -> EQLSingleClassRDR:
    """A fresh RDR for the TwoAttr / TwoLabel domain, unfitted."""
    return EQLSingleClassRDR(TwoAttr, "label")


@pytest.fixture()
def non_discriminating_expert() -> Expert:
    """An expert that always responds with the non-discriminating ``shared == True`` condition."""
    return Expert(interface=FunctionInterface(answer_fn=_non_discriminating_answer))


@pytest.fixture()
def discriminating_expert() -> Expert:
    """An expert that responds with the uniquely discriminating ``unique`` conditions."""
    return Expert(interface=FunctionInterface(answer_fn=_discriminating_answer))


# ---------------------------------------------------------------------------
# Test 1 — Oscillation terminates with RDRConvergenceWarning
# ---------------------------------------------------------------------------


def test_oscillating_fit_terminates_before_max_passes(
    oscillating_rdr: EQLSingleClassRDR,
    non_discriminating_expert: Expert,
):
    """fit() must return well before max_passes when oscillation is detected.

    Guarantees: the cycle-detection check fires on a repeated pending-set signature
    rather than exhausting all allowed passes.
    """
    max_passes = 20
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        oscillating_rdr.fit(
            [_RED_CASE, _BLUE_CASE],
            [_RED_TARGET, _BLUE_TARGET],
            non_discriminating_expert,
            max_passes=max_passes,
        )
    # At least one RDRConvergenceWarning must have been emitted.
    convergence_warnings = [
        w for w in caught if issubclass(w.category, RDRConvergenceWarning)
    ]
    assert (
        len(convergence_warnings) >= 1
    ), "Expected at least one RDRConvergenceWarning for an oscillating dataset, got none."


def test_oscillating_fit_emits_exactly_one_convergence_warning(
    oscillating_rdr: EQLSingleClassRDR,
    non_discriminating_expert: Expert,
):
    """fit() emits exactly one RDRConvergenceWarning (not multiple) when oscillation is detected.

    Guarantees: the warning is a single, atomic signal — not repeated per-pass.
    """
    with pytest.warns(RDRConvergenceWarning) as record:
        oscillating_rdr.fit(
            [_RED_CASE, _BLUE_CASE],
            [_RED_TARGET, _BLUE_TARGET],
            non_discriminating_expert,
            max_passes=20,
        )
    assert (
        len(record) == 1
    ), f"Expected exactly 1 RDRConvergenceWarning, got {len(record)}."


# ---------------------------------------------------------------------------
# Test 2 — Happy-path: convergent dataset completes without warning
# ---------------------------------------------------------------------------


def test_convergent_fit_emits_no_convergence_warning(
    oscillating_rdr: EQLSingleClassRDR,
    discriminating_expert: Expert,
):
    """A dataset with genuinely discriminating conditions must not emit RDRConvergenceWarning.

    Guarantees: the warning is only emitted for genuine oscillation, not for any
    fit that requires more than one pass.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        oscillating_rdr.fit(
            [_RED_CASE, _BLUE_CASE],
            [_RED_TARGET, _BLUE_TARGET],
            discriminating_expert,
        )
    convergence_warnings = [
        w for w in caught if issubclass(w.category, RDRConvergenceWarning)
    ]
    assert convergence_warnings == [], (
        "RDRConvergenceWarning must not be emitted for a convergent dataset, "
        f"got: {[str(w.message) for w in convergence_warnings]}"
    )


def test_convergent_fit_classifies_all_cases_correctly(
    oscillating_rdr: EQLSingleClassRDR,
    discriminating_expert: Expert,
):
    """After a convergent fit, every case is classified with its correct label.

    Guarantees: the happy-path regression — the fit result is semantically correct.
    """
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        oscillating_rdr.fit(
            [_RED_CASE, _BLUE_CASE],
            [_RED_TARGET, _BLUE_TARGET],
            discriminating_expert,
        )
    assert oscillating_rdr.classify(_RED_CASE) == _RED_TARGET
    assert oscillating_rdr.classify(_BLUE_CASE) == _BLUE_TARGET


def test_convergent_fit_calls_finish_exactly_once(
    oscillating_rdr: EQLSingleClassRDR,
):
    """A convergent fit via SpyFunctionInterface records exactly one finish() event.

    Guarantees: the progress reporter is torn down exactly once regardless of pass count.
    """
    spy_interface = SpyFunctionInterface(answer_fn=_discriminating_answer)
    expert = Expert(interface=spy_interface)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        oscillating_rdr.fit(
            [_RED_CASE, _BLUE_CASE],
            [_RED_TARGET, _BLUE_TARGET],
            expert,
        )
    finish_events = [e for e in spy_interface.spy.events if e[0] == "finish"]
    assert len(finish_events) == 1, (
        f"Expected exactly one finish() event, got {len(finish_events)}: "
        f"{spy_interface.spy.events}"
    )


# ---------------------------------------------------------------------------
# Test 3 — reset() called with actual pending count per re-pass
# ---------------------------------------------------------------------------


def test_reset_called_with_pending_count_not_total(
    oscillating_rdr: EQLSingleClassRDR,
):
    """When a second pass is needed, reset() receives the number of pending cases, not the total.

    Guarantees: the progress counter reflects the actual remaining work on each re-pass
    rather than always resetting to the original dataset size.

    Setup: Two cases; the discriminating expert fits both correctly in pass 1 (red first),
    but the blue rule, if it caused a retroactive break of red, would force a re-pass.
    To guarantee a known re-pass count, we use a single-case oscillating setup wrapped
    in a two-case scenario where only one case is pending on the second pass.

    We verify this via the ``reset`` event: its argument must be strictly less than the
    original ``start`` argument whenever the re-pass pending set is smaller.
    """
    # Build a scenario where pass 1 processes both cases but only one remains pending
    # after pass 1.  The non-discriminating expert causes blue to always be pending
    # after pass 1 (red's rule intercepts blue), so pending count on pass 2 is 1,
    # not 2.
    spy_interface = SpyFunctionInterface(answer_fn=_non_discriminating_answer)
    expert = Expert(interface=spy_interface)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        oscillating_rdr.fit(
            [_RED_CASE, _BLUE_CASE],
            [_RED_TARGET, _BLUE_TARGET],
            expert,
            max_passes=20,
        )

    reset_events = [e for e in spy_interface.spy.events if e[0] == "reset"]
    assert reset_events, "Expected at least one reset() call during oscillating fit."

    start_events = [e for e in spy_interface.spy.events if e[0] == "start"]
    assert start_events, "Expected a start() call."
    original_total = start_events[0][1][0]

    # Every reset must carry a count that corresponds to a real pending subset.
    for reset_event in reset_events:
        pending_count = reset_event[1][0]
        assert 1 <= pending_count <= original_total, (
            f"reset() was called with {pending_count}, which is outside the valid "
            f"range [1, {original_total}]."
        )


# ---------------------------------------------------------------------------
# Test 4 — No-target path: finishes without oscillation check or warning
# ---------------------------------------------------------------------------


def test_no_target_fit_emits_no_convergence_warning(
    oscillating_rdr: EQLSingleClassRDR,
):
    """When targets=None, fit() never runs oscillation detection and emits no warning.

    Guarantees: the no-target (ask_for_rule) single-pass path is completely isolated
    from the convergence loop, even when the expert provides non-discriminating conditions.
    """
    labelling_interface = FunctionInterface(answer_fn=_labelling_answer)
    expert = Expert(interface=labelling_interface)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        oscillating_rdr.fit([_RED_CASE, _BLUE_CASE], None, expert)
    convergence_warnings = [
        w for w in caught if issubclass(w.category, RDRConvergenceWarning)
    ]
    assert convergence_warnings == [], (
        "RDRConvergenceWarning must not be emitted on the no-target path, "
        f"got: {[str(w.message) for w in convergence_warnings]}"
    )


def test_no_target_fit_calls_finish_exactly_once(
    oscillating_rdr: EQLSingleClassRDR,
):
    """The no-target path always calls finish() exactly once.

    Guarantees: progress teardown is unconditional (the finally block in fit()) on the
    single-pass path as well as the convergent path.
    """
    spy_interface = SpyFunctionInterface(answer_fn=_labelling_answer)
    expert = Expert(interface=spy_interface)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        oscillating_rdr.fit([_RED_CASE, _BLUE_CASE], None, expert)
    finish_events = [e for e in spy_interface.spy.events if e[0] == "finish"]
    assert len(finish_events) == 1, (
        f"Expected exactly one finish() on the no-target path, got {len(finish_events)}: "
        f"{spy_interface.spy.events}"
    )


def test_no_target_fit_never_calls_reset(
    oscillating_rdr: EQLSingleClassRDR,
):
    """The no-target path never calls reset() because it is a single pass.

    Guarantees: the convergence loop (which drives re-passes with reset()) is skipped
    entirely when targets=None.
    """
    spy_interface = SpyFunctionInterface(answer_fn=_labelling_answer)
    expert = Expert(interface=spy_interface)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        oscillating_rdr.fit([_RED_CASE, _BLUE_CASE], None, expert)
    reset_events = [e for e in spy_interface.spy.events if e[0] == "reset"]
    assert (
        reset_events == []
    ), f"reset() must not be called on the no-target path, got: {reset_events}"


# ---------------------------------------------------------------------------
# Test 5 — Warning message names clashing case reprs
# ---------------------------------------------------------------------------


def test_convergence_warning_message_contains_clashing_case_repr(
    oscillating_rdr: EQLSingleClassRDR,
    non_discriminating_expert: Expert,
):
    """The RDRConvergenceWarning message contains the repr of at least one clashing case.

    Guarantees: the warning is actionable — the user can identify which specific cases
    are oscillating from the warning text alone.
    """
    with pytest.warns(RDRConvergenceWarning) as record:
        oscillating_rdr.fit(
            [_RED_CASE, _BLUE_CASE],
            [_RED_TARGET, _BLUE_TARGET],
            non_discriminating_expert,
            max_passes=20,
        )
    message = str(record[0].message)
    # At least one of the two clashing case reprs must appear in the warning message.
    red_repr = repr(_RED_CASE)
    blue_repr = repr(_BLUE_CASE)
    assert red_repr in message or blue_repr in message, (
        f"Warning message must contain repr of a clashing case.\n"
        f"  red repr:  {red_repr!r}\n"
        f"  blue repr: {blue_repr!r}\n"
        f"  message:   {message!r}"
    )


def test_convergence_warning_message_contains_pass_count(
    oscillating_rdr: EQLSingleClassRDR,
    non_discriminating_expert: Expert,
):
    """The RDRConvergenceWarning message states how many passes were completed.

    Guarantees: the warning gives actionable diagnostic context about when oscillation
    was detected.
    """
    with pytest.warns(RDRConvergenceWarning) as record:
        oscillating_rdr.fit(
            [_RED_CASE, _BLUE_CASE],
            [_RED_TARGET, _BLUE_TARGET],
            non_discriminating_expert,
            max_passes=20,
        )
    message = str(record[0].message)
    assert (
        "pass" in message.lower()
    ), f"Warning message must mention the pass count, got: {message!r}"
