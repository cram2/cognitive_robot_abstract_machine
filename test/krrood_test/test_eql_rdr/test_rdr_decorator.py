"""
Phase 5 tests: ``RDRWrapper`` and the ``rdr()`` factory in ``rdr/decorator.py``.

These tests gate the Phase 5 implementation.  Each test verifies exactly one
observable guarantee; the groups below map one-to-one to the specification
sections in the implementation plan.

Step A — Basic wrapping (no disk I/O, no rules):
  * @rdr("name.py") on a fully-annotated function returns an ``RDRWrapper``.
  * Calling the wrapper with no rules returns the original output.
  * ``wrapper.__name__`` equals the original function's ``__name__``.
  * ``wrapper.__doc__`` equals the original function's ``__doc__``.
  * ``wrapper.rdr`` is an ``EQLSingleClassRDR`` instance.
  * ``wrapper.case_type`` is a subclass of ``FunctionCase``.
  * ``wrapper.case_type.function`` is bound to the original callable.

Step B — Inference mode (fit=False, the default):
  * After ``fit_case`` with a scripted expert a matching case returns the
    expert's conclusion, not the original output.
  * A non-matching case (no rule fires) returns the original function output.

Step C — Fit mode (fit=True):
  * In fit mode ``__call__`` ALWAYS returns the original output.
  * In fit mode ``__call__`` invokes ``rdr.fit_case`` on every call (spy).

Step D — Missing annotations (error cases):
  * Decorating a function without a return annotation raises
    ``FunctionMissingAnnotationsError`` at decoration time.
  * Decorating a function with an unannotated parameter raises
    ``FunctionMissingAnnotationsError`` at decoration time.

Step E — Dual-mode factory semantics:
  * ``@rdr("name.py")`` (bare, no keywords) produces an ``RDRWrapper``.
  * ``@rdr("name.py", expert=e)`` wires ``wrapper.expert == e``.
  * ``@rdr("name.py", fit=True)`` sets ``wrapper.fit_mode`` to ``True``.

Step F — save_path wiring:
  * After construction ``wrapper.rdr.save_path == store.path``.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar
from unittest.mock import patch

import pytest

from krrood.entity_query_language.rdr.expert import Expert
from krrood.entity_query_language.rdr.function_case import FunctionCase
from krrood.entity_query_language.rdr.interface import FunctionInterface
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rdr.utils import UNSET

# ---------------------------------------------------------------------------
# Helpers — NOT collected by pytest (no ``Test`` prefix, no ``test_`` prefix)
# ---------------------------------------------------------------------------


def _scripted_expert_for_distance(conclusion: float) -> Expert:
    """A scripted ``Expert`` that concludes *conclusion* when ``x > 0``.

    Uses the two-call ``FunctionInterface`` pattern: the first ``interact`` call
    is the conclusion question; the second is the conditions question.
    """
    call_count = {"n": 0}

    def answer_fn(ctx, reqs):
        call_count["n"] += 1
        if len(reqs) == 1 and reqs[0].name == "conclusion":
            # Conclusion question — return the scripted conclusion.
            return {"conclusion": conclusion}
        # Conditions question — x > 0.
        return {"conditions": ctx.case_variable.x > 0}

    return Expert(interface=FunctionInterface(answer_fn=answer_fn))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def distance_wrapper(tmp_path):
    """An @rdr-decorated distance function backed by a tmp_path file.

    Using an absolute path avoids ``_rdr_models/`` placement relative to this
    test file and keeps the artefact strictly under tmp_path.
    """
    from krrood.entity_query_language.rdr.decorator import rdr

    filename = str(tmp_path / "distance_rdr.py")

    @rdr(filename)
    def distance(x: float, y: float) -> float:
        """Compute Euclidean distance."""
        return (x**2 + y**2) ** 0.5

    return distance


@pytest.fixture
def fit_mode_wrapper(tmp_path):
    """An @rdr(fit=True)-decorated distance function backed by a tmp_path file."""
    from krrood.entity_query_language.rdr.decorator import rdr

    filename = str(tmp_path / "distance_fit_rdr.py")

    @rdr(filename, fit=True)
    def distance(x: float, y: float) -> float:
        """Compute Euclidean distance (fit mode)."""
        return (x**2 + y**2) ** 0.5

    return distance


@pytest.fixture
def scripted_expert():
    """A scripted expert that concludes ``99.0`` whenever ``x > 0``."""
    return _scripted_expert_for_distance(99.0)


# ---------------------------------------------------------------------------
# Group 1 — Basic wrapping: type, structure, update_wrapper
# ---------------------------------------------------------------------------


class TestBasicWrapping:
    """Step A: @rdr on a fully-annotated function produces a well-formed RDRWrapper."""

    def test_decorator_returns_rdr_wrapper_instance(self, distance_wrapper):
        """@rdr("name.py") must return an ``RDRWrapper`` instance, not the raw function."""
        from krrood.entity_query_language.rdr.decorator import RDRWrapper

        assert isinstance(distance_wrapper, RDRWrapper)

    def test_no_rules_returns_original_output(self, distance_wrapper):
        """Calling the wrapper with no fitted rules must return the original function output."""
        result = distance_wrapper(3.0, 4.0)
        assert result == pytest.approx(5.0)

    def test_wrapper_dunder_name_matches_original(self, distance_wrapper):
        """``wrapper.__name__`` must equal the original function's ``__name__`` (update_wrapper)."""
        assert distance_wrapper.__name__ == "distance"

    def test_wrapper_dunder_doc_matches_original(self, distance_wrapper):
        """``wrapper.__doc__`` must equal the original function's ``__doc__`` (update_wrapper)."""
        assert distance_wrapper.__doc__ == "Compute Euclidean distance."

    def test_wrapper_rdr_attribute_is_eql_single_class_rdr(self, distance_wrapper):
        """``wrapper.rdr`` must be an ``EQLSingleClassRDR`` instance."""
        assert isinstance(distance_wrapper.rdr, EQLSingleClassRDR)

    def test_wrapper_case_type_is_function_case_subclass(self, distance_wrapper):
        """``wrapper.case_type`` must be a strict subclass of ``FunctionCase``."""
        assert issubclass(distance_wrapper.case_type, FunctionCase)
        assert distance_wrapper.case_type is not FunctionCase

    def test_wrapper_case_type_function_classvar_is_original(
        self, distance_wrapper, tmp_path
    ):
        """``wrapper.case_type.function`` must reference the original decorated callable.

        This verifies that ``__post_init__`` correctly assigns
        ``case_type.function = self.func`` after generating or loading the file.
        """
        # The underlying function is the closure; we verify by calling it.
        func = distance_wrapper.case_type.function
        assert callable(func)
        assert func(3.0, 4.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Group 2 — Inference mode (fit=False, the default)
# ---------------------------------------------------------------------------


class TestInferenceMode:
    """Step B: With fit=False (default), classify returns the expert's conclusion when a rule fires."""

    def test_matching_case_returns_expert_conclusion_not_original_output(
        self, distance_wrapper, scripted_expert
    ):
        """After fitting a rule, a case that triggers it must return the expert's conclusion.

        The expert concludes ``99.0`` when ``x > 0``.  The original function
        would return ``5.0`` for ``(3.0, 4.0)``.  The wrapper in inference mode
        must return ``99.0``.
        """
        distance_wrapper.fit_case(
            distance_wrapper.case_type(x=1.0, y=0.0, _output=None),
            target=UNSET,
            expert=scripted_expert,
        )
        result = distance_wrapper(3.0, 4.0)
        assert result == pytest.approx(99.0)

    def test_non_matching_case_returns_original_output(
        self, distance_wrapper, scripted_expert
    ):
        """When no rule fires (x <= 0), the wrapper must return the original function output.

        The rule fires only for ``x > 0``.  Calling with ``x=-1.0`` bypasses
        the rule and falls through to the raw function return value.
        """
        distance_wrapper.fit_case(
            distance_wrapper.case_type(x=1.0, y=0.0, _output=None),
            target=UNSET,
            expert=scripted_expert,
        )
        result = distance_wrapper(-1.0, 0.0)
        # Original function: sqrt((-1)**2 + 0**2) = 1.0
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Group 3 — Fit mode (fit=True)
# ---------------------------------------------------------------------------


class TestFitMode:
    """Step C: In fit mode __call__ always returns the original output and spies on fit_case."""

    def test_fit_mode_always_returns_original_output(
        self, fit_mode_wrapper, scripted_expert
    ):
        """In fit mode, __call__ must return the original function output even after fitting.

        Even if internal rules would classify differently, the caller always
        gets the raw return value so training data is produced faithfully.
        """
        # Add a rule so there IS something that could override the output.
        fit_mode_wrapper.rdr.fit_case(
            fit_mode_wrapper.case_type(x=1.0, y=0.0, _output=None),
            target=UNSET,
            expert=scripted_expert,
        )
        # Now call the wrapper in fit mode; it must still return the raw value.
        result = fit_mode_wrapper(3.0, 4.0)
        assert result == pytest.approx(5.0)

    def test_fit_mode_call_invokes_rdr_fit_case(
        self, fit_mode_wrapper, scripted_expert
    ):
        """In fit mode, calling the wrapper must invoke ``rdr.fit_case`` on every call.

        We spy on ``fit_mode_wrapper.rdr.fit_case`` to count invocations; a
        single ``__call__`` must trigger exactly one ``fit_case`` call.
        """
        with patch.object(
            fit_mode_wrapper.rdr, "fit_case", wraps=fit_mode_wrapper.rdr.fit_case
        ) as spy:
            # Provide an expert on the wrapper so fit_case gets one.
            fit_mode_wrapper.expert = scripted_expert
            fit_mode_wrapper(1.0, 0.0)
            spy.assert_called_once()


# ---------------------------------------------------------------------------
# Group 4 — Missing annotations (error cases)
# ---------------------------------------------------------------------------


class TestMissingAnnotationsErrors:
    """Step D: @rdr must raise FunctionMissingAnnotationsError at decoration time."""

    def test_no_return_annotation_raises_at_decoration_time(self, tmp_path):
        """Applying @rdr to a function without a return annotation must raise immediately.

        The error must be raised when the decorator is applied, not when the
        wrapper is later called.
        """
        from krrood.code_generation import (
            FunctionMissingAnnotationsError,
        )
        from krrood.entity_query_language.rdr.decorator import rdr

        filename = str(tmp_path / "bad_rdr.py")

        with pytest.raises(FunctionMissingAnnotationsError):

            @rdr(filename)
            def no_return(x: float, y: float):  # no -> annotation
                return x + y

    def test_unannotated_parameter_raises_at_decoration_time(self, tmp_path):
        """Applying @rdr to a function with an unannotated parameter must raise immediately.

        The error must be raised when the decorator is applied, not deferred
        until the wrapper is called.
        """
        from krrood.code_generation import (
            FunctionMissingAnnotationsError,
        )
        from krrood.entity_query_language.rdr.decorator import rdr

        filename = str(tmp_path / "bad_param_rdr.py")

        with pytest.raises(FunctionMissingAnnotationsError):

            @rdr(filename)
            def unannotated_param(x, y: float) -> float:  # x has no annotation
                return x + y


# ---------------------------------------------------------------------------
# Group 5 — Dual-mode factory semantics
# ---------------------------------------------------------------------------


class TestFactorySemantics:
    """Step E: rdr() behaves correctly as a bare factory and with keyword arguments."""

    def test_bare_filename_produces_rdr_wrapper(self, tmp_path):
        """@rdr("name.py") with no other arguments must return an RDRWrapper directly.

        This is the simplest usage form; the filename is the sole positional arg.
        """
        from krrood.entity_query_language.rdr.decorator import RDRWrapper, rdr

        filename = str(tmp_path / "bare_rdr.py")

        @rdr(filename)
        def add(a: int, b: int) -> int:
            return a + b

        assert isinstance(add, RDRWrapper)

    def test_expert_kwarg_wires_expert_onto_wrapper(self, tmp_path, scripted_expert):
        """@rdr("name.py", expert=e) must set ``wrapper.expert`` to the supplied expert."""
        from krrood.entity_query_language.rdr.decorator import rdr

        filename = str(tmp_path / "expert_rdr.py")

        @rdr(filename, expert=scripted_expert)
        def add(a: int, b: int) -> int:
            return a + b

        assert add.expert is scripted_expert

    def test_fit_true_kwarg_sets_fit_mode(self, tmp_path):
        """@rdr("name.py", fit=True) must produce a wrapper whose ``fit_mode`` is ``True``."""
        from krrood.entity_query_language.rdr.decorator import rdr

        filename = str(tmp_path / "fit_mode_rdr.py")

        @rdr(filename, fit=True)
        def add(a: int, b: int) -> int:
            return a + b

        assert add.fit_mode is True

    def test_default_fit_mode_is_false(self, distance_wrapper):
        """When ``fit`` is omitted, ``fit_mode`` must default to ``False``."""
        assert distance_wrapper.fit_mode is False


# ---------------------------------------------------------------------------
# Group 6 — save_path wiring
# ---------------------------------------------------------------------------


class TestSavePathWiring:
    """Step F: After construction wrapper.rdr.save_path must equal store.path."""

    def test_rdr_save_path_equals_store_path(self, distance_wrapper, tmp_path):
        """``wrapper.rdr.save_path`` must be set to ``store.path`` in ``__post_init__``.

        This ensures that every rule insertion automatically persists to the
        same file that the decorator manages.
        """
        expected_path = str(tmp_path / "distance_rdr.py")
        assert distance_wrapper.rdr.save_path == expected_path

    def test_rdr_save_path_is_a_string(self, distance_wrapper):
        """``wrapper.rdr.save_path`` must be a ``str``, not a ``Path`` object."""
        assert isinstance(distance_wrapper.rdr.save_path, str)


# ---------------------------------------------------------------------------
# Group 7 — Corner-case provenance round-trip through RDRWrapper
# ---------------------------------------------------------------------------


class TestRdrDecoratorCornerCaseRoundTrip:
    """Step G: Corner cases survive the @rdr auto-save + reload cycle end-to-end.

    This is a guard test: it proves that the wiring between ``RDRWrapper``,
    ``EQLSingleClassRDR.corner_cases``, ``save_rdr_with_case``, and ``load_rdr``
    holds for the decorator path.  No new implementation is expected — Phases 1–4
    provide all the machinery; this test verifies the wiring stays intact.
    """

    def test_rdr_decorator_corner_cases_survive_save_reload(self, tmp_path):
        """Corner cases recorded during fitting must be present after reloading the saved file.

        The ``@rdr`` decorator auto-saves after every ``fit_case`` call (because
        ``save_path`` is wired in ``__post_init__``).  After fitting two rules on
        two distinct cases, the saved file's ``RDR_CORNER_CASES`` dict must be
        non-empty.  Reloading via a fresh ``RDRWrapper`` must reconstruct a
        ``corner_cases`` store whose ``cases`` dict has at least as many entries
        as rules were inserted (one entry per new rule).
        """
        from krrood.entity_query_language.rdr.decorator import RDRWrapper, rdr
        from krrood.entity_query_language.rdr.serialization import (
            load_rdr,
            walk_rules_in_emission_order,
        )

        filename = str(tmp_path / "corner_case_round_trip.py")

        # ------------------------------------------------------------------ #
        # Step 1: build the decorated function backed by the tmp_path file.
        # ------------------------------------------------------------------ #

        @rdr(filename)
        def distance(x: float, y: float) -> float:
            """Compute Euclidean distance."""
            return (x**2 + y**2) ** 0.5

        # ------------------------------------------------------------------ #
        # Step 2: build two scripted experts, each producing a distinct rule.
        #
        # Expert A: concludes 10.0 when x > 0.
        # Expert B: concludes 20.0 when x < 0.
        #
        # The two-call FunctionInterface pattern is used: the first ``interact``
        # call answers the conclusion question; the second answers the conditions
        # question.  Each call to ``fit_case`` inserts exactly one new rule and
        # records one corner case.
        # ------------------------------------------------------------------ #

        def _make_scripted_expert(conclusion: float, condition_fn):
            """Return a scripted Expert that answers conclusion then conditions."""

            call_count = {"count": 0}

            def answer_fn(ctx, reqs):
                call_count["count"] += 1
                if len(reqs) == 1 and reqs[0].name == "conclusion":
                    return {"conclusion": conclusion}
                return {"conditions": condition_fn(ctx)}

            return Expert(interface=FunctionInterface(answer_fn=answer_fn))

        expert_a = _make_scripted_expert(10.0, lambda ctx: ctx.case_variable.x > 0)
        expert_b = _make_scripted_expert(20.0, lambda ctx: ctx.case_variable.x < 0)

        # Case A: x=1.0, y=0.0 — no rule fires yet, so an alternative is added.
        case_a = distance.case_type(x=1.0, y=0.0, _output=1.0)
        distance.fit_case(case_a, target=UNSET, expert=expert_a)

        # Case B: x=-1.0, y=0.0 — the A-rule does not fire (x < 0), so another
        # alternative is added.
        case_b = distance.case_type(x=-1.0, y=0.0, _output=1.0)
        distance.fit_case(case_b, target=UNSET, expert=expert_b)

        # ------------------------------------------------------------------ #
        # Step 3: verify the in-memory store recorded exactly 2 corner cases.
        # ------------------------------------------------------------------ #
        original_count = len(distance.rdr.corner_cases.cases)
        assert (
            original_count == 2
        ), f"Expected 2 corner cases after 2 rule insertions, got {original_count}"

        # ------------------------------------------------------------------ #
        # Step 4: reload from the same file via a fresh RDRWrapper.
        #
        # ``_load_or_generate`` will call ``load_rdr`` (not generate a new file)
        # because the file already exists.  The ``RDR_CORNER_CASES`` dict written
        # by ``rdr_to_python`` must be non-empty and must map back to 2 entries.
        # ------------------------------------------------------------------ #

        @rdr(filename)
        def distance_reloaded(x: float, y: float) -> float:
            """Compute Euclidean distance."""
            return (x**2 + y**2) ** 0.5

        reloaded_count = len(distance_reloaded.rdr.corner_cases.cases)
        assert reloaded_count == original_count, (
            f"Corner cases lost after reload: expected {original_count}, "
            f"got {reloaded_count}.  "
            "Check that RDR_CORNER_CASES is written to the file and that "
            "load_rdr rebuilds CornerCaseStore.from_ordered_cases correctly."
        )
