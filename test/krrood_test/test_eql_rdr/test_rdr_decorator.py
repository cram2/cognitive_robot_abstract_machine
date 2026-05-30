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
