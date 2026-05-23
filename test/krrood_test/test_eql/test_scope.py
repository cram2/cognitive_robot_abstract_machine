"""
Tests for definition-scope capture (Phase 0 of the EQL-RDR refactor).

Covers:
  - StackFrame scope snapshot (opt-in via capture_scope) and the `scope` property
  - capture_caller_scope: finds the user frame, skips EQL internals, snapshots ns
  - eql_factory_namespace: exposes the EQL verbs
  - attach_definition_scope / get_definition_scope: round-trip + factory overlay
  - underspecified(): attaches the caller scope where it was invoked
"""

import inspect

from krrood.entity_query_language._stack import StackFrame
from krrood.entity_query_language.factories import underspecified
from krrood.entity_query_language.scope import (
    attach_definition_scope,
    capture_caller_scope,
    eql_factory_namespace,
    get_definition_scope,
)

MODULE_LEVEL_SENTINEL = "module_global_value"


# ---------------------------------------------------------------------------
# StackFrame scope snapshot
# ---------------------------------------------------------------------------


def test_stack_frame_scope_not_captured_by_default():
    fi = inspect.stack()[0]
    frame = StackFrame.from_frame_info(fi)
    assert frame.global_ns is None
    assert frame.local_ns is None
    assert frame.scope == {}


def test_stack_frame_scope_captured_when_requested():
    local_token = "local_value"  # noqa: F841 — referenced via captured scope
    fi = inspect.stack()[0]
    frame = StackFrame.from_frame_info(fi, capture_scope=True)

    assert frame.local_ns is not None and frame.global_ns is not None
    assert frame.local_ns["local_token"] == "local_value"
    assert frame.global_ns["MODULE_LEVEL_SENTINEL"] == "module_global_value"
    # locals win over globals in the merged view
    assert frame.scope["local_token"] == "local_value"
    assert frame.scope["MODULE_LEVEL_SENTINEL"] == "module_global_value"


def test_stack_frame_scope_is_a_copy_not_live_reference():
    fi = inspect.stack()[0]
    frame = StackFrame.from_frame_info(fi, capture_scope=True)
    # Mutating the snapshot must not touch the real globals.
    frame.global_ns["MODULE_LEVEL_SENTINEL"] = "mutated"
    assert MODULE_LEVEL_SENTINEL == "module_global_value"


def test_stack_frame_positional_construction_still_works():
    # Pre-existing call sites construct StackFrame positionally with 7 args.
    frame = StackFrame("f.py", 1, "fn", None, None, None, "mod")
    assert frame.global_ns is None and frame.local_ns is None


# ---------------------------------------------------------------------------
# capture_caller_scope
# ---------------------------------------------------------------------------


def test_capture_caller_scope_sees_caller_locals_and_globals():
    a_local = 123  # noqa: F841
    scope = capture_caller_scope()
    assert scope["a_local"] == 123
    assert scope["MODULE_LEVEL_SENTINEL"] == "module_global_value"


def test_capture_caller_scope_skips_eql_internal_frames():
    # underspecified() lives in the EQL package; the scope it attaches must be the
    # caller's (this test), proving EQL-internal frames are skipped.
    marker_in_test = object()  # noqa: F841
    q = underspecified(int)
    scope = getattr(q, "_definition_scope_")
    assert "marker_in_test" in scope
    assert scope["MODULE_LEVEL_SENTINEL"] == "module_global_value"
    # No EQL-internal local leaked in (factory locals like `result`).
    assert "result" not in scope


# ---------------------------------------------------------------------------
# eql_factory_namespace
# ---------------------------------------------------------------------------


def test_eql_factory_namespace_has_core_verbs():
    ns = eql_factory_namespace()
    for name in (
        "entity",
        "an",
        "variable",
        "match",
        "underspecified",
        "and_",
        "or_",
        "not_",
        "contains",
        "add",
        "refinement",
        "alternative",
        "next_rule",
        "Add",
        "Refinement",
        "Alternative",
        "Next",
    ):
        assert name in ns, f"missing factory {name!r}"
    assert callable(ns["entity"])


# ---------------------------------------------------------------------------
# attach / get_definition_scope
# ---------------------------------------------------------------------------


def test_attach_and_get_definition_scope_roundtrip():
    captured = {"x": 1, "y": 2}

    class Holder:
        pass

    obj = Holder()
    attach_definition_scope(obj, captured)
    scope = get_definition_scope(obj)
    assert scope["x"] == 1 and scope["y"] == 2
    # Factory overlay present.
    assert "entity" in scope and "refinement" in scope


def test_get_definition_scope_factory_overlay_wins():
    class Holder:
        pass

    obj = Holder()
    # User shadows a factory name; the overlay must win so the shell has the verb.
    attach_definition_scope(obj, {"entity": "shadowed"})
    scope = get_definition_scope(obj)
    assert callable(scope["entity"])


def test_get_definition_scope_without_factories():
    class Holder:
        pass

    obj = Holder()
    attach_definition_scope(obj, {"x": 1})
    scope = get_definition_scope(obj, include_factories=False)
    assert scope == {"x": 1}


def test_get_definition_scope_falls_back_to_live_caller():
    fallback_local = "present"  # noqa: F841
    scope = get_definition_scope(None)
    assert scope["fallback_local"] == "present"
    assert "entity" in scope


def test_underspecified_attaches_scope():
    sentinel_local = "abc"  # noqa: F841
    q = underspecified(int)
    scope = get_definition_scope(q)
    assert scope["sentinel_local"] == "abc"
    assert "entity" in scope
