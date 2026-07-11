"""
Tests for definition-scope capture (:mod:`krrood.entity_query_language.scope`).

Covers:
  - StackFrame scope snapshot (opt-in via capture_scope) and the `scope` property
  - capture_caller_scope: finds the user frame, skips EQL internals, snapshots ns
  - eql_factory_namespace: exposes the EQL verbs
  - attach_definition_scope / get_definition_scope: round-trip + factory overlay
"""

import inspect

from krrood.entity_query_language import factories
from krrood.entity_query_language._stack import StackFrame
from krrood.entity_query_language.rules.conclusion import Add
from krrood.entity_query_language.rules.conclusion_selector import (
    Alternative,
    Next,
    Refinement,
)
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


# ---------------------------------------------------------------------------
# eql_factory_namespace
# ---------------------------------------------------------------------------


def test_eql_factory_namespace_has_core_verbs():
    ns = eql_factory_namespace()
    for name in (
        factories.entity.__name__,
        factories.an.__name__,
        factories.the.__name__,
        factories.variable.__name__,
        factories.and_.__name__,
        factories.or_.__name__,
        factories.not_.__name__,
        factories.contains.__name__,
        factories.add.__name__,
        factories.refinement.__name__,
        factories.alternative.__name__,
        factories.next_rule.__name__,
        Add.__name__,
        Refinement.__name__,
        Alternative.__name__,
        Next.__name__,
    ):
        assert name in ns, f"missing factory {name!r}"
    assert callable(ns[factories.entity.__name__])


def test_eql_factory_namespace_picks_up_new_public_factories_automatically():
    # OCP: any public name defined directly in factories.py is available without
    # editing eql_factory_namespace() -- exercised here via one that was never
    # hand-listed (`a`, the `an` alias for words not starting with a vowel).
    ns = eql_factory_namespace()
    assert factories.a.__name__ in ns
    assert callable(ns[factories.a.__name__])


def test_eql_factory_namespace_excludes_names_shadowing_builtins():
    # `max`/`sum`/`min`/... are real public factories.py aggregators, but dumping
    # them into an interactive shell's flat namespace would shadow the builtins.
    ns = eql_factory_namespace()
    assert factories.max.__name__ not in ns
    assert factories.sum.__name__ not in ns


def test_eql_factory_namespace_exposes_module_for_builtin_shadowing_names():
    # The excluded names above stay reachable via the `eql` module object itself.
    ns = eql_factory_namespace()
    assert ns["eql"].__name__ == factories.__name__
    assert callable(ns["eql"].max)
    assert callable(ns["eql"].sum)


def test_eql_factory_namespace_excludes_private_names():
    ns = eql_factory_namespace()
    assert factories._quantify_or_build_match.__name__ not in ns


def test_eql_factory_namespace_excludes_names_imported_into_factories():
    # Match/Entity/etc. are imported into factories.py for its own use, not
    # defined there -- they shouldn't leak into the flat verb namespace.
    ns = eql_factory_namespace()
    assert factories.Match.__name__ not in ns
    assert factories.Entity.__name__ not in ns


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
    assert factories.entity.__name__ in scope and factories.refinement.__name__ in scope


def test_get_definition_scope_factory_overlay_wins():
    class Holder:
        pass

    obj = Holder()
    # User shadows a factory name; the overlay must win so the shell has the verb.
    attach_definition_scope(obj, {factories.entity.__name__: "shadowed"})
    scope = get_definition_scope(obj)
    assert callable(scope[factories.entity.__name__])


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
    assert factories.entity.__name__ in scope


def test_get_definition_scope_ignores_objects_with_no_attached_scope():
    fallback_local = "still_present"  # noqa: F841

    class Holder:
        pass

    scope = get_definition_scope(Holder())
    assert scope["fallback_local"] == "still_present"
