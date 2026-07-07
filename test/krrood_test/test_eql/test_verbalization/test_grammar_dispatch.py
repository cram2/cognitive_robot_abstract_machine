"""
Standalone unit tests for the grammar dispatch primitive (``select``) and the
EQL-tree fold (``engine.fold``) — exercised with synthetic constructs/rules so
the dispatch mechanics are validated independently of the real grammar.
"""

from __future__ import annotations

import pytest

from krrood.entity_query_language.verbalization.context import MicroplanningServices
from krrood.entity_query_language.verbalization.engine import fold
from krrood.entity_query_language.verbalization.exceptions import (
    UnverbalizableExpressionError,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    PhraseFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    RuleContext,
    PhraseRule,
    select,
)

from ...dataset.minimal_symbolic_expression import MinimalSymbolicExpression


# Synthetic construct hierarchy (deeper class == more specific).
class Base:
    _id_ = None
    _name_ = "base"


class Mid(Base):
    pass


class Leaf(Mid):
    pass


class Other:
    _id_ = None
    _name_ = "other"


def _rule_context():
    """A RuleContext for dispatch tests (no recursion needed by select)."""
    return RuleContext(
        recurse=lambda node, options: node, services=MicroplanningServices()
    )


def _custom(construct, name, build_fn, *, guard=None, base=PhraseRule):
    """Build a PhraseRule subclass instance for testing (subclass *base* to test specificity)."""
    namespace = {
        "construct": construct,
        "build": lambda self, node, context: build_fn(node, context),
    }
    if guard is not None:
        namespace["when"] = lambda self, node, context: guard(node)
    return type(f"R_{name}", (base,), namespace)()


def _rule(construct, name, **kw):
    """A rule whose build just emits its own name."""
    return _custom(construct, name, lambda node, context: WordFragment(name), **kw)


# ── select: specificity ──────────────────────────────────────────────────────


def test_select_prefers_deeper_construct():
    rules = [_rule(Base, "base"), _rule(Mid, "mid"), _rule(Leaf, "leaf")]
    assert type(select(Leaf(), rules, _rule_context())).__name__ == "R_leaf"
    assert type(select(Mid(), rules, _rule_context())).__name__ == "R_mid"
    assert type(select(Base(), rules, _rule_context())).__name__ == "R_base"


def test_select_guarded_beats_unguarded_same_construct():
    rules = [_rule(Mid, "plain"), _rule(Mid, "guarded", guard=lambda n: True)]
    assert type(select(Mid(), rules, _rule_context())).__name__ == "R_guarded"


def test_select_prefers_more_derived_rule_class_same_construct_both_guarded():
    # A rule that subclasses another (a special case of it) wins when both guards hold.
    base = _rule(Mid, "base", guard=lambda n: True)
    derived = _rule(Mid, "derived", guard=lambda n: True, base=type(base))
    rules = [base, derived]
    assert type(select(Mid(), rules, _rule_context())).__name__ == "R_derived"


def test_select_guard_can_exclude():
    rules = [_rule(Mid, "only-ok", guard=lambda n: getattr(n, "ok", False))]
    node = Mid()
    assert select(node, rules, _rule_context()) is None
    node.ok = True
    assert type(select(node, rules, _rule_context())).__name__ == "R_only-ok"


def test_select_returns_none_when_nothing_matches():
    rules = [_rule(Mid, "mid")]
    assert select(Other(), rules, _rule_context()) is None


# ── fold: dispatch, recursion, override, fallback ────────────────────────────


def test_fold_dispatches_to_selected_rule():
    rules = [_rule(Leaf, "leaf")]
    assert (
        flatten_fragment_to_plain_text(fold(Leaf(), MicroplanningServices(), rules))
        == "leaf"
    )


def test_fold_child_re_enters_the_fold():
    # A parent rule recurses into a child node via context.child.
    child = Other()
    parent = Mid()
    rules = [
        _custom(
            Mid,
            "parent",
            lambda node, context: PhraseFragment(
                [WordFragment("p"), context.child(child)]
            ),
        ),
        _custom(Other, "child", lambda node, context: WordFragment("c")),
    ]
    assert (
        flatten_fragment_to_plain_text(fold(parent, MicroplanningServices(), rules))
        == "p c"
    )


def test_fold_binding_override_short_circuits_before_dispatch():
    # Only a real expression carries an _id_ the override map is keyed on.
    node = MinimalSymbolicExpression()
    context = MicroplanningServices()
    context.binding.binding_overrides[node._id_] = WordFragment("OVERRIDE")
    # No rules at all — the override must still win.
    assert flatten_fragment_to_plain_text(fold(node, context, [])) == "OVERRIDE"


def test_fold_raises_when_no_rule_covers_the_node():
    node = Mid()
    node._name_ = "uncovered"
    with pytest.raises(UnverbalizableExpressionError):
        fold(node, MicroplanningServices(), [])


def test_enters_query_scope_wraps_build_but_not_when():
    """A rule declaring ``enters_query_scope`` builds at depth+1 (children see > 0), while its
    ``when`` guard still sees the rule's own (outer) depth — and the depth is restored after.
    """
    observed = {}

    class ScopedRule(PhraseRule):
        construct = Mid
        enters_query_scope = True

        def when(self, node, context):
            observed["when_depth"] = context.configuration.query_depth
            return True

        def build(self, node, context):
            observed["build_depth"] = context.configuration.query_depth
            return WordFragment("scoped")

    context = MicroplanningServices()
    fold(Mid(), context, [ScopedRule()])
    assert observed["when_depth"] == 0  # the guard judges the rule's own position
    assert observed["build_depth"] == 1  # everything inside is one query level deeper
    assert context.configuration.query_depth == 0  # restored on exit
