"""
Surgical tests for Phase 2 of the auto-condition inference feature:
``condition_resolver.py`` — ResolutionSource, ResolvedCondition,
ConditionResolver ABC, ChainConditionResolver, and the two concrete stubs.

Each test class pins exactly one contract; each test method verifies one
observable guarantee.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import pytest

from krrood.entity_query_language.rdr.condition_resolver import (
    ChainConditionResolver,
    ConditionResolver,
    CornerCaseKnowledgeResolver,
    ResolvedCondition,
    ResolutionSource,
    TargetKnowledgeResolver,
)

# ---------------------------------------------------------------------------
# Minimal stub helpers — named after the pattern they exercise, not "Mock"
# ---------------------------------------------------------------------------

# A sentinel SymbolicExpression stand-in.  The resolver tests do not execute
# EQL expressions; they only inspect the *returned object*.  Using a plain
# sentinel avoids pulling in the full EQL machinery for structural tests.
_SENTINEL_EXPR = object()


class _AlwaysNoneResolver(ConditionResolver):
    """A resolver that never succeeds — always returns None."""

    def resolve(self, *args: Any, **kwargs: Any) -> Optional[ResolvedCondition]:
        return None


class _AlwaysTargetResolver(ConditionResolver):
    """A resolver that always returns a fixed TARGET_KNOWLEDGE result."""

    def __init__(self) -> None:
        self.call_count = 0

    def resolve(self, *args: Any, **kwargs: Any) -> Optional[ResolvedCondition]:
        self.call_count += 1
        return ResolvedCondition(_SENTINEL_EXPR, ResolutionSource.TARGET_KNOWLEDGE)


class _AlwaysCornerResolver(ConditionResolver):
    """A resolver that always returns a fixed CORNER_CASE_KNOWLEDGE result."""

    def __init__(self) -> None:
        self.call_count = 0

    def resolve(self, *args: Any, **kwargs: Any) -> Optional[ResolvedCondition]:
        self.call_count += 1
        return ResolvedCondition(_SENTINEL_EXPR, ResolutionSource.CORNER_CASE_KNOWLEDGE)


# Convenience: call a resolver / chain with dummy arguments.
_DUMMY_ARGS = (None, None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# ResolutionSource enum
# ---------------------------------------------------------------------------


class TestResolutionSource:
    """ResolutionSource exposes exactly the two expected enum members."""

    def test_target_knowledge_member_exists(self):
        """ResolutionSource.TARGET_KNOWLEDGE must be a valid enum member.

        Guarantee: the attribute resolves without AttributeError.
        """
        member = ResolutionSource.TARGET_KNOWLEDGE
        assert isinstance(member, ResolutionSource)

    def test_corner_case_knowledge_member_exists(self):
        """ResolutionSource.CORNER_CASE_KNOWLEDGE must be a valid enum member.

        Guarantee: the attribute resolves without AttributeError.
        """
        member = ResolutionSource.CORNER_CASE_KNOWLEDGE
        assert isinstance(member, ResolutionSource)

    def test_target_knowledge_string_value(self):
        """ResolutionSource.TARGET_KNOWLEDGE has the value 'target_knowledge'.

        Guarantee: callers that serialise the enum by value get a stable string.
        """
        assert ResolutionSource.TARGET_KNOWLEDGE.value == "target_knowledge"

    def test_corner_case_knowledge_string_value(self):
        """ResolutionSource.CORNER_CASE_KNOWLEDGE has the value 'corner_case_knowledge'.

        Guarantee: callers that serialise the enum by value get a stable string.
        """
        assert ResolutionSource.CORNER_CASE_KNOWLEDGE.value == "corner_case_knowledge"

    def test_two_members_total(self):
        """Exactly two members exist — no accidental extras.

        Guarantee: future additions need an explicit test update (API surface is frozen).
        """
        assert len(list(ResolutionSource)) == 2

    def test_members_are_distinct(self):
        """The two members are not equal to each other.

        Guarantee: switching on source never confuses the two strategies.
        """
        assert ResolutionSource.TARGET_KNOWLEDGE != ResolutionSource.CORNER_CASE_KNOWLEDGE


# ---------------------------------------------------------------------------
# ResolvedCondition dataclass
# ---------------------------------------------------------------------------


class TestResolvedCondition:
    """ResolvedCondition is a frozen dataclass with expression and source fields."""

    def test_construction_stores_expression(self):
        """The expression passed at construction is retrievable unchanged.

        Guarantee: no copy or transformation is applied to the expression.
        """
        rc = ResolvedCondition(_SENTINEL_EXPR, ResolutionSource.TARGET_KNOWLEDGE)
        assert rc.expression is _SENTINEL_EXPR

    def test_construction_stores_source(self):
        """The source passed at construction is retrievable unchanged.

        Guarantee: the provenance enum value is not coerced or renamed.
        """
        rc = ResolvedCondition(_SENTINEL_EXPR, ResolutionSource.CORNER_CASE_KNOWLEDGE)
        assert rc.source is ResolutionSource.CORNER_CASE_KNOWLEDGE

    def test_is_frozen_expression_field(self):
        """Mutating the expression field of a ResolvedCondition raises an error.

        Guarantee: frozen=True is in effect — callers cannot accidentally overwrite
        a resolved condition's expression after creation.
        """
        rc = ResolvedCondition(_SENTINEL_EXPR, ResolutionSource.TARGET_KNOWLEDGE)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            rc.expression = object()  # type: ignore[misc]

    def test_is_frozen_source_field(self):
        """Mutating the source field of a ResolvedCondition raises an error.

        Guarantee: frozen=True applies to every field, not just expression.
        """
        rc = ResolvedCondition(_SENTINEL_EXPR, ResolutionSource.TARGET_KNOWLEDGE)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            rc.source = ResolutionSource.CORNER_CASE_KNOWLEDGE  # type: ignore[misc]

    def test_equality_based_on_field_values(self):
        """Two ResolvedConditions with the same fields compare as equal.

        Guarantee: frozen dataclass equality semantics are in place (structural equality).
        """
        expr = object()
        rc1 = ResolvedCondition(expr, ResolutionSource.TARGET_KNOWLEDGE)
        rc2 = ResolvedCondition(expr, ResolutionSource.TARGET_KNOWLEDGE)
        assert rc1 == rc2

    def test_inequality_on_different_source(self):
        """Two ResolvedConditions with different sources are not equal.

        Guarantee: source is part of the equality contract.
        """
        rc1 = ResolvedCondition(_SENTINEL_EXPR, ResolutionSource.TARGET_KNOWLEDGE)
        rc2 = ResolvedCondition(_SENTINEL_EXPR, ResolutionSource.CORNER_CASE_KNOWLEDGE)
        assert rc1 != rc2


# ---------------------------------------------------------------------------
# ChainConditionResolver structural tests
# ---------------------------------------------------------------------------


class TestChainConditionResolver:
    """ChainConditionResolver implements chain-of-responsibility correctly."""

    def test_returns_second_resolver_result_when_first_returns_none(self):
        """When the first resolver returns None, the chain falls through to the second.

        Guarantee: a None result from one resolver does not swallow a valid result
        that a later resolver can provide.
        """
        first = _AlwaysNoneResolver()
        second = _AlwaysTargetResolver()
        chain = ChainConditionResolver([first, second])

        result = chain.resolve(*_DUMMY_ARGS)

        assert result is not None
        assert isinstance(result, ResolvedCondition)
        assert result.source is ResolutionSource.TARGET_KNOWLEDGE

    def test_short_circuits_when_first_resolver_returns_non_none(self):
        """When the first resolver returns a result, the second is never called.

        Guarantee: chain short-circuits at the first non-None result — O(1) in
        the best case and prevents unexpected side effects from later resolvers.
        """
        first = _AlwaysTargetResolver()
        second = _AlwaysCornerResolver()
        chain = ChainConditionResolver([first, second])

        result = chain.resolve(*_DUMMY_ARGS)

        assert result is not None
        assert result.source is ResolutionSource.TARGET_KNOWLEDGE
        assert second.call_count == 0

    def test_returns_none_when_all_resolvers_return_none(self):
        """When every resolver in the chain returns None, the chain itself returns None.

        Guarantee: the caller receives None and can fall back to the expert prompt.
        """
        chain = ChainConditionResolver(
            [_AlwaysNoneResolver(), _AlwaysNoneResolver(), _AlwaysNoneResolver()]
        )

        result = chain.resolve(*_DUMMY_ARGS)

        assert result is None

    def test_empty_chain_returns_none(self):
        """A chain with no resolvers returns None without error.

        Guarantee: zero-length resolver list is a valid (though degenerate) configuration.
        """
        chain = ChainConditionResolver([])

        result = chain.resolve(*_DUMMY_ARGS)

        assert result is None

    def test_backward_inference_default_returns_chain_condition_resolver(self):
        """backward_inference_default() produces a ChainConditionResolver instance.

        Guarantee: the factory method returns the correct type for downstream isinstance
        checks and duck-typing.
        """
        result = ChainConditionResolver.backward_inference_default()

        assert isinstance(result, ChainConditionResolver)

    def test_backward_inference_default_has_two_resolvers(self):
        """backward_inference_default() installs exactly two resolvers.

        Guarantee: the standard chain has the precise arity implied by the two-phase
        algorithm design (Phase 1 + Phase 2).
        """
        chain = ChainConditionResolver.backward_inference_default()

        assert len(chain.resolvers) == 2

    def test_backward_inference_default_first_resolver_is_target_knowledge(self):
        """backward_inference_default() places TargetKnowledgeResolver first.

        Guarantee: Phase 1 (target knowledge) is always tried before Phase 2
        (corner-case knowledge) — the ordering is part of the public contract.
        """
        chain = ChainConditionResolver.backward_inference_default()

        assert isinstance(chain.resolvers[0], TargetKnowledgeResolver)

    def test_backward_inference_default_second_resolver_is_corner_case_knowledge(self):
        """backward_inference_default() places CornerCaseKnowledgeResolver second.

        Guarantee: Phase 2 (corner-case knowledge) is the fallback strategy when
        Phase 1 finds nothing.
        """
        chain = ChainConditionResolver.backward_inference_default()

        assert isinstance(chain.resolvers[1], CornerCaseKnowledgeResolver)

    def test_chain_returns_first_resolver_result_not_second(self):
        """The result value comes from the first non-None resolver, not a later one.

        Guarantee: result identity is preserved end-to-end through the chain loop —
        no extra wrapping or replacement occurs.
        """
        first = _AlwaysTargetResolver()
        second = _AlwaysCornerResolver()
        chain = ChainConditionResolver([first, second])

        result = chain.resolve(*_DUMMY_ARGS)

        # The first resolver returned TARGET_KNOWLEDGE; that must be what we get back.
        assert result is not None
        assert result.source is ResolutionSource.TARGET_KNOWLEDGE

    def test_all_resolvers_are_tried_when_none_succeed(self):
        """Every resolver is invoked when none of them return a result.

        Guarantee: the chain does not bail out early on None — it exhausts all options
        before giving up.
        """
        first = _AlwaysTargetResolver()
        second = _AlwaysCornerResolver()

        # Wrap each in a counting None-always resolver to observe call counts.
        class _CountingNone(ConditionResolver):
            def __init__(self) -> None:
                self.call_count = 0

            def resolve(self, *args: Any, **kwargs: Any) -> Optional[ResolvedCondition]:
                self.call_count += 1
                return None

        r1 = _CountingNone()
        r2 = _CountingNone()
        r3 = _CountingNone()
        chain = ChainConditionResolver([r1, r2, r3])
        chain.resolve(*_DUMMY_ARGS)

        assert r1.call_count == 1
        assert r2.call_count == 1
        assert r3.call_count == 1


# ---------------------------------------------------------------------------
# ConditionResolver ABC contract
# ---------------------------------------------------------------------------


class TestConditionResolverABC:
    """ConditionResolver is an abstract base class that cannot be instantiated directly."""

    def test_cannot_instantiate_abstract_resolver(self):
        """Instantiating ConditionResolver directly must raise TypeError.

        Guarantee: the ABC contract is enforced — concrete subclasses must implement
        resolve() or they cannot be constructed.
        """
        with pytest.raises(TypeError):
            ConditionResolver()  # type: ignore[abstract]

    def test_concrete_subclass_without_resolve_cannot_instantiate(self):
        """A subclass that does not implement resolve() also raises TypeError.

        Guarantee: the @abstractmethod decorator is not accidentally omitted or
        bypassed by a partial implementation.
        """

        class _IncompleteResolver(ConditionResolver):
            pass  # resolve() not implemented

        with pytest.raises(TypeError):
            _IncompleteResolver()  # type: ignore[abstract]

    def test_concrete_subclass_with_resolve_can_instantiate(self):
        """A subclass that implements resolve() can be constructed without error.

        Guarantee: the ABC permits any concrete implementation, not just the two
        built-in strategies.
        """
        # _AlwaysNoneResolver implements resolve(); construction must succeed.
        resolver = _AlwaysNoneResolver()
        assert isinstance(resolver, ConditionResolver)
