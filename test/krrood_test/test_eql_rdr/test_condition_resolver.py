"""
Self-contained tests for auto-condition resolution
(:mod:`krrood.entity_query_language.rdr.condition_resolver`).

``ConclusionKnowledge``/``SufficientConditionSet``/``GuardCondition`` are built by hand
here rather than through a real rule tree or :class:`EQLSingleClassRDR`, so this test
module -- and the condition-resolution slice it covers -- stays testable independently
of the rest of the RDR engine.
"""

from dataclasses import dataclass
from enum import Enum

from krrood.entity_query_language.factories import not_, variable
from krrood.entity_query_language.operators.core_logical_operators import Not
from krrood.entity_query_language.rdr.backward_inference import (
    ConclusionSufficientConditionSets,
    GuardCondition,
    SufficientConditionSet,
)
from krrood.entity_query_language.rdr.condition_resolver import (
    ChainConditionResolver,
    ConditionResolver,
    CornerCaseKnowledgeResolver,
    ResolutionMode,
    ResolvedCondition,
    TargetSufficientConditionsBasedResolver,
)


@dataclass(unsafe_hash=True)
class Animal:
    """Minimal RDR classification target used only by this test module."""

    name: str
    has_fur: bool = False
    can_fly: bool = False
    lays_eggs: bool = False


class Species(Enum):
    MAMMAL = "mammal"
    BIRD = "bird"
    REPTILE = "reptile"


def _empty_knowledge(value):
    return ConclusionSufficientConditionSets(conclusion_value=value, sufficient_condition_sets=())


# ---------------------------------------------------------------------------
# ResolutionMode / ResolvedCondition
# ---------------------------------------------------------------------------


def test_resolution_mode_has_automatic_and_hint_members():
    assert ResolutionMode.AUTOMATIC.value == "automatic"
    assert ResolutionMode.HINT.value == "hint"


def test_resolved_condition_carries_expression_and_resolver_type():
    animal = variable(Animal, domain=[])
    resolved = ResolvedCondition(animal.has_fur, TargetSufficientConditionsBasedResolver)

    assert resolved.expression is animal.has_fur
    assert resolved.resolver_type is TargetSufficientConditionsBasedResolver


# ---------------------------------------------------------------------------
# _materialize
# ---------------------------------------------------------------------------


def test_guard_expression_returns_the_bare_expression_when_not_negated():
    animal = variable(Animal, domain=[])
    guard = GuardCondition(animal.has_fur, negated=False)

    assert guard.as_expression is animal.has_fur


def test_guard_expression_wraps_a_negated_guard_in_not():
    animal = variable(Animal, domain=[])
    guard = GuardCondition(animal.has_fur, negated=True)

    guard_expression = guard.as_expression

    assert isinstance(guard_expression, Not)
    assert guard_expression._child_ is animal.has_fur

    # materialized itself (not_(has_fur)) must be True exactly when has_fur is
    # False, and False when has_fur is True -- the inverse of the bare condition.
    materialized_guard = GuardCondition(guard_expression, negated=False)
    cat = Animal("cat", has_fur=True)
    snake = Animal("snake", has_fur=False)
    assert materialized_guard.holds_for(animal, cat) is False
    assert materialized_guard.holds_for(animal, snake) is True


# ---------------------------------------------------------------------------
# TargetKnowledgeResolver
# ---------------------------------------------------------------------------


def test_target_knowledge_resolver_finds_a_discriminating_guard():
    animal = variable(Animal, domain=[])
    target_knowledge = ConclusionSufficientConditionSets(
        conclusion_value=Species.BIRD,
        sufficient_condition_sets=(
            SufficientConditionSet(
                conditions=(GuardCondition(animal.can_fly, negated=False),)
            ),
        ),
    )

    case = Animal("bat", can_fly=True)
    corner_case = Animal("cat", can_fly=False)

    resolved = TargetSufficientConditionsBasedResolver().resolve(
        case=case,
        case_variable=animal,
        target_conclusion=Species.BIRD,
        current_conclusion=Species.MAMMAL,
        corner_case=corner_case,
        target_knowledge=target_knowledge,
        current_knowledge=_empty_knowledge(Species.MAMMAL),
    )

    assert resolved is not None
    assert resolved.resolver_type is TargetSufficientConditionsBasedResolver
    assert resolved.expression is animal.can_fly


def test_target_knowledge_resolver_returns_none_when_no_guard_discriminates():
    animal = variable(Animal, domain=[])
    target_knowledge = ConclusionSufficientConditionSets(
        conclusion_value=Species.BIRD,
        sufficient_condition_sets=(
            SufficientConditionSet(
                conditions=(GuardCondition(animal.can_fly, negated=False),)
            ),
        ),
    )

    # Both case and corner_case agree on can_fly -- no guard tells them apart.
    case = Animal("bat", can_fly=True)
    corner_case = Animal("eagle", can_fly=True)

    resolved = TargetSufficientConditionsBasedResolver().resolve(
        case=case,
        case_variable=animal,
        target_conclusion=Species.BIRD,
        current_conclusion=Species.MAMMAL,
        corner_case=corner_case,
        target_knowledge=target_knowledge,
        current_knowledge=_empty_knowledge(Species.MAMMAL),
    )

    assert resolved is None


def test_target_knowledge_resolver_returns_none_with_no_known_paths():
    animal = variable(Animal, domain=[])

    resolved = TargetSufficientConditionsBasedResolver().resolve(
        case=Animal("bat"),
        case_variable=animal,
        target_conclusion=Species.BIRD,
        current_conclusion=Species.MAMMAL,
        corner_case=Animal("cat"),
        target_knowledge=_empty_knowledge(Species.BIRD),
        current_knowledge=_empty_knowledge(Species.MAMMAL),
    )

    assert resolved is None


# ---------------------------------------------------------------------------
# CornerCaseKnowledgeResolver
# ---------------------------------------------------------------------------


def _two_path_current_knowledge(animal):
    """Two independent sufficient condition sets both concluding the wrong value.

    ``active_guard`` (can_fly) models the path that actually fired and caused the
    misclassification; ``non_active_guard`` (lays_eggs) models an unrelated,
    already-known path to the same wrong conclusion.
    """
    active_guard = animal.can_fly == True  # noqa: E712
    non_active_guard = animal.lays_eggs == True  # noqa: E712
    active_path = SufficientConditionSet(
        conditions=(GuardCondition(active_guard, negated=False),)
    )
    non_active_path = SufficientConditionSet(
        conditions=(GuardCondition(non_active_guard, negated=False),)
    )
    knowledge = ConclusionSufficientConditionSets(
        conclusion_value=Species.MAMMAL,
        sufficient_condition_sets=(active_path, non_active_path),
    )
    return active_guard, non_active_guard, knowledge


def test_corner_case_resolver_skips_the_active_path_via_firing_anchor():
    animal = variable(Animal, domain=[])
    active_guard, non_active_guard, current_knowledge = _two_path_current_knowledge(animal)

    # Discriminates on the active guard only -- must NOT be returned, since that
    # path is excluded via firing_anchor.
    case = Animal("weird", can_fly=True, lays_eggs=False)
    corner_case = Animal("plain", can_fly=False, lays_eggs=False)

    resolved = CornerCaseKnowledgeResolver().resolve(
        case=case,
        case_variable=animal,
        target_conclusion=Species.BIRD,
        current_conclusion=Species.MAMMAL,
        corner_case=corner_case,
        target_knowledge=_empty_knowledge(Species.BIRD),
        current_knowledge=current_knowledge,
        firing_anchor=active_guard,
    )

    assert resolved is None


def test_corner_case_resolver_finds_a_discriminating_guard_on_a_non_active_path():
    animal = variable(Animal, domain=[])
    active_guard, non_active_guard, current_knowledge = _two_path_current_knowledge(animal)

    case = Animal("weird", can_fly=False, lays_eggs=True)
    corner_case = Animal("plain", can_fly=False, lays_eggs=False)

    resolved = CornerCaseKnowledgeResolver().resolve(
        case=case,
        case_variable=animal,
        target_conclusion=Species.BIRD,
        current_conclusion=Species.MAMMAL,
        corner_case=corner_case,
        target_knowledge=_empty_knowledge(Species.BIRD),
        current_knowledge=current_knowledge,
        firing_anchor=active_guard,
    )

    assert resolved is not None
    assert resolved.resolver_type is CornerCaseKnowledgeResolver
    assert resolved.expression is non_active_guard


def test_corner_case_resolver_searches_every_path_when_firing_anchor_is_none():
    animal = variable(Animal, domain=[])
    active_guard, non_active_guard, current_knowledge = _two_path_current_knowledge(animal)

    # Discriminates on the guard that would otherwise be excluded as "active" --
    # with no firing_anchor, nothing is excluded, so it must be found.
    case = Animal("weird", can_fly=True, lays_eggs=False)
    corner_case = Animal("plain", can_fly=False, lays_eggs=False)

    resolved = CornerCaseKnowledgeResolver().resolve(
        case=case,
        case_variable=animal,
        target_conclusion=Species.BIRD,
        current_conclusion=Species.MAMMAL,
        corner_case=corner_case,
        target_knowledge=_empty_knowledge(Species.BIRD),
        current_knowledge=current_knowledge,
        firing_anchor=None,
    )

    assert resolved is not None
    assert resolved.expression is active_guard


def test_corner_case_resolver_returns_none_when_no_path_discriminates():
    animal = variable(Animal, domain=[])
    _active_guard, _non_active_guard, current_knowledge = _two_path_current_knowledge(animal)

    case = Animal("weird", can_fly=False, lays_eggs=False)
    corner_case = Animal("plain", can_fly=False, lays_eggs=False)

    resolved = CornerCaseKnowledgeResolver().resolve(
        case=case,
        case_variable=animal,
        target_conclusion=Species.BIRD,
        current_conclusion=Species.MAMMAL,
        corner_case=corner_case,
        target_knowledge=_empty_knowledge(Species.BIRD),
        current_knowledge=current_knowledge,
        firing_anchor=None,
    )

    assert resolved is None


# ---------------------------------------------------------------------------
# ChainConditionResolver
# ---------------------------------------------------------------------------


class _AlwaysFailsResolver(ConditionResolver):
    def resolve(self, *args, **kwargs):
        return None


class _AlwaysResolvesResolver(ConditionResolver):
    def __init__(self, expression):
        self.expression = expression

    def resolve(self, *args, **kwargs):
        return ResolvedCondition(self.expression, type(self))


def test_chain_resolver_returns_the_first_non_none_result():
    animal = variable(Animal, domain=[])
    chain = ChainConditionResolver(
        resolvers=[
            _AlwaysFailsResolver(),
            _AlwaysResolvesResolver(animal.has_fur),
            _AlwaysFailsResolver(),
        ]
    )

    resolved = chain.resolve(
        case=Animal("cat"),
        case_variable=animal,
        target_conclusion=Species.MAMMAL,
        current_conclusion=Species.REPTILE,
        corner_case=Animal("snake"),
        target_knowledge=_empty_knowledge(Species.MAMMAL),
        current_knowledge=_empty_knowledge(Species.REPTILE),
    )

    assert resolved is not None
    assert resolved.expression is animal.has_fur
    assert resolved.resolver_type is _AlwaysResolvesResolver


def test_chain_resolver_returns_none_when_every_resolver_fails():
    animal = variable(Animal, domain=[])
    chain = ChainConditionResolver(resolvers=[_AlwaysFailsResolver(), _AlwaysFailsResolver()])

    resolved = chain.resolve(
        case=Animal("cat"),
        case_variable=animal,
        target_conclusion=Species.MAMMAL,
        current_conclusion=Species.REPTILE,
        corner_case=Animal("snake"),
        target_knowledge=_empty_knowledge(Species.MAMMAL),
        current_knowledge=_empty_knowledge(Species.REPTILE),
    )

    assert resolved is None


def test_chain_resolver_default_order_is_target_then_corner_case():
    chain = ChainConditionResolver.backward_inference_default()

    assert [type(resolver) for resolver in chain.resolvers] == [
        TargetSufficientConditionsBasedResolver,
        CornerCaseKnowledgeResolver,
    ]


def test_chain_resolver_default_prefers_target_knowledge_over_corner_case():
    animal = variable(Animal, domain=[])
    target_knowledge = ConclusionSufficientConditionSets(
        conclusion_value=Species.BIRD,
        sufficient_condition_sets=(
            SufficientConditionSet(
                conditions=(GuardCondition(animal.can_fly, negated=False),)
            ),
        ),
    )
    active_guard, _non_active_guard, current_knowledge = _two_path_current_knowledge(animal)

    case = Animal("bat", can_fly=True, lays_eggs=True)
    corner_case = Animal("cat", can_fly=False, lays_eggs=False)

    resolved = ChainConditionResolver.backward_inference_default().resolve(
        case=case,
        case_variable=animal,
        target_conclusion=Species.BIRD,
        current_conclusion=Species.MAMMAL,
        corner_case=corner_case,
        target_knowledge=target_knowledge,
        current_knowledge=current_knowledge,
        firing_anchor=active_guard,
    )

    assert resolved is not None
    assert resolved.resolver_type is TargetSufficientConditionsBasedResolver
    assert resolved.expression is animal.can_fly
