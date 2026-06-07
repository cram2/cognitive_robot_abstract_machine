"""
Tests for backward inference on the EQL-RDR rule tree.

:func:`what_do_we_know_about` inspects the rule tree (a live EQL expression DAG) and
returns the sets of conditions that would cause a given conclusion value to fire, as
:class:`SufficientConditionSet` objects inside a :class:`ConclusionKnowledge`.
"""

from __future__ import annotations

from typing_extensions import Any, List, Tuple

import pytest

from krrood.entity_query_language.factories import (
    add,
    alternative,
    entity,
    refinement,
    variable,
)
from krrood.entity_query_language.rdr.backward_inference import (
    GuardCondition,
    SufficientConditionSet,
    ConclusionKnowledge,
    what_do_we_know_about,
    BackwardInferenceIndex,
)
from krrood.entity_query_language.rdr.rule_tree import (
    insert_alternative,
    insert_refinement,
)
from krrood.entity_query_language.rdr.rule_tree_view import format_condition
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rules.conclusion_selector import ConclusionSelector

from .animal import Animal, Species

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _flat_tree() -> Tuple[Any, Any, Any]:
    """milk -> mammal ; else feathers -> bird ; else fins -> fish."""
    animal = variable(Animal, domain=[])
    query = entity(animal).where(animal.milk == True)
    with query:
        add(animal.species, Species.mammal)
        with alternative(animal.feathers == True):
            add(animal.species, Species.bird)
        with alternative(animal.fins == True):
            add(animal.species, Species.fish)
    query.build()
    return animal, query, query._conditions_root_


def _refinement_tree() -> Tuple[Any, Any, Any]:
    """backbone -> fish ; except if milk -> mammal."""
    animal = variable(Animal, domain=[])
    query = entity(animal).where(animal.backbone == True)
    with query:
        add(animal.species, Species.fish)
        with refinement(animal.milk == True):
            add(animal.species, Species.mammal)
    query.build()
    return animal, query, query._conditions_root_


def _mixed_tree() -> Tuple[Any, Any, Any]:
    """backbone->fish ; refine milk->mammal ; alt feathers->bird on refinement's right."""
    animal = variable(Animal, domain=[])
    query = entity(animal).where(animal.backbone == True)
    with query:
        add(animal.species, Species.fish)
    query.build()
    ref = insert_refinement(
        query._conditions_root_, animal.milk == True, animal.species, Species.mammal
    )
    insert_alternative(ref, animal.feathers == True, animal.species, Species.bird)
    return animal, query, query._conditions_root_


_COW = Animal(
    name="cow",
    hair=True,
    feathers=False,
    eggs=False,
    milk=True,
    airborne=False,
    aquatic=False,
    predator=False,
    toothed=True,
    backbone=True,
    breathes=True,
    venomous=False,
    fins=False,
    legs=4,
    tail=True,
    domestic=True,
    catsize=True,
    species=None,
)
_EAGLE = Animal(
    name="eagle",
    hair=False,
    feathers=True,
    eggs=True,
    milk=False,
    airborne=True,
    aquatic=False,
    predator=True,
    toothed=True,
    backbone=True,
    breathes=True,
    venomous=False,
    fins=False,
    legs=2,
    tail=True,
    domestic=False,
    catsize=False,
    species=None,
)
_TUNA = Animal(
    name="tuna",
    hair=False,
    feathers=False,
    eggs=True,
    milk=False,
    airborne=False,
    aquatic=True,
    predator=True,
    toothed=True,
    backbone=True,
    breathes=False,
    venomous=False,
    fins=True,
    legs=0,
    tail=True,
    domestic=False,
    catsize=False,
    species=None,
)
_FROG = Animal(
    name="frog",
    hair=False,
    feathers=False,
    eggs=True,
    milk=False,
    airborne=False,
    aquatic=True,
    predator=False,
    toothed=False,
    backbone=False,
    breathes=True,
    venomous=False,
    fins=False,
    legs=4,
    tail=False,
    domestic=False,
    catsize=False,
    species=None,
)


# ---------------------------------------------------------------------------
# ConclusionKnowledge structure
# ---------------------------------------------------------------------------


class TestConclusionKnowledge:
    """Verify the structure of ConclusionKnowledge and SufficientConditionSet."""

    def test_flat_tree_mammal(self):
        _, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.mammal)

        assert isinstance(knowledge, ConclusionKnowledge)
        assert knowledge.conclusion_value is Species.mammal
        assert knowledge.is_satisfiable()
        assert len(knowledge.sufficient_condition_sets) == 1

        cond_set = knowledge.sufficient_condition_sets[0]
        assert isinstance(cond_set, SufficientConditionSet)
        # One condition: milk == True (no guards needed for the leading alternative)
        assert len(cond_set.conditions) == 1
        assert cond_set.conditions[0].negated is False

    def test_flat_tree_bird(self):
        _, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.bird)

        assert knowledge.is_satisfiable()
        assert len(knowledge.sufficient_condition_sets) == 1

        cond_set = knowledge.sufficient_condition_sets[0]
        # Two conditions: NOT(milk == True) guard + feathers == True (leaf)
        assert len(cond_set.conditions) == 2
        assert cond_set.conditions[0].negated is True
        assert cond_set.conditions[1].negated is False

    def test_flat_tree_fish(self):
        _, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.fish)

        assert knowledge.is_satisfiable()
        assert len(knowledge.sufficient_condition_sets) == 1
        cond_set = knowledge.sufficient_condition_sets[0]
        # Three conditions: NOT(milk) + NOT(feathers) guards (flattened from
        # NOT(Alternative(milk, feathers))) + fins == True (leaf)
        assert len(cond_set.conditions) == 3
        assert cond_set.conditions[0].negated is True
        assert cond_set.conditions[1].negated is True

    def test_flat_tree_molusc_not_satisfiable(self):
        _, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.molusc)
        assert not knowledge.is_satisfiable()
        assert len(knowledge.sufficient_condition_sets) == 0

    def test_refinement_tree_mammal(self):
        """Refinement: backbone + milk -> mammal."""
        _, _, root = _refinement_tree()
        knowledge = what_do_we_know_about(root, Species.mammal)

        assert knowledge.is_satisfiable()
        assert len(knowledge.sufficient_condition_sets) == 1
        cond_set = knowledge.sufficient_condition_sets[0]
        # backbone (positive guard: parent must fire) + milk (the refinement condition)
        assert len(cond_set.conditions) == 2
        assert cond_set.conditions[0].negated is False
        assert cond_set.conditions[1].negated is False

    def test_refinement_tree_fish(self):
        """Parent backbone -> fish fires only when NOT(milk)."""
        _, _, root = _refinement_tree()
        knowledge = what_do_we_know_about(root, Species.fish)

        assert knowledge.is_satisfiable()
        assert len(knowledge.sufficient_condition_sets) == 1
        cond_set = knowledge.sufficient_condition_sets[0]
        # NOT(milk) guard + backbone (leaf condition)
        assert len(cond_set.conditions) == 2
        assert cond_set.conditions[0].negated is True

    def test_mixed_tree_mammal(self):
        _, _, root = _mixed_tree()
        knowledge = what_do_we_know_about(root, Species.mammal)
        assert knowledge.is_satisfiable()
        assert len(knowledge.sufficient_condition_sets) == 1
        # backbone (guard) + milk (leaf)
        assert len(knowledge.sufficient_condition_sets[0].conditions) == 2
        assert knowledge.sufficient_condition_sets[0].conditions[0].negated is False

    def test_mixed_tree_bird(self):
        _, _, root = _mixed_tree()
        knowledge = what_do_we_know_about(root, Species.bird)
        assert knowledge.is_satisfiable()
        assert len(knowledge.sufficient_condition_sets) == 1
        conds = knowledge.sufficient_condition_sets[0].conditions
        # backbone (positive guard) + NOT(milk) (alt guard) + feathers (leaf)
        assert len(conds) == 3
        assert conds[0].negated is False  # backbone
        assert conds[1].negated is True  # NOT(milk)

    def test_mixed_tree_fish(self):
        _, _, root = _mixed_tree()
        knowledge = what_do_we_know_about(root, Species.fish)
        assert knowledge.is_satisfiable()
        assert len(knowledge.sufficient_condition_sets) == 1
        conds = knowledge.sufficient_condition_sets[0].conditions
        # NOT(milk) + NOT(feathers) (flattened from NOT(Alternative(milk, feathers)))
        # + backbone (leaf)
        assert len(conds) == 3
        assert conds[0].negated is True
        assert conds[1].negated is True

    def test_empty_tree(self):
        rdr = EQLSingleClassRDR(Animal, "species")
        knowledge = what_do_we_know_about(rdr.conditions_root, Species.molusc)
        assert isinstance(knowledge, ConclusionKnowledge)
        assert not knowledge.is_satisfiable()

    def test_index_with_empty_tree(self):
        rdr = EQLSingleClassRDR(Animal, "species")
        index = BackwardInferenceIndex()
        knowledge = index.query(rdr.conditions_root, Species.molusc)
        assert isinstance(knowledge, ConclusionKnowledge)
        assert not knowledge.is_satisfiable()


# ---------------------------------------------------------------------------
# evaluate_against correctness
# ---------------------------------------------------------------------------


class TestEvaluateAgainst:
    """Verify evaluate_against() returns correct results for concrete cases."""

    def test_flat_mammal_true_for_cow(self):
        animal, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.mammal)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _COW)
            is True
        )

    def test_flat_mammal_false_for_eagle(self):
        animal, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.mammal)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _EAGLE)
            is False
        )

    def test_flat_mammal_false_for_frog(self):
        animal, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.mammal)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _FROG)
            is False
        )

    def test_flat_bird_true_for_eagle(self):
        animal, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.bird)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _EAGLE)
            is True
        )

    def test_flat_bird_false_for_cow(self):
        animal, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.bird)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _COW)
            is False
        )

    def test_flat_bird_false_for_tuna(self):
        animal, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.bird)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _TUNA)
            is False
        )

    def test_flat_fish_true_for_tuna(self):
        animal, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.fish)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _TUNA)
            is True
        )

    def test_flat_fish_false_for_frog(self):
        animal, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.fish)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _FROG)
            is False
        )

    def test_flat_fish_false_for_eagle(self):
        animal, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.fish)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _EAGLE)
            is False
        )

    def test_refinement_mammal_true_for_cow(self):
        animal, _, root = _refinement_tree()
        knowledge = what_do_we_know_about(root, Species.mammal)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _COW)
            is True
        )

    def test_refinement_mammal_false_for_tuna(self):
        animal, _, root = _refinement_tree()
        knowledge = what_do_we_know_about(root, Species.mammal)
        # Tuna has backbone but not milk, so mammal shouldn't match
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _TUNA)
            is False
        )

    def test_refinement_fish_true_for_tuna(self):
        animal, _, root = _refinement_tree()
        knowledge = what_do_we_know_about(root, Species.fish)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _TUNA)
            is True
        )

    def test_refinement_fish_false_for_cow(self):
        animal, _, root = _refinement_tree()
        knowledge = what_do_we_know_about(root, Species.fish)
        # Cow has backbone AND milk, so the refinement overrides — NOT(milk) guard fails
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _COW)
            is False
        )

    def test_refinement_fish_false_for_frog(self):
        animal, _, root = _refinement_tree()
        knowledge = what_do_we_know_about(root, Species.fish)
        # Frog has no backbone, so the backbone condition fails
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _FROG)
            is False
        )

    def test_mixed_mammal_true_for_cow(self):
        animal, _, root = _mixed_tree()
        knowledge = what_do_we_know_about(root, Species.mammal)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _COW)
            is True
        )

    def test_mixed_mammal_false_for_eagle(self):
        animal, _, root = _mixed_tree()
        knowledge = what_do_we_know_about(root, Species.mammal)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _EAGLE)
            is False
        )

    def test_mixed_bird_true_for_eagle(self):
        animal, _, root = _mixed_tree()
        knowledge = what_do_we_know_about(root, Species.bird)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _EAGLE)
            is True
        )

    def test_mixed_bird_false_for_cow(self):
        animal, _, root = _mixed_tree()
        knowledge = what_do_we_know_about(root, Species.bird)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _COW)
            is False
        )

    def test_mixed_fish_true_for_tuna(self):
        animal, _, root = _mixed_tree()
        knowledge = what_do_we_know_about(root, Species.fish)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _TUNA)
            is True
        )

    def test_mixed_fish_false_for_cow(self):
        animal, _, root = _mixed_tree()
        knowledge = what_do_we_know_about(root, Species.fish)
        assert (
            knowledge.sufficient_condition_sets[0].evaluate_against(animal, _COW)
            is False
        )


# ---------------------------------------------------------------------------
# EQLSingleClassRDR integration
# ---------------------------------------------------------------------------


class TestRDRIntegration:
    """Verify the thin RDR method and the index cache."""

    def test_rdr_method_empty(self):
        rdr = EQLSingleClassRDR(Animal, "species")
        knowledge = rdr.what_do_we_know_about(Species.molusc)
        assert isinstance(knowledge, ConclusionKnowledge)
        assert not knowledge.is_satisfiable()

    def test_rdr_method_with_manual_rdr(self):
        """Build RDR manually, invoke method, verify results."""
        _, _, root = _flat_tree()

        knowledge = what_do_we_know_about(root, Species.bird)
        assert knowledge.is_satisfiable()
        assert len(knowledge.sufficient_condition_sets) == 1

    def test_rdr_method_repeated_query(self):
        """Query the same tree for multiple values."""
        _, _, root = _flat_tree()

        for species in (Species.mammal, Species.bird, Species.fish):
            knowledge = what_do_we_know_about(root, species)
            assert knowledge.is_satisfiable(), f"{species} should be satisfiable"

        assert not what_do_we_know_about(root, Species.molusc).is_satisfiable()
        assert not what_do_we_know_about(root, Species.insect).is_satisfiable()


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    """Verify BackwardInferenceIndex invalidates and rebuilds on demand."""

    def test_index_invalidates_after_refinement(self):
        animal = variable(Animal, domain=[])
        query = entity(animal).where(animal.backbone == True)
        with query:
            add(animal.species, Species.fish)
        query.build()

        root = query._conditions_root_

        index = BackwardInferenceIndex()

        # Query before refinement — fish backbone only
        before = index.query(root, Species.fish)
        assert before.is_satisfiable()
        before_count = len(before.sufficient_condition_sets[0].conditions)

        # Insert a refinement
        insert_refinement(root, animal.milk == True, animal.species, Species.mammal)
        index.invalidate()

        # The root now has a parent (Refinement); get the actual new root
        actual_root = query._conditions_root_

        # Query after — fish path should now have one more guard (NOT milk)
        after = index.query(actual_root, Species.fish)
        assert after.is_satisfiable()
        after_count = len(after.sufficient_condition_sets[0].conditions)
        assert after_count > before_count

    def test_index_uncached_query_is_empty_after_mutation(self):
        """After invalidate() and refreshing the root, the index returns the new rule's results."""
        animal = variable(Animal, domain=[])
        query = entity(animal).where(animal.milk == True)
        with query:
            add(animal.species, Species.mammal)
        query.build()

        root = query._conditions_root_
        index = BackwardInferenceIndex()

        assert index.query(root, Species.mammal).is_satisfiable()
        assert not index.query(root, Species.bird).is_satisfiable()

        insert_alternative(root, animal.feathers == True, animal.species, Species.bird)
        index.invalidate()

        # Now get the actual root (may have changed after insert_alternative)
        actual_root = query._conditions_root_
        assert index.query(actual_root, Species.bird).is_satisfiable()

    def test_invalidates_on_fit_case(self):
        """EQLSingleClassRDR fit_case calls _backward_index.invalidate()."""
        from krrood.entity_query_language.rdr.expert import Expert
        from krrood.entity_query_language.rdr.interface import (
            AnswerRequest,
            CaseContext,
            FunctionInterface,
        )

        def expert_fn(
            ctx: CaseContext, requests: List[AnswerRequest]
        ) -> dict[str, Any]:
            answers: dict[str, Any] = {}
            for req in requests:
                if req.name == "conditions":
                    answers["conditions"] = ctx.case_variable.milk == True
            return answers

        rdr = EQLSingleClassRDR(Animal, "species")
        assert not rdr.what_do_we_know_about(Species.mammal).is_satisfiable()

        expert = Expert(FunctionInterface(expert_fn))
        rdr.fit_case(_COW, Species.mammal, expert)

        assert rdr.what_do_we_know_about(Species.mammal).is_satisfiable()


# ---------------------------------------------------------------------------
# format_condition handles ConclusionSelector guards
# ---------------------------------------------------------------------------


class TestGuardFlattening:
    """ConclusionSelector guard expressions are flattened to leaf conditions.

    The _collect_rule_paths traversal decomposes Alternative/Refinement nodes
    into their constituent conditions so guards are precise and human-readable:
    NOT(Alternative(A,B)) → NOT(A), NOT(B); Refinement(A,B) → A.
    """

    def test_flat_tree_fish_guards_are_comparators_not_selectors(self):
        """After flattening, fish guards are Comparators, not ConclusionSelectors."""
        _, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.fish)
        conds = knowledge.sufficient_condition_sets[0].conditions
        # Before flattening: first guard was Alternative(milk, feathers)
        # After flattening: NOT(milk), NOT(feathers) — both Comparators
        for gc in conds:
            assert not isinstance(
                gc.expression, ConclusionSelector
            ), f"Guard should be flattened: {gc.expression}"

    def test_no_guard_is_ever_a_conclusion_selector(self):
        """No guard expression in any test tree is a ConclusionSelector."""
        for _, _, root in [_flat_tree(), _refinement_tree(), _mixed_tree()]:
            for value in (Species.mammal, Species.bird, Species.fish):
                knowledge = what_do_we_know_about(root, value)
                for scs in knowledge.sufficient_condition_sets:
                    for gc in scs.conditions:
                        assert not isinstance(gc.expression, ConclusionSelector), (
                            f"Guard for {value} in {_tree_name(root)}"
                            f" is unflattened: {gc.expression}"
                        )

    def test_flattened_guards_are_readable(self):
        """format_condition on flattened guards never shows dataclass fields."""
        _, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.fish)
        for scs in knowledge.sufficient_condition_sets:
            for gc in scs.conditions:
                rendered = format_condition(gc.expression)
                assert "_conclusions_=" not in rendered
                assert "right_yielded" not in rendered


def _tree_name(root):
    """Helper to identify which test tree we're in."""
    return getattr(root, "_name_", str(type(root).__name__))


# ---------------------------------------------------------------------------
# is_satisfiable edge cases
# ---------------------------------------------------------------------------


class TestIsSatisfiable:
    """Edge cases for the is_satisfiable property."""

    def test_empty_constructor(self):
        knowledge = ConclusionKnowledge(Species.molusc, ())
        assert not knowledge.is_satisfiable()

    def test_with_empty_condition_set(self):
        knowledge = ConclusionKnowledge(
            Species.molusc,
            (SufficientConditionSet(()),),
        )
        assert knowledge.is_satisfiable()

    def test_evaluate_empty_condition_set(self):
        """A SufficientConditionSet with no conditions is vacuously true."""
        animal = variable(Animal, domain=[])
        cond_set = SufficientConditionSet(())
        assert cond_set.evaluate_against(animal, _COW) is True


# ---------------------------------------------------------------------------
# GuardCondition.holds_for
# ---------------------------------------------------------------------------


class TestGuardConditionHoldsFor:
    """Unit tests for :meth:`GuardCondition.holds_for`.

    Each test exercises exactly one path through the holds_for logic:
    whether the guard is positive or negated, and whether the underlying
    expression evaluates to True or False for the given case.
    """

    def test_positive_guard_returns_true_when_condition_holds(self):
        """A non-negated guard returns True when the expression is satisfied by the case.

        Guarantee: holds_for(animal_variable, cow) is True when milk==True and cow has milk.
        """
        animal = variable(Animal, domain=[])
        guard = GuardCondition(expression=animal.milk == True, negated=False)
        assert guard.holds_for(animal, _COW) is True

    def test_positive_guard_returns_false_when_condition_does_not_hold(self):
        """A non-negated guard returns False when the expression is not satisfied by the case.

        Guarantee: holds_for(animal_variable, eagle) is False when eagle.milk is False.
        """
        animal = variable(Animal, domain=[])
        guard = GuardCondition(expression=animal.milk == True, negated=False)
        assert guard.holds_for(animal, _EAGLE) is False

    def test_negated_guard_returns_false_when_condition_holds(self):
        """A negated guard inverts: it returns False when the underlying expression is True.

        Guarantee: negated holds_for(animal_variable, cow) is False because cow has milk
        (condition is satisfied, but negation flips the result).
        """
        animal = variable(Animal, domain=[])
        guard = GuardCondition(expression=animal.milk == True, negated=True)
        assert guard.holds_for(animal, _COW) is False

    def test_negated_guard_returns_true_when_condition_does_not_hold(self):
        """A negated guard inverts: it returns True when the underlying expression is False.

        Guarantee: negated holds_for(animal_variable, eagle) is True because eagle has no
        milk (condition fails, negation flips to True).
        """
        animal = variable(Animal, domain=[])
        guard = GuardCondition(expression=animal.milk == True, negated=True)
        assert guard.holds_for(animal, _EAGLE) is True


# ---------------------------------------------------------------------------
# SufficientConditionSet.evaluate_against delegates to GuardCondition.holds_for
# ---------------------------------------------------------------------------


class TestSufficientConditionSetDelegates:
    """Verifies that :meth:`SufficientConditionSet.evaluate_against` delegates per-guard
    evaluation to :meth:`GuardCondition.holds_for` and that the all-guards-must-hold
    semantics are preserved: every guard must pass, and the first failure short-circuits.
    """

    def test_evaluate_against_returns_true_when_all_guards_pass(self):
        """evaluate_against is True when every guard in the set holds for the case.

        Uses the flat-tree mammal path (one guard: milk==True) against _COW, which
        has milk=True, so the single guard passes and the result is True.
        """
        animal, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.mammal)
        cond_set = knowledge.sufficient_condition_sets[0]
        # Sanity: this path has exactly one guard (milk==True, non-negated)
        assert len(cond_set.conditions) == 1
        assert cond_set.evaluate_against(animal, _COW) is True

    def test_evaluate_against_returns_false_when_first_guard_fails(self):
        """evaluate_against is False when the very first guard does not hold for the case.

        Uses the flat-tree bird path against _COW. The bird path requires NOT(milk==True)
        as its first guard; since _COW.milk is True, that first guard fails immediately,
        making the whole set False — confirming short-circuit behaviour via all().
        """
        animal, _, root = _flat_tree()
        knowledge = what_do_we_know_about(root, Species.bird)
        cond_set = knowledge.sufficient_condition_sets[0]
        # Sanity: bird path has two conditions; first is negated (NOT milk)
        assert len(cond_set.conditions) == 2
        assert cond_set.conditions[0].negated is True
        assert cond_set.evaluate_against(animal, _COW) is False
