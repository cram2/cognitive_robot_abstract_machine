"""
Tests for the per-selector branch semantics
(:mod:`krrood.entity_query_language.rdr.branch_semantics`).

Selector nodes are constructed directly rather than grown through the ``with``-context
DSL, so each selector's two halves -- sibling-guard decomposition and child-branch
enumeration -- are asserted in isolation from how a rule tree happens to be built.
"""

from dataclasses import dataclass
from enum import Enum

import pytest
from typing_extensions import Iterable, List, Tuple

from krrood.entity_query_language.factories import add, variable
from krrood.entity_query_language.rdr.branch_semantics import (
    AlternativeBranchSemantics,
    GuardedBranch,
    NextBranchSemantics,
    RefinementBranchSemantics,
    SelectorBranchSemantics,
)
from krrood.entity_query_language.rdr.backward_inference import (
    _collect_rule_paths,
    _leaf_guards,
)
from krrood.entity_query_language.rdr.guard_condition import GuardCondition
from krrood.entity_query_language.rdr.exceptions import AmbiguousBranchSemanticsError
from krrood.entity_query_language.rules.conclusion_selector import (
    Alternative,
    Next,
    Refinement,
)


@dataclass(unsafe_hash=True)
class Animal:
    """
    Minimal classification target used only by this test module.
    """

    name: str
    has_fur: bool = False
    can_fly: bool = False
    lays_eggs: bool = False


class Species(Enum):
    MAMMAL = "mammal"
    BIRD = "bird"


def _identified_guards(
    guards: Iterable[GuardCondition],
) -> List[Tuple[str, bool]]:
    """
    Render guards as ``(expression id, polarity)`` pairs for comparison.

    Symbolic expressions must be compared by :attr:`_id_`: ``==`` on them builds a
    ``Comparator`` rather than reporting equality, and every such comparison is truthy,
    so an assertion written against the expressions themselves holds for any two of
    them.

    :param guards: The guards to render.
    :return: One ``(id, negated)`` pair per guard, in order.
    """
    return [(guard.original_expression._id_, guard.negated) for guard in guards]


# %% test-only selectors, defined here so no production selector gains a rival semantics


@dataclass(eq=False)
class PrioritySelector(Alternative):
    """
    A selector refining ``Alternative`` so specificity ranking has something to rank.

    Its semantics deliberately differ from ``Alternative``'s in both halves, so a test
    can tell which one the traversal actually used.
    """


@dataclass(eq=False)
class UnclaimedSelector(Alternative):
    """
    A selector whose only semantics are two deliberately colliding ones.
    """


@dataclass
class PrioritySelectorBranchSemantics(SelectorBranchSemantics[PrioritySelector]):
    """
    Right-first branch order and a left-only sibling guard -- unlike ``Alternative``.
    """

    @classmethod
    def sibling_guards(cls, selector, negated, decompose) -> List[GuardCondition]:
        return decompose(selector.left, negated)

    @classmethod
    def branches(cls, selector, decompose) -> List[GuardedBranch]:
        return [GuardedBranch(selector.right, ()), GuardedBranch(selector.left, ())]


@dataclass
class FirstCollidingSemantics(SelectorBranchSemantics[UnclaimedSelector]):
    """
    One of two equally specific semantics for :class:`UnclaimedSelector`.
    """

    @classmethod
    def sibling_guards(cls, selector, negated, decompose) -> List[GuardCondition]:
        raise NotImplementedError

    @classmethod
    def branches(cls, selector, decompose) -> List[GuardedBranch]:
        raise NotImplementedError


@dataclass
class SecondCollidingSemantics(SelectorBranchSemantics[UnclaimedSelector]):
    """
    The other equally specific semantics for :class:`UnclaimedSelector`.
    """

    @classmethod
    def sibling_guards(cls, selector, negated, decompose) -> List[GuardCondition]:
        raise NotImplementedError

    @classmethod
    def branches(cls, selector, decompose) -> List[GuardedBranch]:
        raise NotImplementedError


# %% dispatch


def test_each_production_selector_resolves_to_its_own_semantics():
    animal = variable(Animal, domain=[])

    assert (
        SelectorBranchSemantics.most_specific_for(
            Refinement(animal.has_fur, animal.can_fly)
        )
        is RefinementBranchSemantics
    )
    assert (
        SelectorBranchSemantics.most_specific_for(
            Alternative(animal.has_fur, animal.can_fly)
        )
        is AlternativeBranchSemantics
    )
    assert (
        SelectorBranchSemantics.most_specific_for(
            Next((animal.has_fur, animal.can_fly))
        )
        is NextBranchSemantics
    )


def test_a_non_selector_expression_has_no_branch_semantics():
    """
    A leaf predicate is not a branch point, so the traversal treats it as a leaf rule.
    """
    animal = variable(Animal, domain=[])

    assert SelectorBranchSemantics.most_specific_for(animal.has_fur) is None


# %% refinement


def test_refinement_sibling_guard_is_its_base_condition_only():
    """
    The refinement branch was taken exactly when its base condition passed; the refined-
    away fallback is a separate subtree, not part of the guard.
    """
    animal = variable(Animal, domain=[])
    selector = Refinement(animal.has_fur, animal.can_fly)

    guards = RefinementBranchSemantics.sibling_guards(selector, False, _leaf_guards)

    assert len(guards) == 1
    assert guards[0].original_expression is animal.has_fur
    assert guards[0].negated is False


def test_refinement_branches_exclude_each_other():
    animal = variable(Animal, domain=[])
    selector = Refinement(animal.has_fur, animal.can_fly)

    branches = RefinementBranchSemantics.branches(selector, _leaf_guards)

    assert [branch.child_expression._id_ for branch in branches] == [
        animal.has_fur._id_,
        animal.can_fly._id_,
    ]
    # The base applies only where the refinement does not override it.
    assert _identified_guards(branches[0].entry_guards) == [(animal.can_fly._id_, True)]
    # The refinement applies only where the base it refines applied.
    assert _identified_guards(branches[1].entry_guards) == [
        (animal.has_fur._id_, False)
    ]


# %% alternative


def test_alternative_negated_sibling_guard_is_de_morgan_over_both_sides():
    animal = variable(Animal, domain=[])
    selector = Alternative(animal.has_fur, animal.lays_eggs)

    guards = AlternativeBranchSemantics.sibling_guards(selector, True, _leaf_guards)

    assert _identified_guards(guards) == [
        (animal.has_fur._id_, True),
        (animal.lays_eggs._id_, True),
    ]


def test_alternative_branches_guard_the_second_against_the_first():
    animal = variable(Animal, domain=[])
    selector = Alternative(animal.has_fur, animal.lays_eggs)

    branches = AlternativeBranchSemantics.branches(selector, _leaf_guards)

    assert [branch.child_expression._id_ for branch in branches] == [
        animal.has_fur._id_,
        animal.lays_eggs._id_,
    ]
    assert len(branches[0].entry_guards) == 0
    assert _identified_guards(branches[1].entry_guards) == [(animal.has_fur._id_, True)]


# %% next


def test_next_branches_are_independent_and_carry_no_cross_guards():
    animal = variable(Animal, domain=[])
    selector = Next((animal.has_fur, animal.can_fly, animal.lays_eggs))

    branches = NextBranchSemantics.branches(selector, _leaf_guards)

    assert [branch.child_expression._id_ for branch in branches] == [
        animal.has_fur._id_,
        animal.can_fly._id_,
        animal.lays_eggs._id_,
    ]
    assert all(len(branch.entry_guards) == 0 for branch in branches)


def test_next_sibling_guard_covers_every_child_at_the_requested_polarity():
    animal = variable(Animal, domain=[])
    selector = Next((animal.has_fur, animal.can_fly))

    guards = NextBranchSemantics.sibling_guards(selector, True, _leaf_guards)

    assert _identified_guards(guards) == [
        (animal.has_fur._id_, True),
        (animal.can_fly._id_, True),
    ]


# %% open/closed


def test_a_new_selector_is_decomposed_without_editing_the_traversal():
    """
    ``_leaf_guards`` gains a selector purely from a new semantics class.

    ``Alternative`` decomposes to both sides; :class:`PrioritySelector` decomposes to
    its left only. Getting the left-only answer proves the dispatch used the new class
    rather than a hard-coded branch.
    """
    animal = variable(Animal, domain=[])
    selector = PrioritySelector(animal.has_fur, animal.lays_eggs)

    guards = _leaf_guards(selector, negated=False)

    assert _identified_guards(guards) == [(animal.has_fur._id_, False)]


def test_a_new_selector_directs_the_path_walk_without_editing_the_traversal():
    """
    ``_collect_rule_paths`` follows the new semantics' branch order, not
    ``Alternative``'s.
    """
    animal = variable(Animal, domain=[])
    first_condition = animal.has_fur
    with first_condition:
        add(animal.species, Species.MAMMAL)
    second_condition = animal.can_fly
    with second_condition:
        add(animal.species, Species.BIRD)

    paths = list(
        _collect_rule_paths(PrioritySelector(first_condition, second_condition), [])
    )

    # Right-first, per PrioritySelectorBranchSemantics.branches.
    assert [path.add_nodes[0].unwrapped_value for path in paths] == [
        Species.BIRD,
        Species.MAMMAL,
    ]


def test_a_more_specific_selector_outranks_the_semantics_it_refines():
    animal = variable(Animal, domain=[])

    semantics = SelectorBranchSemantics.most_specific_for(
        PrioritySelector(animal.has_fur, animal.lays_eggs)
    )

    assert semantics is PrioritySelectorBranchSemantics


def test_equally_specific_semantics_collide_rather_than_picking_one_silently():
    animal = variable(Animal, domain=[])

    with pytest.raises(AmbiguousBranchSemanticsError) as collision:
        SelectorBranchSemantics.most_specific_for(
            UnclaimedSelector(animal.has_fur, animal.lays_eggs)
        )

    message = str(collision.value)
    assert FirstCollidingSemantics.__name__ in message
    assert SecondCollidingSemantics.__name__ in message
