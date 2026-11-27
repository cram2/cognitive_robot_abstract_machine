"""
Tests that a rule tree concludes once for every value of the variables its conclusions
are about, and that an alternative concludes only for the values no earlier branch holds
for.
"""

from __future__ import annotations

from krrood.entity_query_language.rules.conclusion_selector import ValueCombination
from krrood.entity_query_language.factories import (
    add,
    alternative,
    an,
    deduced_variable,
    entity,
    inference,
    refinement,
    variable,
)

from ...dataset.rule_tree_domains import (
    Child,
    ChildLabel,
    Household,
    HouseholdChildLabel,
    Verdict,
    Vote,
    Voter,
)


# %% helpers


def labels_by_child_name(labels: list[ChildLabel]) -> list[tuple[str, str]]:
    """
    :param labels: The labels a rule tree concluded.
    :return: The name of each label's child with its text, sorted.
    """
    return sorted((label.child.name, label.text) for label in labels)


# %% alternatives


def test_an_else_concludes_for_children_an_earlier_branch_failed_on_the_household():
    """
    The final alternative labels a child even when the branch before it failed on the
    household before reaching the child.
    """
    household = variable(Household, domain=[Household(size=2)])
    young, old = Child(name="young", age=5), Child(name="old", age=19)
    child = variable(Child, domain=[young, old])
    label = deduced_variable(ChildLabel)
    query = an(entity(label).where(household.size == 1, child.age > 50))
    with query:
        add(label, inference(ChildLabel)(child=child, text="retired"))
        with alternative(household.size >= 3, child.age >= 18):
            add(label, inference(ChildLabel)(child=child, text="large family adult"))
        with alternative():
            add(label, inference(ChildLabel)(child=child, text="other"))

    assert labels_by_child_name(query.tolist()) == [
        ("old", "other"),
        ("young", "other"),
    ]


def test_a_default_concludes_only_for_voters_no_vote_approves_for():
    """
    A vote that fails the root's join does not send its voter to the default when
    another vote approves for them.
    """
    alice, bob = Voter(name="alice"), Voter(name="bob")
    voter = variable(Voter, domain=[alice, bob])
    vote = variable(
        Vote,
        domain=[Vote(voter=alice, approves=True), Vote(voter=bob, approves=False)],
    )
    verdict = deduced_variable(Verdict)
    query = an(entity(verdict).where(vote.voter == voter, vote.approves == True))
    with query:
        add(verdict, inference(Verdict)(voter=voter, approves=True))
        with alternative():
            add(verdict, inference(Verdict)(voter=voter, approves=False))

    assert sorted(
        (verdict.voter.name, verdict.approves) for verdict in query.tolist()
    ) == [("alice", True), ("bob", False)]


def test_an_else_is_an_alternative_only_for_the_variables_every_conclusion_uses():
    """
    A variable only the first branch's conclusion uses does not tell apart what the else
    concludes for: a child the first branch labelled is not labelled by the else.
    """
    household = variable(Household, domain=[Household(size=1), Household(size=5)])
    young, old = Child(name="young", age=5), Child(name="old", age=19)
    child = variable(Child, domain=[young, old])
    label = deduced_variable(ChildLabel)
    query = an(entity(label).where(household.size >= 3, child.age >= 18))
    with query:
        add(
            label,
            inference(HouseholdChildLabel)(
                child=child, household=household, text="adult"
            ),
        )
        with alternative():
            add(label, inference(ChildLabel)(child=child, text="other"))

    assert labels_by_child_name(query.tolist()) == [
        ("old", "adult"),
        ("young", "other"),
    ]


# %% combinations of values


def test_combinations_of_equal_but_distinct_values_differ():
    """
    Two combinations are the same only when they hold the very same objects.
    """
    value = [1]

    assert ValueCombination(values=(value,)) == ValueCombination(values=(value,))
    assert ValueCombination(values=([1],)) != ValueCombination(values=([1],))


def test_combinations_of_different_lengths_differ():
    """
    A combination is not the same as a longer one that starts with its values.
    """
    value = [1]

    assert ValueCombination(values=(value,)) != ValueCombination(values=(value, value))


# %% variables only the conclusions use


def test_a_child_only_the_conclusions_use_is_labelled_for_every_value():
    """
    A variable no condition uses is enumerated by the conclusion that uses it.
    """
    household = variable(Household, domain=[Household(size=10)])
    child = variable(
        Child, domain=[Child(name=name, age=1) for name in ("a", "b", "c")]
    )
    label = deduced_variable(ChildLabel)
    query = an(entity(label).where(household.size > 5))
    with query:
        add(label, inference(ChildLabel)(child=child, text="large"))

    assert labels_by_child_name(query.tolist()) == [
        ("a", "large"),
        ("b", "large"),
        ("c", "large"),
    ]


def test_a_child_only_the_conclusions_of_two_alternatives_use_is_labelled_once_each():
    """
    A variable only the conclusions of two alternatives use reaches the first branch
    that holds, once per value.
    """
    household = variable(Household, domain=[Household(size=10)])
    child = variable(
        Child, domain=[Child(name=name, age=1) for name in ("a", "b", "c")]
    )
    label = deduced_variable(ChildLabel)
    query = an(entity(label).where(household.size > 5))
    with query:
        add(label, inference(ChildLabel)(child=child, text="large"))
        with alternative():
            add(label, inference(ChildLabel)(child=child, text="small"))

    assert labels_by_child_name(query.tolist()) == [
        ("a", "large"),
        ("b", "large"),
        ("c", "large"),
    ]


# %% empty domains


def test_an_else_if_chain_over_no_children_concludes_nothing():
    """
    An else-if chain over an empty domain has nothing to conclude about.
    """
    child = variable(Child, domain=[])
    label = deduced_variable(ChildLabel)
    query = an(entity(label).where(child.age > 10))
    with query:
        add(label, inference(ChildLabel)(child=child, text="old"))
        with alternative(child.age > 3):
            add(label, inference(ChildLabel)(child=child, text="middle"))
        with alternative():
            add(label, inference(ChildLabel)(child=child, text="young"))

    assert query.tolist() == []


def test_a_default_after_several_root_conditions_over_no_children_concludes_nothing():
    """
    A default after a root of several conditions over an empty domain has nothing to
    conclude about.
    """
    child = variable(Child, domain=[])
    label = deduced_variable(ChildLabel)
    query = an(entity(label).where(child.age > 10, child.name == "a"))
    with query:
        add(label, inference(ChildLabel)(child=child, text="old"))
        with alternative():
            add(label, inference(ChildLabel)(child=child, text="young"))

    assert query.tolist() == []


# %% alternatives inside a refinement


def test_every_alternative_in_a_refinement_block_is_an_else_if_of_that_refinement():
    """
    A second alternative in a refinement block extends that refinement's else-if chain.
    """
    child = variable(
        Child,
        domain=[
            Child(name=name, age=age)
            for name, age in [("a", 1), ("b", 5), ("c", 12), ("d", 30)]
        ],
    )
    label = deduced_variable(ChildLabel)
    query = an(entity(label).where(child.age >= 0))
    with query:
        add(label, inference(ChildLabel)(child=child, text="default"))
        with refinement(child.age < 3):
            add(label, inference(ChildLabel)(child=child, text="baby"))
            with alternative(child.age < 10):
                add(label, inference(ChildLabel)(child=child, text="kid"))
            with alternative(child.age < 18):
                add(label, inference(ChildLabel)(child=child, text="teen"))

    assert labels_by_child_name(query.tolist()) == [
        ("a", "baby"),
        ("b", "kid"),
        ("c", "teen"),
        ("d", "default"),
    ]


def test_an_alternative_after_a_nested_refinement_is_an_else_if_of_the_outer_one():
    """
    An alternative written after a nested refinement is an else-if of the refinement
    whose block it is in.
    """
    child = variable(
        Child,
        domain=[
            Child(name=name, age=age)
            for name, age in [("a", 1), ("b", 2), ("c", 5), ("d", 30)]
        ],
    )
    label = deduced_variable(ChildLabel)
    query = an(entity(label).where(child.age >= 0))
    with query:
        add(label, inference(ChildLabel)(child=child, text="default"))
        with refinement(child.age < 3):
            add(label, inference(ChildLabel)(child=child, text="baby"))
            with refinement(child.age < 2):
                add(label, inference(ChildLabel)(child=child, text="newborn"))
            with alternative(child.age < 10):
                add(label, inference(ChildLabel)(child=child, text="kid"))

    assert labels_by_child_name(query.tolist()) == [
        ("a", "newborn"),
        ("b", "baby"),
        ("c", "kid"),
        ("d", "default"),
    ]
