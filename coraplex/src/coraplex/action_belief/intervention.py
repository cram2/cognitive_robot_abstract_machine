from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from typing_extensions import Any, Iterable, List, Optional, Tuple, Type

from coraplex.action_belief.action_belief_query import ActionBeliefQuery
from coraplex.action_belief.action_belief_space import ACTION_BELIEF_SPACES
from coraplex.robot_plans.actions.base import ActionDescription

# %% diagnosis

_MAX_COMBINATION_SIZE = 2
"""
Largest number of registered bucket-A fields :func:`diagnose` will change together
while searching for a fix.

The search escalates from one changed field up to this many, stopping at the first
size that finds a passing combination, so a fix that only needs one field changed is
never buried under pair trials. Bucket-A domains in this registry are small enough
that even a full pairwise search is cheap (worst case today, ``ReachAction``'s four
fields, is a couple dozen throwaway-world checks, on par with one composite action's
own ranking cost -- see ``action_belief_query._EAGER_EVALUATION_CEILING``), but
searching triples and beyond for every failure isn't worth the extra cost when two
simultaneous changes already covers every case this pipeline has seen.
"""


@dataclass
class ParameterChange:
    """
    One registered choice-point field's value before and after a proposed correction.
    """

    field_name: str
    """
    The field this change applies to, as registered in
    :data:`~coraplex.action_belief.action_belief_space.ACTION_BELIEF_SPACES`.
    """

    old_value: Any
    """
    The value the failing action was actually grounded with.
    """

    new_value: Any
    """
    The value :func:`diagnose` found flips ``pre_condition`` from fail to pass, when
    applied together with every other change in the same fix.
    """

    def __str__(self) -> str:
        return f"{self.field_name} {self.old_value} -> {self.new_value}"


@dataclass
class InterventionTrial:
    """
    One perturbation :func:`diagnose` tried, and whether it passed.
    """

    changes: Tuple[ParameterChange, ...]
    """
    The field(s) changed together for this trial, one entry per field.
    """

    passed: bool
    """
    Whether ``pre_condition`` held after applying :attr:`changes`.
    """


@dataclass
class InterventionResult:
    """
    Outcome of :func:`diagnose`: the smallest combination of registered choice-point
    changes, if any, that flips a failing action's ``pre_condition`` from fail to pass.
    """

    action_type: Type[ActionDescription]
    """
    The action type that was diagnosed.
    """

    fix: Optional[Tuple[ParameterChange, ...]]
    """
    The first combination of changes, at the smallest size searched, that flipped
    ``pre_condition`` from fail to pass. A fix that only needs one field changed is a
    1-tuple; a fix that needs two fields changed together is a 2-tuple, and so on, up
    to :data:`_MAX_COMBINATION_SIZE`.

    ``None`` if no combination up to that size did, or if :attr:`action_type` is not
    registered in :data:`~coraplex.action_belief.action_belief_space.ACTION_BELIEF_SPACES`.
    """

    other_fixes_count: int = 0
    """
    How many *other* combinations, beyond :attr:`fix` and at the same size, also
    flipped fail to pass.
    """

    trials: List[InterventionTrial] = field(default_factory=list)
    """
    Every combination tried, at every size searched, for callers that need the full
    search rather than just the reported fix.
    """

    def __str__(self) -> str:
        name = self.action_type.__name__
        if self.action_type not in ACTION_BELIEF_SPACES:
            return (
                f"{name}: not a registered action type -- outside this pipeline's reach"
            )
        if self.fix is None:
            return (
                f"{name}: no bucket A change reproduces success -- likely outside "
                f"this pipeline's reach"
            )
        change_description = ", ".join(str(change) for change in self.fix)
        suffix = (
            f" (+{self.other_fixes_count} other fixes found)"
            if self.other_fixes_count
            else ""
        )
        return (
            f"{name}: {change_description} fixes it "
            f"(pre_condition: fail -> pass){suffix}"
        )


def diagnose(
    action: ActionDescription, max_combination_size: int = _MAX_COMBINATION_SIZE
) -> InterventionResult:
    """
    Search for the smallest combination of registered bucket-A choice-point fields
    that, changed together, flips ``pre_condition`` from fail to pass, re-checking on
    a fresh deep copy of the world for every combination tried.

    The search tries every single field first; only if none of those pass does it
    escalate to trying every pair of fields together, and so on up to
    ``max_combination_size``, stopping at the first size that finds a fix.

    Bucket-B (``Location``-backed) choice points are not perturbed: an already grounded
    action only carries the one concrete pose it was built with, not the ``Location``
    that produced it, so there is no alternative-pose domain left to retry here.

    :param action: The already-grounded, already-failing action to diagnose. Runs
        as a side effect of :class:`~coraplex.plans.failures.AllChildrenFailed` and
        :class:`~coraplex.exceptions.ConditionNotSatisfied`'s ``__post_init__``, so
        it relies on ``action.context``/``action.robot`` already being valid at
        that point -- every current raise site guarantees this by only ever
        setting a real, plan-attached action.
    :param max_combination_size: Largest number of fields to change together; see
        :data:`_MAX_COMBINATION_SIZE`.
    :return: The smallest fix found (if any), plus every combination tried.
    """
    action_type = type(action)
    if action_type not in ACTION_BELIEF_SPACES:
        return InterventionResult(action_type=action_type, fix=None)

    query = ActionBeliefQuery(
        action_type=action_type,
        fixed_kwargs=action.designator_parameter,
        context=action.context,
    )
    current_values = query.extract_candidate(action)
    bucket_a_points = [point for point in query.choice_points if point.bucket == "A"]

    trials: List[InterventionTrial] = []
    largest_size = min(max_combination_size, len(bucket_a_points))
    for combination_size in range(1, largest_size + 1):
        fixes_at_this_size: List[Tuple[ParameterChange, ...]] = []
        for points in itertools.combinations(bucket_a_points, combination_size):
            other_values_per_point = [
                [
                    value
                    for value in point.domain
                    if value != current_values[point.field_name]
                ]
                for point in points
            ]
            for new_values in itertools.product(*other_values_per_point):
                changes = tuple(
                    ParameterChange(
                        point.field_name, current_values[point.field_name], new_value
                    )
                    for point, new_value in zip(points, new_values)
                )
                perturbed = query.perturb(
                    action, {change.field_name: change.new_value for change in changes}
                )
                passed = query.predict_feasible(perturbed)
                trials.append(InterventionTrial(changes=changes, passed=passed))
                if passed:
                    fixes_at_this_size.append(changes)

        if fixes_at_this_size:
            return InterventionResult(
                action_type=action_type,
                fix=fixes_at_this_size[0],
                other_fixes_count=len(fixes_at_this_size) - 1,
                trials=trials,
            )

    return InterventionResult(action_type=action_type, fix=None, trials=trials)


def format_suggestions(results: Iterable[InterventionResult]) -> str:
    """
    :return: One line per :class:`InterventionResult`, joined by newlines.
    """
    return "\n".join(str(result) for result in results)
