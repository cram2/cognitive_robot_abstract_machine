from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Iterable, List, Optional, Tuple, Type

from coraplex.action_belief.action_belief_query import ActionBeliefQuery
from coraplex.action_belief.action_belief_space import ACTION_BELIEF_SPACES
from coraplex.robot_plans.actions.base import ActionDescription

# %% diagnosis


@dataclass
class InterventionResult:
    """
    Outcome of :func:`diagnose`: which single registered choice-point change, if any,
    flips a failing action's ``pre_condition`` from fail to pass.
    """

    action_type: Type[ActionDescription]
    """
    The action type that was diagnosed.
    """

    fix: Optional[Tuple[str, Any, Any]]
    """
    ``(field_name, old_value, new_value)`` of the first perturbation that flipped
    ``pre_condition`` from fail to pass.

    ``None`` if no single-field change did, or
    if :attr:`action_type` is not registered in
    :data:`~coraplex.action_belief.action_belief_space.ACTION_BELIEF_SPACES`.
    """

    other_fixes_count: int = 0
    """
    How many *other* perturbations, beyond :attr:`fix`, also flipped fail to pass.
    """

    trials: List[Tuple[str, Any, Any, bool]] = field(default_factory=list)
    """
    Every perturbation tried, as ``(field_name, old_value, new_value, passed)``, for
    callers that need the full search rather than just the reported fix.
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
        field_name, old_value, new_value = self.fix
        suffix = (
            f" (+{self.other_fixes_count} other fixes found)"
            if self.other_fixes_count
            else ""
        )
        return (
            f"{name}: {field_name} {old_value} -> {new_value} fixes it "
            f"(pre_condition: fail -> pass){suffix}"
        )


def diagnose(action: ActionDescription) -> InterventionResult:
    """
    Perturb one registered bucket-A choice-point field of ``action`` at a time, re-
    checking ``pre_condition`` on a fresh deep copy of the world after each change, and
    report the first perturbation that flips fail to pass.

    Bucket-B (``Location``-backed) choice points are not perturbed: an already grounded
    action only carries the one concrete pose it was built with, not the ``Location``
    that produced it, so there is no alternative-pose domain left to retry here.

    :param action: The already-grounded, already-failing action to diagnose.
    :return: The first fix found (if any), plus every perturbation tried.
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

    trials: List[Tuple[str, Any, Any, bool]] = []
    fixes: List[Tuple[str, Any, Any]] = []
    for point in ACTION_BELIEF_SPACES[action_type].choice_points:
        if point.bucket != "A":
            continue
        current_value = current_values[point.field_name]
        for candidate_value in point.domain:
            if candidate_value == current_value:
                continue
            perturbed = query.perturbed_action(
                action, point.field_name, candidate_value
            )
            passed = query.predict_feasible(perturbed)
            trials.append((point.field_name, current_value, candidate_value, passed))
            if passed:
                fixes.append((point.field_name, current_value, candidate_value))

    fix = fixes[0] if fixes else None
    return InterventionResult(
        action_type=action_type,
        fix=fix,
        other_fixes_count=max(len(fixes) - 1, 0),
        trials=trials,
    )


def format_suggestions(results: Iterable[InterventionResult]) -> str:
    """
    :return: One line per :class:`InterventionResult`, joined by newlines.
    """
    return "\n".join(str(result) for result in results)
