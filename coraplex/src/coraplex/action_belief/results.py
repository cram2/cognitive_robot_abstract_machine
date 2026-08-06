from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Any, Dict, List, Tuple, Type

from coraplex.robot_plans.actions.base import ActionDescription


@dataclass
class ActionBeliefResult:
    """
    Outcome of ranking one grounding of a registered action type against its
    :class:`~coraplex.action_belief.action_belief_space.ActionBeliefSpace`.

    .. note::
        Excluded from ORM generation for now: :attr:`chosen`, :attr:`rejected_examples`
        and :attr:`ranked_candidates` are ``Dict[str, Any]``-keyed, which ORMatic maps
        lossily. Logging this for training needs those fields properly typed first.
    """

    action_type: Type[ActionDescription]
    """
    The action type this result was computed for.
    """

    chosen: Dict[str, Any]
    """
    The highest-posterior feasible candidate, keyed by choice-point field name.

    Empty if no candidate was feasible.
    """

    posterior: float
    """
    Renormalized probability of :attr:`chosen` among the feasible candidates. ``0.0``
    if no candidate was feasible.
    """

    candidates_enumerated: int
    """
    Total number of candidates
    :class:`~coraplex.action_belief.action_belief_query.ActionBeliefQuery` checked.
    """

    candidates_feasible: int
    """
    Number of candidates whose ``pre_condition`` held.
    """

    rejected_examples: List[Tuple[Dict[str, Any], str]]
    """
    Rejected candidates paired with why their ``pre_condition`` failed.
    """

    ranked_candidates: List[Tuple[Dict[str, Any], float]]
    """
    Every feasible candidate with its posterior, descending, for retry ordering.
    """

    def __str__(self) -> str:
        fields_repr = ", ".join(
            f"{field_name.rsplit('.', 1)[-1]}={value}"
            for field_name, value in self.chosen.items()
        )
        return (
            f"{self.action_type.__name__} grounded: {fields_repr} "
            f"({self.candidates_feasible}/{self.candidates_enumerated} feasible, "
            f"p={self.posterior:.2f})"
        )
