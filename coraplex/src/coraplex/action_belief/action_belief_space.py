from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Any, Dict, List, Literal, Optional, Type

from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    CuttingTechnique,
    SlicingPriority,
    VerticalAlignment,
    WipingTechnique,
)
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    WipingAction,
)
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import (
    GraspingAction,
    PickUpAction,
    ReachAction,
)
from coraplex.robot_plans.actions.core.placing import PlaceAction

# %% choice-point taxonomy


@dataclass
class ActionBeliefPoint:
    """
    One field on an :class:`~coraplex.robot_plans.actions.base.ActionDescription`
    subclass that :class:`~coraplex.action_belief.action_belief_query.ActionBeliefQuery`
    can enumerate, rank, or perturb.
    """

    field_name: str
    """
    Dotted path to the field on the action's dataclass, e.g.
    ``"grasp_description.approach_direction"``.
    """

    bucket: Literal["A", "B"]
    """
    ``"A"`` for a finite discrete domain, ``"B"`` for a continuous field backed by a
    :class:`~coraplex.locations.base.Location`.
    """

    domain: Optional[List[Any]] = None
    """
    Explicit enumerable values, for a bucket ``"A"`` choice point. ``None`` for
    bucket ``"B"``.
    """

    location_field: Optional[str] = None
    """
    Name of the :class:`~coraplex.locations.base.Location`-backed field, for a bucket
    ``"B"`` choice point. ``None`` for bucket ``"A"``.
    """


@dataclass
class ActionBeliefSpace:
    """
    The choice points :class:`~coraplex.action_belief.action_belief_query.ActionBeliefQuery`
    enumerates and ranks for one registered action type.
    """

    action_type: Type[ActionDescription]
    """
    The action subclass this space describes.
    """

    choice_points: List[ActionBeliefPoint]
    """
    Every ranked/perturbable field on :attr:`action_type`.
    """

    prior: Optional[Dict[str, float]] = None
    """
    Prior weight per choice-point value. ``None`` means uniform.
    """


# %% registry

ACTION_BELIEF_SPACES: Dict[Type[ActionDescription], ActionBeliefSpace] = {
    PickUpAction: ActionBeliefSpace(
        PickUpAction,
        [
            ActionBeliefPoint("arm", "A", domain=[Arms.LEFT, Arms.RIGHT]),
            ActionBeliefPoint(
                "grasp_description.approach_direction",
                "A",
                domain=list(ApproachDirection),
            ),
            ActionBeliefPoint(
                "grasp_description.vertical_alignment",
                "A",
                domain=list(VerticalAlignment),
            ),
        ],
    ),
    ReachAction: ActionBeliefSpace(
        ReachAction,
        [
            ActionBeliefPoint("arm", "A", domain=[Arms.LEFT, Arms.RIGHT]),
            ActionBeliefPoint(
                "grasp_description.approach_direction",
                "A",
                domain=list(ApproachDirection),
            ),
            ActionBeliefPoint(
                "grasp_description.vertical_alignment",
                "A",
                domain=list(VerticalAlignment),
            ),
            ActionBeliefPoint("reverse_reach_order", "A", domain=[False, True]),
        ],
    ),
    GraspingAction: ActionBeliefSpace(
        GraspingAction,
        [
            ActionBeliefPoint("arm", "A", domain=[Arms.LEFT, Arms.RIGHT]),
            ActionBeliefPoint(
                "grasp_description.approach_direction",
                "A",
                domain=list(ApproachDirection),
            ),
            ActionBeliefPoint(
                "grasp_description.vertical_alignment",
                "A",
                domain=list(VerticalAlignment),
            ),
        ],
    ),
    PlaceAction: ActionBeliefSpace(
        PlaceAction,
        [
            ActionBeliefPoint("arm", "A", domain=[Arms.LEFT, Arms.RIGHT]),
            ActionBeliefPoint("target_location", "B", location_field="target_location"),
        ],
    ),
    OpenAction: ActionBeliefSpace(
        OpenAction,
        [ActionBeliefPoint("arm", "A", domain=[Arms.LEFT, Arms.RIGHT])],
    ),
    CloseAction: ActionBeliefSpace(
        CloseAction,
        [ActionBeliefPoint("arm", "A", domain=[Arms.LEFT, Arms.RIGHT])],
    ),
    NavigateAction: ActionBeliefSpace(
        NavigateAction,
        [
            ActionBeliefPoint("keep_joint_states", "A", domain=[False, True]),
            ActionBeliefPoint("target_location", "B", location_field="target_location"),
        ],
    ),
    CuttingAction: ActionBeliefSpace(
        CuttingAction,
        [
            ActionBeliefPoint("technique", "A", domain=list(CuttingTechnique)),
            ActionBeliefPoint("slicing_priority", "A", domain=list(SlicingPriority)),
        ],
    ),
    WipingAction: ActionBeliefSpace(
        WipingAction,
        [ActionBeliefPoint("technique", "A", domain=list(WipingTechnique))],
    ),
}
