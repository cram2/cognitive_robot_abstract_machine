import dataclasses
import sys

from coraplex.action_belief.action_belief_space import ACTION_BELIEF_SPACES
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    CuttingTechnique,
    SlicingPriority,
    VerticalAlignment,
    WipingTechnique,
)
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

# %% registry content


def test_every_planned_action_type_is_registered():
    assert set(ACTION_BELIEF_SPACES.keys()) == {
        PickUpAction,
        ReachAction,
        GraspingAction,
        PlaceAction,
        OpenAction,
        CloseAction,
        NavigateAction,
        CuttingAction,
        WipingAction,
    }


def test_pick_up_action_choice_points():
    space = ACTION_BELIEF_SPACES[PickUpAction]
    assert space.action_type is PickUpAction
    assert [point.field_name for point in space.choice_points] == [
        "arm",
        "grasp_description.approach_direction",
        "grasp_description.vertical_alignment",
    ]
    assert all(point.bucket == "A" for point in space.choice_points)
    domain_by_field = {point.field_name: point.domain for point in space.choice_points}
    assert domain_by_field["arm"] == [Arms.LEFT, Arms.RIGHT]
    assert domain_by_field["grasp_description.approach_direction"] == list(
        ApproachDirection
    )
    assert domain_by_field["grasp_description.vertical_alignment"] == list(
        VerticalAlignment
    )


def test_reach_action_choice_points():
    space = ACTION_BELIEF_SPACES[ReachAction]
    assert [point.field_name for point in space.choice_points] == [
        "arm",
        "grasp_description.approach_direction",
        "grasp_description.vertical_alignment",
        "reverse_reach_order",
    ]
    reverse_reach_order_point = space.choice_points[-1]
    assert reverse_reach_order_point.bucket == "A"
    assert reverse_reach_order_point.domain == [False, True]


def test_grasping_action_choice_points():
    space = ACTION_BELIEF_SPACES[GraspingAction]
    assert [point.field_name for point in space.choice_points] == [
        "arm",
        "grasp_description.approach_direction",
        "grasp_description.vertical_alignment",
    ]


def test_place_action_choice_points():
    space = ACTION_BELIEF_SPACES[PlaceAction]
    assert [point.field_name for point in space.choice_points] == [
        "arm",
        "target_location",
    ]
    target_location_point = space.choice_points[-1]
    assert target_location_point.bucket == "B"
    assert target_location_point.location_field == "target_location"
    assert target_location_point.domain is None


def test_open_and_close_action_choice_points():
    for action_type in (OpenAction, CloseAction):
        space = ACTION_BELIEF_SPACES[action_type]
        assert [point.field_name for point in space.choice_points] == ["arm"]
        assert space.choice_points[0].domain == [Arms.LEFT, Arms.RIGHT]


def test_navigate_action_choice_points():
    space = ACTION_BELIEF_SPACES[NavigateAction]
    assert [point.field_name for point in space.choice_points] == [
        "keep_joint_states",
        "target_location",
    ]
    assert space.choice_points[0].bucket == "A"
    assert space.choice_points[0].domain == [False, True]
    assert space.choice_points[1].bucket == "B"
    assert space.choice_points[1].location_field == "target_location"


def test_cutting_action_choice_points():
    space = ACTION_BELIEF_SPACES[CuttingAction]
    domain_by_field = {point.field_name: point.domain for point in space.choice_points}
    assert domain_by_field["technique"] == list(CuttingTechnique)
    assert domain_by_field["slicing_priority"] == list(SlicingPriority)


def test_wiping_action_choice_points():
    space = ACTION_BELIEF_SPACES[WipingAction]
    assert [point.field_name for point in space.choice_points] == ["technique"]
    assert space.choice_points[0].domain == list(WipingTechnique)


def test_no_registered_space_declares_a_prior_yet():
    assert all(space.prior is None for space in ACTION_BELIEF_SPACES.values())


# %% field-name safety net


def _unwrap_optional(annotation):
    origin_args = getattr(annotation, "__args__", None)
    if origin_args and type(None) in origin_args:
        remaining = [arg for arg in origin_args if arg is not type(None)]
        return remaining[0] if len(remaining) == 1 else annotation
    return annotation


def _resolve_field_type(dataclass_type, field_name: str):
    """
    Resolve one field's annotation to a type object, evaluating a stringified (``from
    __future__ import annotations``) annotation against the field's own defining module
    rather than the whole class hierarchy, so unrelated forward references elsewhere in
    the MRO (e.g. TYPE_CHECKING-only imports) can't break it.
    """
    matching_fields = [
        f for f in dataclasses.fields(dataclass_type) if f.name == field_name
    ]
    if not matching_fields:
        return None
    annotation = matching_fields[0].type
    if isinstance(annotation, str):
        annotation = eval(annotation, vars(sys.modules[dataclass_type.__module__]))
    return _unwrap_optional(annotation)


def _field_name_resolves(action_type, dotted_field_name: str) -> bool:
    *intermediate_segments, last_segment = dotted_field_name.split(".")
    current_type = action_type
    for segment in intermediate_segments:
        if not dataclasses.is_dataclass(current_type):
            return False
        current_type = _resolve_field_type(current_type, segment)
        if current_type is None:
            return False
    if not dataclasses.is_dataclass(current_type):
        return False
    return any(f.name == last_segment for f in dataclasses.fields(current_type))


def test_all_registered_field_names_resolve():
    """
    Every ``ActionBeliefPoint.field_name`` in ``ACTION_BELIEF_SPACES`` must resolve
    against its action type's actual dataclass fields, so a renamed or removed field
    fails here instead of silently at runtime.
    """
    for action_type, space in ACTION_BELIEF_SPACES.items():
        for choice_point in space.choice_points:
            assert _field_name_resolves(action_type, choice_point.field_name), (
                f"{action_type.__name__}.{choice_point.field_name} does not "
                f"resolve - field renamed or removed?"
            )
