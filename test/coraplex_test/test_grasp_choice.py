"""
How a pick-up settles on the grasp it takes.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from krrood.entity_query_language.factories import evaluate_condition, variable
from coraplex.datastructures.enums import Arms
from coraplex.locations import factories
from coraplex.exceptions import GraspPoseMissing, OffersNoGrasp
from coraplex.locations.pose_validator import AreReachableBy
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose


def _reach_of(pick_up: PickUpAction) -> ReachAction:
    """
    :return: The reach the pick-up's plan performs.

    A pick-up reaches through the grasp it is built from, so the reach only appears
    once the plan below it has been expanded.
    """
    pick_up.plan_node.notify()
    [reach_node] = pick_up.plan_node.plan.get_nodes_by_designator_type(ReachAction)
    return reach_node.designator


def test_pick_up_takes_the_grasp_it_is_given(immutable_model_world):
    """
    A caller that settled on a grasp -- together with the pose the robot stands at, say
    -- has the pick-up take that one instead of ranking the object's grasps again.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    given = Pose.from_xyz_rpy(yaw=np.pi / 3, reference_frame=milk.root)

    pick_up = PickUpAction(milk, Arms.LEFT, grasp_pose=given)
    sequential([pick_up], context=context)

    assert pick_up.grasp_pose is given


def test_pick_up_takes_the_objects_first_grasp_when_given_none(immutable_model_world):
    """
    Without a grasp the pick-up takes the first one the object offers, which depends on
    the object alone.

    Nothing about the robot may enter into it, or the same description would mean
    different things depending on where the robot stands.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]

    pick_up = PickUpAction(milk, Arms.LEFT)
    sequential([pick_up], context=context)

    np.testing.assert_allclose(
        pick_up.grasp_pose.to_homogeneous_matrix().to_np(),
        next(iter(milk.grasp_poses())).to_homogeneous_matrix().to_np(),
    )


def test_the_default_grasp_does_not_depend_on_where_the_robot_stands(
    immutable_model_world,
):
    """
    Moving the robot must not change the grasp an otherwise identical description takes.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]

    from_here = PickUpAction(milk, Arms.LEFT).grasp_pose
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )
    from_there = PickUpAction(milk, Arms.LEFT).grasp_pose

    np.testing.assert_allclose(
        from_here.to_homogeneous_matrix().to_np(),
        from_there.to_homogeneous_matrix().to_np(),
    )


def test_pick_up_reaches_for_the_grasp_it_settled_on(immutable_model_world):
    """
    The grasp the pick-up chose is the one its plan reaches for, so a caller's choice
    reaches the motions rather than stopping at the action.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    given = Pose.from_xyz_rpy(yaw=np.pi / 3, reference_frame=milk.root)

    pick_up = PickUpAction(milk, Arms.LEFT, grasp_pose=given)
    sequential([pick_up], context=context)

    assert _reach_of(pick_up).grasp_pose is given


def test_pre_condition_checks_only_the_grasp_it_was_given(immutable_model_world):
    """
    A caller that named a grasp is asking for that grasp, so the pre-condition fails
    when it cannot be reached even though the object offers others that can be.

    Taking one of those instead would be performing a different action than the one
    described.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )
    end_effector = ViewManager.get_end_effector_view(Arms.LEFT, view)
    unreachable = end_effector.grasp_poses_by_distance(milk)[0]

    pick_up = PickUpAction(milk, Arms.LEFT, grasp_pose=unreachable)
    sequential([pick_up], context=context)

    assert not evaluate_condition(
        PickUpAction.pre_condition(
            pick_up.bound_variables, context, pick_up.designator_parameter
        )
    )


def test_pre_condition_judges_the_default_grasp_only(immutable_model_world):
    """
    The pre-condition asks about the grasp the action takes, which without a named one
    is the object's first.

    That the object offers others that could be reached is not the question, because the
    action would not take them.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )

    pick_up = PickUpAction(milk, Arms.LEFT)
    sequential([pick_up], context=context)
    reaches_its_grasp = AreReachableBy.for_grasp(
        pick_up.grasp_pose,
        Arms.LEFT,
        body_T_grasp=pick_up.grasp_pose,
        context=context,
    )()

    assert (
        evaluate_condition(
            PickUpAction.pre_condition(
                pick_up.bound_variables, context, pick_up.designator_parameter
            )
        )
        is reaches_its_grasp
    )


def test_pick_up_keeps_its_grasp_even_when_it_cannot_be_reached(immutable_model_world):
    """
    The action takes the grasp it resolved and no other.

    Quietly swapping in one that works would perform a different action than the one
    described, and whether the grasp can be reached is the pre-condition's question.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )

    pick_up = PickUpAction(milk, Arms.LEFT)
    sequential([pick_up], context=context)

    np.testing.assert_allclose(
        _reach_of(pick_up).grasp_pose.to_homogeneous_matrix().to_np(),
        next(iter(milk.grasp_poses())).to_homogeneous_matrix().to_np(),
    )


# %% objects that offer nothing to hold


@dataclass(eq=False)
class GraspableOfferingNoGrasp(Milk):
    """
    A graspable whose geometry admits no grip, as an annotation with an empty rim would.
    """

    def grasp_poses(self):
        return iter(())


def test_an_object_offering_no_grasp_is_refused(immutable_model_world):
    """
    An action with nothing to take hold by says so where it is described, rather than
    carrying a missing grasp into the motions.
    """
    world, view, context = immutable_model_world
    ungraspable = GraspableOfferingNoGrasp(
        root=world.get_semantic_annotations_by_type(Milk)[0].root
    )

    with pytest.raises(OffersNoGrasp):
        PickUpAction(ungraspable, Arms.LEFT)


def test_a_reach_with_neither_a_grasp_nor_an_object_is_refused(immutable_model_world):
    """
    A reach onto a bare pose has to name the grasp it aims at, since there is no object
    to take one from.
    """
    with pytest.raises(GraspPoseMissing):
        ReachAction(arm=Arms.LEFT)


# %% the grasp domain is asked at execution, not at plan build


def test_reachable_grasps_searches_at_execution_not_construction(monkeypatch):
    """
    Handing the domain to a variable must not start the search.

    The domain is wrapped rather than consumed, so a generator is what defers the work
    to the first ``next``. Searching eagerly would answer about the world the plan was
    built in rather than the one the transport runs in.
    """
    searches = []

    def record_and_refuse(*args, **kwargs):
        searches.append(True)
        raise AssertionError("the search must not run before the domain is consumed")

    monkeypatch.setattr(factories, "grasping_location", record_and_refuse)

    variable(Pose, domain=factories.ReachableGrasps(object(), object(), Arms.LEFT))

    assert searches == []


def test_reachable_grasps_sees_the_world_as_it_is_when_consumed(monkeypatch):
    """
    The grasps are the ones of the world at the moment they are asked for, not of the
    world the domain was built in.
    """
    moved = {"value": "before"}
    observed = []

    class LocationStandingIn:
        """
        A location yielding one pose, whose validator kept a grasp.
        """

        validators = [SimpleNamespace(reachable_grasp=None)]

        def __iter__(self):
            observed.append(moved["value"])
            self.validators[0].reachable_grasp = Pose.from_xyz_rpy(1.0, 0.0, 0.0)
            yield Pose.from_xyz_rpy(0.0, 0.0, 0.0)

    monkeypatch.setattr(
        factories, "grasping_location", lambda *args, **kwargs: LocationStandingIn()
    )

    grasps = factories.ReachableGrasps(object(), object(), Arms.LEFT)
    moved["value"] = "after"
    next(iter(grasps), None)

    assert observed == ["after"]


def test_reachable_grasps_yields_grasps_the_object_offers(immutable_model_world):
    """
    Every grasp handed out is one of the object's own, so a caller naming one names
    something the object actually admits.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    # Where test_pose_validator establishes the right arm can reach it.
    milk.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.7, 1.4, 1.0, reference_frame=world.root
    )

    grasp = next(iter(factories.ReachableGrasps(milk, context, Arms.RIGHT)), None)

    assert grasp is not None, "the milk is reachable, so a grasp must be found"
    assert any(
        np.allclose(
            grasp.to_homogeneous_matrix().to_np(),
            offered.to_homogeneous_matrix().to_np(),
        )
        for offered in milk.grasp_poses()
    )
