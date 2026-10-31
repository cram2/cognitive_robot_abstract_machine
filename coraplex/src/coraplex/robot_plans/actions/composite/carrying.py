from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.plans.plan_transformation import PlanRewriting
from coraplex.robot_plans.actions.base import ParallelAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.node import StatechartNode
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from semantic_digital_twin.spatial_types.spatial_types import Vector3

# %% driving with held objects upright


@dataclass(eq=False, repr=False, kw_only=True)
class CarryAction(ParallelAction):
    """
    Drives the robot's base to a pose while every object the robot holds keeps its
    z-axis pointing up and every arm holding nothing is parked.

    An object counts as held when it hangs below a gripper's tool frame when this action
    first starts, so the steps of the arms join it then.
    """

    navigate: NavigateAction
    """
    The drive.
    """

    upright_threshold: float = AlignPlanes.threshold
    """
    Angle in radians between a held object's z-axis and the world's z-axis within which
    the object counts as upright.
    """

    accepts_children_after_compile: bool = field(init=False, default=True)

    _arm_steps_added: bool = field(default=False, init=False, repr=False)
    """
    Whether the steps of the arms joined this action, which happens once.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [self.navigate]

    def on_start(self) -> None:
        """
        Adds the steps of the arms the first time this action starts, rewritten by the
        plan transformations of :attr:`context`.
        """
        super().on_start()
        if self._arm_steps_added:
            return
        self._arm_steps_added = True
        rewriting = self.context.get_extension(PlanRewriting)
        with self.statechart.modify():
            for step in self._create_arm_steps():
                self.add_node(step)
                if rewriting is not None:
                    rewriting.rewrite(step)
            self.statechart.request_rebuild(self)

    def _create_arm_steps(self) -> list[StatechartNode]:
        """
        :return: New steps parking every arm that holds nothing and keeping every object
            the robot holds upright.
        """
        held_bodies = {arm: arm.end_effector.held_body for arm in self.robot.all_arms}
        free_arms = [arm for arm, held_body in held_bodies.items() if held_body is None]
        park_free_arms = [ParkArmsAction(arms=free_arms)] if free_arms else []
        keep_upright = [
            AlignPlanes(
                root_link=self.world.root,
                tip_link=held_body,
                tip_normal=Vector3.Z(reference_frame=held_body),
                goal_normal=Vector3.Z(reference_frame=self.world.root),
                threshold=self.upright_threshold,
            )
            for held_body in held_bodies.values()
            if held_body is not None
        ]
        return [*park_free_arms, *keep_upright]
