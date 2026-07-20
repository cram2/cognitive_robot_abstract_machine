"""
Execution of one reproducible trial of the tool-based action experiment.

This module is run as its own process per trial (see
:mod:`experiments.tool_based_actions.experiment.run_experiment`), so a crashing or
hanging simulation never takes the campaign down with it.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

from krrood.entity_query_language.factories import a, variable
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Tool
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from typing_extensions import List, Optional

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.locations.base import DeferredLocation, Location
from coraplex.locations.costmaps import OccupancyCostmap, RingCostmap
from coraplex.view_manager import ViewManager
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.testing import setup_world, start_visualization

from experiments.tool_based_actions.experiment.configuration import (
    ExperimentConfiguration,
    ToolBasedTask,
    TrialSpecification,
)
from experiments.tool_based_actions.experiment.results import (
    ResultRecorder,
    TargetResult,
)
from experiments.tool_based_actions.experiment.scene import (
    SceneSampler,
    discover_spawn_surfaces,
)
from experiments.tool_based_actions.experiment.task_definitions import (
    ExperimentTarget,
    ToolTaskDefinition,
    definition_for_task,
)

logger = logging.getLogger(__name__)


def tool_base_location(target_pose: Pose, context: Context, arm: Arms) -> Location:
    """
    Costmap location for base poses around a tool action target.

    Unlike :func:`coraplex.locations.factories.reachability_location` this validates
    no grasp pose sequence: tool actions act on the target with the mounted tool
    instead of grasping it, so base candidates only need to be collision-free and
    within arm's reach of the target.

    :param target_pose: The pose of the target to act on.
    :param context: The plan context providing world and robot.
    :param arm: The arm the tool is mounted on.
    :return: A location yielding collision-free base poses around the target.
    """
    ring = RingCostmap(
        resolution=0.02,
        width=200,
        height=200,
        std=15,
        distance=ViewManager.get_arm_view(arm, context.robot).approximate_length()
        * 0.66,
        world=context.world,
        origin=target_pose,
    )
    costmap = OccupancyCostmap.default_map(context, target_pose) & ring
    return Location(context, target_pose, costmap, [])


@dataclass
class TrialRunner:
    """
    Runs one trial: spawn the seeded scene, act on every target, record results.
    """

    specification: TrialSpecification
    """
    The trial to run.
    """

    configuration: ExperimentConfiguration
    """
    The campaign configuration the trial belongs to.
    """

    def run(self) -> List[TargetResult]:
        """
        Execute the trial on the simulated robot.

        :return: One result per target, in execution order.
        """
        world = setup_world()
        start_visualization(world)
        robot = PR2.from_world(world)
        print(world.root)
        context = Context(world=world, robot=robot, _debug=False, ros_node=None)
        context.evaluate_conditions = False

        definition = definition_for_task(self.specification.task)
        tool = definition.attach_tool(world, robot)
        targets = self._spawn_targets(world, definition)

        results = []
        with simulated_robot:
            for target in targets:
                results.append(self._act_on_target(context, definition, tool, target))
        return results

    def _spawn_targets(
        self, world: World, definition: ToolTaskDefinition
    ) -> List[ExperimentTarget]:
        """
        :param world: The world to spawn into.
        :param definition: The task definition spawning the targets.
        :return: The spawned targets of this trial's seeded scene.
        """
        surfaces = discover_spawn_surfaces(
            world,
            surface_names=self.configuration.surface_names,
            margin=self.configuration.surface_margin,
            height_offset=self.configuration.spawn_height_offset,
        )
        sampler = SceneSampler(
            surfaces=surfaces,
            clearance=self.configuration.target_clearance,
            seed=self.specification.seed,
        )
        count = sampler.sample_target_count(
            self.configuration.minimum_targets_per_trial,
            self.configuration.maximum_targets_per_trial,
        )
        placements = sampler.sample_placements(
            count,
            name_prefix=f"{self.specification.task.value}_{self.specification.seed}",
        )
        logger.info("Trial %s spawns %d targets.", self.specification.identifier, count)
        return [definition.spawn_target(world, placement) for placement in placements]

    def _act_on_target(
        self,
        context: Context,
        definition: ToolTaskDefinition,
        tool: Tool,
        target: ExperimentTarget,
    ) -> TargetResult:
        """
        Perform the tool action on one target and capture its outcome.

        :param context: The plan context of the trial.
        :param definition: The task definition building the action.
        :param tool: The tool attached to the robot.
        :param target: The target to act on.
        :return: The recorded outcome of the action.
        """
        plan = sequential(
            [
                SetGripperAction(definition.arm, GripperState.CLOSE),
                ParkArmsAction(Arms.BOTH),
                MoveTorsoAction(TorsoState.HIGH),
                self._navigate_to_reachable_base_pose(context, definition, target),
                definition.build_action(target, tool),
            ],
            context=context,
        ).plan

        start = time.monotonic()
        failure_reason = self._perform_and_capture_failure(plan)
        duration = time.monotonic() - start

        placement = target.placement
        return TargetResult(
            trial_identifier=self.specification.identifier,
            task=self.specification.task,
            seed=self.specification.seed,
            robot_name=self.specification.robot_name,
            environment_name=self.specification.environment_name,
            target_name=placement.name,
            target_x=placement.x,
            target_y=placement.y,
            target_yaw=placement.yaw,
            surface_name=placement.surface_name,
            success=failure_reason is None,
            duration=duration,
            failure_reason=failure_reason,
        )

    @staticmethod
    def _perform_and_capture_failure(plan: Plan) -> Optional[str]:
        """
        Perform the plan, translating any raised error into a recordable reason.

        The experiment must observe failures instead of aborting on them, so this is the
        one place a broad exception handler is justified.

        :param plan: The plan to perform.
        :return: None on success, otherwise a compact failure description.
        """
        try:
            plan.perform()
        except Exception as error:
            logger.warning("Target failed: %s", error, exc_info=True)
            return f"{type(error).__name__}: {error}"
        return None

    def _navigate_to_reachable_base_pose(
        self,
        context: Context,
        definition: ToolTaskDefinition,
        target: ExperimentTarget,
    ):
        """
        :param context: The plan context of the trial.
        :param definition: The task definition providing the acting arm.
        :param target: The target to approach.
        :return: An underspecified navigation whose base pose is drawn from a
            costmap location around the target.
        """
        return a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=DeferredLocation(
                    lambda: tool_base_location(target.pose, context, definition.arm)
                ),
            ),
            keep_joint_states=True,
        )


def run_trial(
    specification: TrialSpecification, configuration: ExperimentConfiguration
) -> List[TargetResult]:
    """
    Run one trial and append its results to the campaign's results file.

    :param specification: The trial to run.
    :param configuration: The campaign configuration.
    :return: The recorded results of the trial.
    """
    recorder = ResultRecorder(configuration.results_file)
    results = TrialRunner(
        specification=specification, configuration=configuration
    ).run()
    for result in results:
        recorder.record(result)
    return results


def main() -> None:
    """
    Command line entry point running exactly one trial.
    """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", required=True, choices=[task.value for task in ToolBasedTask]
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--configuration-json", required=True)
    arguments = parser.parse_args()

    configuration = ExperimentConfiguration.from_json(arguments.configuration_json)
    specification = TrialSpecification(
        task=ToolBasedTask(arguments.task),
        seed=arguments.seed,
        robot_name=configuration.robot_name,
        environment_name=configuration.environment_name,
    )
    results = run_trial(specification, configuration)
    successes = sum(1 for result in results if result.success)
    logger.info(
        "Trial %s finished: %d/%d targets succeeded.",
        specification.identifier,
        successes,
        len(results),
    )


if __name__ == "__main__":
    main()
