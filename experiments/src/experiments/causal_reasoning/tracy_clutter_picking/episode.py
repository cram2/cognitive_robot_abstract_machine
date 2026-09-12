"""
One attempt to pick the target carton out of a layout's clutter, with MuJoCo standing in
for Tracy and the target held by contact friction alone: the same
:class:`~experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.pick_and_place_action.PickUpActionMujoco`
the Montessori and cube-stacking demos use, run as an ordinary coraplex plan.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.exceptions import MotionDidNotFinish
from coraplex.plans.factories import sequential
from coraplex.view_manager import ViewManager
from PIL import Image
from typing_extensions import Dict, List, Optional

from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutterPickOutcome,
    ClutterSceneLayout,
)
from experiments.causal_reasoning.tracy_clutter_picking.exceptions import (
    EpisodePlanningFailedError,
)
from experiments.causal_reasoning.tracy_clutter_picking.scene import MilkClutterWorld
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.equipment import (
    joint_state_of_type,
)
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.pick_and_place_action import (
    PickUpActionMujoco,
)
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.real_time_simulation import (
    RealTimeSimulation,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState

logger = logging.getLogger(__name__)


@dataclass
class BodyPositions:
    """
    Where the cartons stand at one moment of an attempt.
    """

    positions: Dict[str, np.ndarray]
    """
    Each carton's centre in the world root frame, keyed by body name.
    """

    @classmethod
    def read(cls, simulation: RealTimeSimulation, names: List[str]) -> BodyPositions:
        """
        :param simulation: The running simulation to read from.
        :param names: The bodies to read.
        :return: Their current centres, copied out of the simulator's live state so the
            reading stays put once the simulation moves on.
        """
        positions = simulation.mirror.simulator.get_bodies_positions(names).result
        return cls({name: np.array(position) for name, position in positions.items()})

    def horizontal_displacement(self, other: BodyPositions, name: str) -> float:
        """
        :param other: An earlier reading.
        :param name: The body to compare.
        :return: How far the body moved in the plane since ``other``, in metres.
        """
        delta = self.positions[name] - other.positions[name]
        return float(math.hypot(delta[0], delta[1]))

    def rise(self, other: BodyPositions, name: str) -> float:
        """
        :param other: An earlier reading.
        :param name: The body to compare.
        :return: How far the body rose since ``other``, in metres.
        """
        return float(self.positions[name][2] - other.positions[name][2])


@dataclass
class PickEpisode:
    """
    Runs one attempt on a layout and records what it did to the scene.
    """

    headless: bool = True
    """
    Whether to run without MuJoCo's viewer window.
    """

    real_time_factor: Optional[float] = None
    """
    See
    :attr:`~experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.real_time_simulation.RealTimeSimulation.real_time_factor`;
    unpaced by default, for collecting data in batch.
    """

    screenshot_directory: Optional[Path] = None
    """
    Where to save a screenshot before and after the pick, or ``None`` for none.
    """

    keep_viewer_open: bool = False
    """
    Whether to keep the viewer window open after the attempt until it is closed.
    """

    lift_threshold: float = 0.1
    """
    How far, in metres, the target has to have risen at the end of the attempt to count
    as lifted: well below the hover height a held carton returns to, well above
    anything a carton merely nudged by the fingers reaches.
    """

    settle_time: float = 1.0
    """
    Simulated seconds the scene is left to come to rest before and after the pick.
    """

    screenshot_width: int = 640
    """
    Width, in pixels, of a screenshot of the run: the widest MuJoCo renders offscreen
    without the model declaring a bigger framebuffer.
    """

    screenshot_height: int = 480
    """
    Height, in pixels, of a screenshot of the run.
    """

    scene: Optional[MilkClutterWorld] = field(init=False, default=None)
    """
    The scene of the attempt currently running.
    """

    def run(self, layout: ClutterSceneLayout) -> ClutterPickOutcome:
        """
        Build the layout's clutter, pick its target, and measure the result.

        :param layout: The layout to attempt.
        :return: What the attempt did to the scene.
        :raises EpisodePlanningFailedError: If a motion of the pick could not be
            planned; the attempt is then dropped rather than recorded.
        """
        self.scene = MilkClutterWorld(layout)
        names = [milk.name.name for milk in self.scene.milks]
        with RealTimeSimulation(
            world=self.scene.world,
            headless=self.headless,
            real_time_factor=self.real_time_factor,
        ) as simulation:
            self._hold_park(simulation)
            simulation.advance(self.settle_time)
            before = BodyPositions.read(simulation, names)
            self._screenshot(simulation, "before_pick")
            self._pick(simulation)
            simulation.advance(self.settle_time)
            after = BodyPositions.read(simulation, names)
            self._screenshot(simulation, "after_pick")
            outcome = self._outcome(before, after)
            logger.info(
                "Pick %s: target rose %.3fm.",
                "lifted" if outcome.lifted else "failed",
                outcome.lift_height,
            )
            if self.keep_viewer_open and not self.headless:
                self._wait_for_viewer(simulation)
        return outcome

    def _hold_park(self, simulation: RealTimeSimulation) -> None:
        """
        Hand every arm servo its park angle, the pose the world was built in, so the
        arms hold still while the cartons settle.

        :param simulation: The running simulation.
        """
        for arm in (self.scene.robot.left_arm, self.scene.robot.right_arm):
            park = joint_state_of_type(arm, StaticJointState.PARK)
            for connection, target in zip(park.connections, park.target_values):
                simulation.command(
                    self.scene.actuators[connection.raw_dof.name.name], target
                )

    def _pick(self, simulation: RealTimeSimulation) -> None:
        """
        Pick the target with the left arm, closing the fingers along the layout's grasp
        yaw.

        :param simulation: The running simulation.
        :raises EpisodePlanningFailedError: If a reach could not be planned.
        """
        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.TOP,
            ViewManager.get_end_effector_view(self.scene.pick_arm, self.scene.robot),
        )
        context = Context(self.scene.world, self.scene.robot, evaluate_conditions=False)
        plan = sequential(
            [
                PickUpActionMujoco(
                    object_designator=self.scene.target,
                    arm=self.scene.pick_arm,
                    grasp_description=grasp_description,
                    simulation=simulation,
                    actuators=self.scene.actuators,
                    grasp_yaw=self.scene.layout.target.yaw
                    + self.scene.layout.grasp_yaw,
                    grasp_half_width=self.scene.milk_size.x / 2,
                )
            ],
            context,
        ).plan
        try:
            plan.perform()
        except MotionDidNotFinish as error:
            raise EpisodePlanningFailedError(str(error)) from error

    def _outcome(
        self, before: BodyPositions, after: BodyPositions
    ) -> ClutterPickOutcome:
        """
        :param before: Where the cartons stood before the pick.
        :param after: Where they are after it.
        :return: The attempt's outcome.
        """
        lift_height = after.rise(before, self.scene.target.name.name)
        return ClutterPickOutcome(
            lift_height=lift_height,
            lifted=lift_height > self.lift_threshold,
            neighbour_displacements=[
                after.horizontal_displacement(before, neighbour.name.name)
                for neighbour in self.scene.neighbours
            ],
        )

    def _screenshot(self, simulation: RealTimeSimulation, label: str) -> None:
        """
        Save what the scene camera sees, if a screenshot directory was given.

        :param simulation: The running simulation.
        :param label: What the moment is called in the file name.
        """
        if self.screenshot_directory is None:
            return
        image = simulation.mirror.simulator.capture_rgb(
            camera_name=self.scene.camera_name,
            height=self.screenshot_height,
            width=self.screenshot_width,
        ).result
        self.screenshot_directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(self.screenshot_directory / f"{label}.png")

    @staticmethod
    def _wait_for_viewer(simulation: RealTimeSimulation) -> None:
        """
        Keep stepping the physics until the viewer window is closed.

        :param simulation: The running simulation.
        """
        while simulation.is_running:
            simulation.advance(0.02)
