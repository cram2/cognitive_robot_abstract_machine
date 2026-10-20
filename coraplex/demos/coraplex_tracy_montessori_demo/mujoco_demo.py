"""
The montessori demo carried out against MuJoCo physics, in a MuJoCo viewer. Run it from
this folder::

    python mujoco_demo.py

The physics is stepped from the control loop rather than from a thread of its own: every
tick of the controller writes its command into the world, the servos take it as their set
point, and the physics advances one cycle before the next tick reads the world back. That
is what :class:`~coraplex.execution_environment.PhysicallySimulatedRobot` has every
motion of a plan do.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import mujoco

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import PhysicallySimulatedRobot
from demo import DemoRunner, ShapeSortingPlan
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.robots.tracy import Tracy


@dataclass
class MujocoRunner(DemoRunner):
    """
    Carries the sorting plan out against MuJoCo physics.
    """

    headless: bool = False
    """
    Whether the physics runs without a viewer.
    """

    step_size: float = 1e-3
    """
    How long one physics step lasts, in seconds.
    """

    friction_cone: mujoco.mjtCone = mujoco.mjtCone.mjCONE_ELLIPTIC
    """
    The friction cone the contacts use: the elliptic cone is the exact one, where the
    default pyramid only approximates it and lets a squeezed piece creep between the pads.
    """

    impedance_ratio: float = 10.0
    """
    How much harder sliding is than pressing in: MuJoCo's own guidance for grasping with
    an elliptic cone; at the default of 1 a held piece slides out under its own weight
    despite a firm squeeze.
    """

    no_slip_iterations: int = 10
    """
    Passes spent removing the slip soft contacts leave, so a held piece stays where the
    pads hold it.
    """

    settling_duration: timedelta = timedelta(seconds=1)
    """
    How long the pieces are left to come to rest on the table before the robot reaches
    for them.
    """

    def run(self) -> None:
        world = self.scene.build_world()
        [tracy] = world.get_semantic_annotations_by_type(Tracy)
        plan = ShapeSortingPlan(scene=self.scene).build(
            Context(world=world, robot=tracy, evaluate_conditions=False)
        )
        simulation = MujocoSim(
            world=world,
            headless=self.headless,
            step_size=self.step_size,
            cone=self.friction_cone,
            impratio=self.impedance_ratio,
            noslip_iterations=self.no_slip_iterations,
        )
        simulation.start_stepped_simulation()
        try:
            simulation.step_simulation(self.settling_duration)
            with PhysicallySimulatedRobot(simulation=simulation):
                plan.perform()
        finally:
            simulation.stop_simulation()


if __name__ == "__main__":
    MujocoRunner().run()
