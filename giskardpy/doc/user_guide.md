# User Guide

This guide provides an overview of how to use the Giskard Python API for robot motion planning and control.

## Overview

Giskard uses a constraint-based approach to motion planning. Instead of specifying a precise trajectory, you define a set of constraints and goals that the robot should satisfy.

The core components of Giskard are:

- **Goals**: Represent the target state or behavior of the robot (e.g., reaching a Cartesian pose).
- **Motion Statecharts**: A state machine that organizes goals and transitions between them.
- **Executor**: The component responsible for solving the constraints and commanding the robot.
- **Simulation Pacer**: Controls the timing of the execution in simulation.

## Examples

The following examples demonstrate the basic usage of Giskard:

- [Basic Motion](examples/basic_motion.md): Shows how to set up a simple motion with a timer.
- [Cartesian Goals](examples/cartesian_goals.md): Demonstrates moving a robot to a specific pose in Cartesian space.

## Advanced Usage

For more complex scenarios, you can compose multiple goals into `Parallel` or `Sequence` nodes, and use custom `Monitors` to trigger transitions in the `MotionStatechart`.
