# Giskardpy
Giskardpy is an open source library for implementing motion control frameworks.
It uses constraint and optimization based task space control to control the whole body of mobile manipulators.

Giskardpy is part of the [CRAM monorepo](https://github.com/cram2/cognitive_robot_abstract_machine) and builds on
`krrood` and `semantic_digital_twin`, which provide the world model it controls.

## Key Features

- **Constraint-based control**: define motion goals as constraints and let a QP solver compute the joint commands.
- **Motion statecharts**: compose complex behaviors from reusable nodes with start, pause, end and reset conditions.
- **Simulation and execution**: run the same motion statechart in simulation or on a robot through different executors and pacers.
- **Semantic digital twin integration**: operate on `semantic_digital_twin` worlds, bodies and connections.
- **ROS 2 support**: optional executor and tooling to connect to a ROS 2 system.

## Installation

Giskardpy depends on other packages of the monorepo, so it is not installed on its own.
Follow the [installation instructions of the monorepo](../README.md#installation) to set up
the whole repository, including the virtual environment and the system dependencies.

ROS 2 is only required for the ROS 2 executor and for visualization.
The [monorepo README](../README.md#optional-setup-your-ros-workspace) explains how to set up a ROS workspace.

## ROS Interface

This is a pure Python library with the core functionality. The ROS 2 action server and interfaces that use
giskardpy live in `giskardpy_ros`, which is part of
[cram_ros2_packages](https://github.com/cram2/cram_ros2_packages). ROS 1 is not supported.


## How to cite
```
@phdthesis{stelter25giskard,
	author = {Simon Stelter},
	title = {A Robot-Agnostic Kinematic Control Framework: Task Composition via Motion Statecharts and Linear Model Predictive Control},
	year = {2025},
	doi = {10.26092/elib/3743},
}
```
