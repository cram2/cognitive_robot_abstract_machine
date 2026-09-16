# User Guide

This guide provides an overview of how to use the Giskard Python API for robot motion planning and control.

## Examples

The following examples demonstrate the basic usage of Giskard:

- [Basic Motion](examples/basic_motion.md): Shows how to set up a simple motion with a timer.
- [Cartesian Goals](examples/cartesian_goals.md): Demonstrates moving a robot to a specific pose in Cartesian space.

## Advanced Usage

For more complex scenarios, you can compose nodes with templates such as `Sequence`, `Parallel`, `TryInOrder` or `RepeatUntil`, and use custom `Monitors` to trigger transitions in the `MotionStatechart`. See [Motion Statecharts](motion_statecharts.md#templates) for how each template wires its children.
