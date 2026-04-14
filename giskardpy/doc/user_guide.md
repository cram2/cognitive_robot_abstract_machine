# User Guide

This guide provides an overview of how to use the Giskard Python API for robot motion planning and control.

## Examples

The following examples demonstrate the basic usage of Giskard:

- [Basic Motion](examples/basic_motion.md): Shows how to set up a simple motion with a timer.
- [Cartesian Goals](examples/cartesian_goals.md): Demonstrates moving a robot to a specific pose in Cartesian space.

## Advanced Usage

For more complex scenarios, you can compose multiple goals into `Parallel` or `Sequence` nodes, and use custom `Monitors` to trigger transitions in the `MotionStatechart`.
