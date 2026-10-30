# Adding a new robot to CoraPlex

To add a new robot to CoraPlex, you need two things:

* A robot description, expressed as an `AbstractRobot` subclass
* Motions that can be executed for the robot, including any robot-specific motion overrides

## Robot Description

The robot description defines the semantic properties of the robot that cannot be extracted from the robot's URDF
automatically. This includes the kinematic chains the robot can move (like the arms), the descriptions of the end
effectors, and the descriptions of the cameras mounted on the robot.

A robot description is an `AbstractRobot` subclass that composes the robot from the parts in
`semantic_digital_twin.robots.robot_parts` and is reconstructed from a `World` via `from_world`. An overview of
the available parts and how a robot is composed from them can be found in the {doc}`abstract_robot` page; the existing
`PR2` and `HSRB` subclasses serve as concrete templates.

## Motion Execution

The giskard goals and tasks in {mod}`giskardpy.motion_statechart` are what actually control the robot, and the ones
an action already builds suffice to control a new robot in simulation. The {doc}`motion_execution` page explains how an
action's goals are collected into a giskard motion state chart and executed.
