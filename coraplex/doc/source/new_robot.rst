==============================
Adding a new robot to CoraPlex
==============================

To add a new robot to CoraPlex, you need two things:
    * A robot description, expressed as an ``AbstractRobot`` subclass
    * Motions that can be executed for the robot, including any robot-specific motion overrides

--------------------------------
Robot Description
--------------------------------
The robot description defines the semantic properties of the robot that cannot be extracted from the robot's URDF
automatically. This includes the kinematic chains the robot can move (like the arms), the descriptions of the end
effectors, and the descriptions of the cameras mounted on the robot.

A robot description is an ``AbstractRobot`` subclass that composes the robot from the parts in
``semantic_digital_twin.robots.robot_parts`` and is reconstructed from a ``World`` via ``from_world``. An overview of
the available parts and how a robot is composed from them can be found in the :doc:`abstract_robot` page; the existing
``PR2`` and ``HSRB`` subclasses serve as concrete templates.

--------------------------------
Motion Execution
--------------------------------
Motions are the components that actually control the robot. The default motions in :mod:`coraplex.robot_plans.motions`
already suffice to control a new robot in simulation. If a robot needs a different implementation of a particular
motion, you can provide a robot-specific override via :class:`~coraplex.alternative_motion_mapping.AlternativeMotion`
(see the existing mappings in :mod:`coraplex.alternative_motion_mappings`). The :doc:`process_modules` page explains how
motions are turned into giskard motion state charts and executed.
