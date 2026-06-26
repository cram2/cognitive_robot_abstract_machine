Motion Execution in CoraPlex
============================

Motion execution is CoraPlex's bridge from symbolic intentions to concrete robot motions. It translates the motion
designators produced by a plan into giskard motion state charts and runs them, either in simulation or on a real robot.
By keeping the "how" of actuation behind a common abstraction, plans remain robot-agnostic and execution-aware without
being robot-specific.

.. note::
    Earlier versions of CoraPlex used a ``ProcessModule``/``ProcessModuleManager`` mechanism. That layer has been
    replaced by the motion / executable / execution-environment model described here.

Motions
-------
A motion is a :class:`~coraplex.robot_plans.motions.base.BaseMotion` designator that acts as a *builder* for a single
giskard motion state chart goal. Every motion creates exactly one goal and never creates other motions or actions. The
goal is exposed through the ``motion_chart`` property, which returns a giskard :class:`~giskardpy.motion_statechart.graph_node.Task`.
Concrete motions live in :mod:`coraplex.robot_plans.motions` (for example
:class:`~coraplex.robot_plans.motions.robot_body.MoveJointsMotion`).

Executables
-----------
The motions of a plan are collected into a :class:`~coraplex.plans.executables.GiskardExecutable`. The executable holds
a mapping from the plan's motion nodes to their giskard tasks and assembles them into a single
:class:`~giskardpy.motion_statechart.motion_statechart.MotionStatechart`. While building the chart it also:

- wires the tasks into an interruptible, pausable sequence,
- adds optional pre- and post-condition monitors that gate the start and successful end of the motion,
- adds an :class:`~giskardpy.motion_statechart.goals.collision_avoidance.ExternalCollisionAvoidance` goal when
  collision avoidance is enabled.

Calling :meth:`~coraplex.plans.executables.GiskardExecutable.execute` builds the chart and runs it according to the
active execution type.

Choosing Between Simulated and Real Execution
---------------------------------------------
The execution context is selected with the :class:`~coraplex.execution_environment.ExecutionEnvironment` context
managers. Entering an environment sets the class-level ``execution_type`` and ``collision_avoidance`` on
:class:`~coraplex.plans.executables.GiskardExecutable`; leaving it restores the previous values, so environments can be
nested safely.

.. code-block:: python

    from coraplex.execution_environment import simulated_robot, real_robot

    with simulated_robot:
        plan.perform()

    with real_robot:
        plan.perform()

Four pre-built environments are provided in :mod:`coraplex.execution_environment`: ``simulated_robot``, ``real_robot``,
``semi_real_robot`` and ``no_execution``. The execution type itself is the :class:`~coraplex.datastructures.enums.ExecutionType`
enum (``SIMULATED``, ``REAL``, ``SEMI_REAL``, ``NO_EXECUTION``).

Collision avoidance can be toggled per environment:

.. code-block:: python

    with simulated_robot(collision_avoidance=True):
        plan.perform()

What happens for each execution type
------------------------------------
:meth:`~coraplex.plans.executables.GiskardExecutable.execute` dispatches on the active execution type:

- ``SIMULATED``: the chart is compiled and ticked against the world of the context until it reports an end motion. If
  it does not finish within the tick budget a :class:`~coraplex.exceptions.MotionDidNotFinish` exception is raised.
- ``REAL``: the chart is sent to giskard via the ``GiskardWrapper`` while a watcher thread monitors for interrupts.
- ``NO_EXECUTION``: the chart is built but not run, which is useful for inspecting or validating a plan.

Robot-Specific Motions
----------------------
Some robots need a different implementation of a motion. Instead of a manager hierarchy, CoraPlex uses
:class:`~coraplex.alternative_motion_mapping.AlternativeMotion`. An alternative is selected automatically by
``motion_chart`` when its generic robot type matches the current robot and its ``execution_type`` matches the active
context. Robot-specific mappings live in :mod:`coraplex.alternative_motion_mappings` (for example the HSRB, Stretch and
Tiago motion mappings).

Key takeaways
-------------
- Motions are builders for single giskard goals; plans never execute them directly.
- A :class:`~coraplex.plans.executables.GiskardExecutable` assembles the motions into one motion state chart and runs it.
- :class:`~coraplex.execution_environment.ExecutionEnvironment` context managers choose simulated, real, semi-real or
  no execution, and toggle collision avoidance.
- :class:`~coraplex.alternative_motion_mapping.AlternativeMotion` provides robot-specific motion overrides.
