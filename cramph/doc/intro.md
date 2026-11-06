# cramph

cramph is a generic statechart engine. A statechart is a graph of nodes, each running its own
small life cycle, whose transitions are conditions over the states of other nodes. cramph
provides

- nodes with life cycles (not started, running, paused, succeeded, failed, interrupted) and
  trinary observations,
- transition conditions that start, pause, end and reset nodes,
- compilation of a statechart into a tick that advances all nodes together,
- composite nodes such as `Sequence`, `Parallel`, `TryInOrder`, `TryAll` and `RepeatUntil`,
  and nodes that choose their child only once they run,
- generic monitors such as `CountTicks`, `CountSeconds` and `Print`,
- statecharts that take new nodes while they run,
- an executor that ticks a statechart, paced in real time, in simulation time, in lockstep
  with a physics simulation or as fast as possible,
- JSON serialization and plotting of statecharts and their execution history.

cramph knows nothing about motion control.
[giskardpy](https://cram2.github.io/cognitive_robot_abstract_machine/giskardpy) builds its
motion tasks, monitors and goals on top of it.

## Getting Started

[Statecharts](statecharts.md) explains how nodes, conditions and ticks work, and
[Running a Statechart](examples/running_a_statechart.md) builds, runs and plots a first
statechart.
