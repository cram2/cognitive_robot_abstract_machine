# cramph

cramph is a generic statechart engine. It provides

- nodes with life cycles (not started, running, paused, done, ...) and observation states,
- transition conditions that start, pause, end and reset nodes based on the states of other nodes,
- compilation of a statechart into a tick function that advances all nodes by one step,
- composite nodes such as sequences, parallels and retries,
- JSON serialization and plotting of statecharts and their execution history.

cramph knows nothing about motion control. [giskardpy](../giskardpy) builds its motion tasks,
monitors and goals on top of it.

## Documentation

The documentation, including how life cycles, transition conditions and ticks work, lives in
[doc](doc) and is published at https://cram2.github.io/cognitive_robot_abstract_machine/cramph.
