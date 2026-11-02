# Troubleshooting

This page contains the most common errors that could happen when using CoraPlex and how to resolve them.

## Stop Iteration

Stop iterations usually happen when you try to resolve a designator for which there is no solution or if you iterate over a
designator and reached the end of all possible solutions.

When you try to resolve a designator which has no solution the error will look something like this.

```{code-block} python
:emphasize-lines: 7

    753 def ground(self) -> Union[Object, bool]:
    754     """
    755     Return the first object from the bullet world that fits the description.
    756
    757     :return: A resolved object designator
    758     """
--> 759     return next(iter(self))

StopIteration:
```

If you encounter such an error the most likely reason is that you put the wrong arguments into your DesignatorDescription.
The best solution is to double check the input arguments of the DesignatorDescription.

## Error when performing Actions or Motions

If you get an error when trying to perform an action that complains about a missing context extension, such as
`RobotAccess`, then the context of the executor running the plan does not hold it. Add it to the context the executor
is given, and build the statechart in the executor's context. This is also explained in the
[Action Designator Example](https://cram2.github.io/cognitive_robot_abstract_machine/coraplex/notebooks/action_designator.html#Navigate-Action).

```python
from coraplex.plans.context_extensions import RobotAccess
from cramph.context import StatechartContext
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from cramph.statechart import Statechart

executor = KinematicallySimulatedPlanExecutor(
    StatechartContext.with_extensions(world, [RobotAccess(robot)])
)
statechart = Statechart(context=executor.context)
statechart.add_node(NavigateAction(target_location=pose))
executor.compile(statechart)
executor.execute()
```
