from __future__ import division

from typing import Union, List

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.goals.goal import Goal
from giskardpy.motion_statechart.tasks.task import Task
from giskardpy.utils.decorators import validated_dataclass


@validated_dataclass
class SimpleSequenceGoal(Goal):
    sequence: List[List[Union[Goal, Task]]]
    """
    Takes a sequence of Tasks and Goals as input and concatenates them via start and end conditions.
    A Sequence is a list of phases where each phase represent a set of tasks and goals that should run in parallel.
    The next phase is activated when the observation state of all tasks and goals in a phase becomes true.
    :param sequence: Sequence of Tasks and Goals.
    """

    def __post_init__(self):
        if not self.sequence:
            raise Exception("Received an empty sequence")

        def concat_conditions(phase: List[Union[Goal, Task]]) -> str:
            condition = f"{phase[0].name}"
            if len(phase) == 1:
                return condition

            for item in phase[1:]:
                condition += f" and {item.name}"
            return condition

        start_condition = ""

        for phase in self.sequence:
            end_condition = concat_conditions(phase)
            for item in phase:
                if isinstance(item, Goal):
                    self.add_goal(item)
                if isinstance(item, Task):
                    self.add_task(item)
                else:
                    raise Exception(f"{item} is not a Goal or Task")

                item.start_condition = start_condition
                item.end_condition = end_condition

            start_condition = end_condition

        self.observation_expression = cas.logic_and(
            *[item.observation_expression for item in self.sequence[-1]]
        )
