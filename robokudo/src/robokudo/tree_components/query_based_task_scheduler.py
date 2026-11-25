"""
Query-based task scheduling for behavior trees.

This module provides a task scheduler that uses queries in the CAS (Common Analysis
Structure) to determine which perception subtree to execute. It allows dynamic selection
of perception pipelines based on the current query state.

Original implementation by Malte Huerkamp.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from py_trees.composites import Composite, Sequence
from robokudo_msgs.action import Query
from typing_extensions import Any, Callable

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.tree_components.task_scheduler import TaskSchedulerBase
from robokudo.utils.tree import add_child_to_parent, setup_with_descendants_on_behavior


# %% Query scheduling
@dataclass(kw_only=True, eq=False)
class QueryBasedScheduler(
    TaskSchedulerBase,
    BaseAnnotator,
):
    """
    A Task Scheduler that checks the active Query in the CAS to infer which perception
    subtree to execute. You can apply a function to infer per use-case which perception
    tree you want to incorporate.

    Original implementation by Malte Huerkamp

    :ivar tasks: Dictionary mapping task identifiers to behavior trees
    :type tasks: dict
    :ivar filter_function: Function that maps queries to task identifiers
    :type filter_function: Callable[[QueryGoal], str]
    """

    tasks: dict[str, Composite] | None = None
    """
    Available task trees indexed by the query-selected task identifier.
    """

    filter_function: Callable[[Query.Goal], str] | None = None
    """
    Selects the task identifier for the current query.
    """

    def __post_init__(self) -> None:
        """
        Initialize annotator resources after
        :meth:`robokudo.annotators.core.BaseAnnotator.__post_init__`.
        """
        super().__post_init__()
        if self.tasks is None:
            self.tasks = dict()

    def setup(self, **kwargs: Any) -> bool:
        """
        Set up all task trees.
        """
        for task in self.tasks:
            setup_with_descendants_on_behavior(self.tasks[task])
        return True

    def plan_new_job(self) -> Sequence | None:
        """
        Select a task from the current query and construct its execution sequence.

        :return: The task sequence, or ``None`` if no task matches.
        """
        parent = self.parent
        assert isinstance(parent, Sequence)

        new_job = Sequence("Task", memory=True)

        query = self.get_cas().get(CASViews.QUERY)

        task_key = self.filter_function(query)
        self.rk_logger.info(f"Running perception task: {task_key}")

        if task_key not in self.tasks.keys():
            self.rk_logger.debug(f"Can't find task pipeline for '{task_key}'")
            return None

        job_elements = self.tasks[task_key]
        add_child_to_parent(new_job, job_elements)
        return new_job
