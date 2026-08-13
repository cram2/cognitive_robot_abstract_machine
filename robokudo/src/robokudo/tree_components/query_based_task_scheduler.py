"""Query-based task scheduling for behavior trees.

This module provides a task scheduler that uses queries in the CAS (Common Analysis Structure)
to determine which perception subtree to execute. It allows dynamic selection of perception
pipelines based on the current query state.

Original implementation by Malte Huerkamp.
"""

from py_trees.composites import Composite, Sequence
from robokudo_msgs.action import Query
from typing_extensions import Dict, Any, Optional, Callable

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.tree_components.task_scheduler import TaskSchedulerBase
from robokudo.utils.tree import add_child_to_parent, setup_with_descendants_on_behavior


class QueryBasedScheduler(
    TaskSchedulerBase,
    BaseAnnotator,
):
    """A Task Scheduler that checks the active Query in the CAS to infer which perception subtree to execute.
    You can apply a function to infer per use-case which perception tree you want to incorporate.

    Original implementation by Malte Huerkamp

    :ivar tasks: Dictionary mapping task identifiers to behavior trees
    :type tasks: dict
    :ivar filter_fn: Function that maps queries to task identifiers
    :type filter_fn: Callable[[QueryGoal], str]
    """

    def __init__(
        self,
        name: str = "QueryBasedScheduler",
        tasks: Optional[Dict[str, Composite]] = None,
        filter_fn: Optional[Callable[[Query.Goal], str]] = None,
    ) -> None:
        """Initialize the query-based scheduler.

        Tasks should be a dict with key='task-identifier' and value a py_trees.Behaviour)

        :param name: Name of the scheduler node
        :param tasks: Dictionary mapping task IDs to behavior trees
        :param: filter_fn a callable/function which returns a string with the identifier of the subtree to include.
        This function will receive the CASViews.QUERY and can then decide which subtree identifier is the desired one.
        """
        super().__init__(name)
        if tasks is None:
            tasks = dict()
        self.tasks: Dict[str, Composite] = tasks
        self.filter_fn: Optional[Callable[[Query.Goal], str]] = filter_fn

    def setup(self, **kwargs: Any) -> bool:
        """Set up all task trees."""
        for task in self.tasks:
            setup_with_descendants_on_behavior(self.tasks[task])
        return True

    def plan_new_job(self) -> Optional[Sequence]:
        """Plan the next job based on the current query.

        This method:
        * Gets the current query from CAS
        * Uses filter_fn to determine which task to run
        * Creates a new sequence with the selected task

        :return: New job sequence containing selected task, or None if no task found
        """
        parent = self.parent
        assert isinstance(parent, Sequence)

        new_job = Sequence("Task", memory=True)

        query = self.get_cas().get(CASViews.QUERY)

        task_key = self.filter_fn(query)
        self.logger.info(f"Running perception task: {task_key}")

        if task_key in self.tasks.keys():
            job_elements = self.tasks[task_key]
            # p = job_elements.parent
            add_child_to_parent(new_job, job_elements)
        else:
            self.logger.debug(f"Can't find task pipeline for '{task_key}'")
            return None

        return new_job
