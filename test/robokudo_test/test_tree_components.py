from typing import Any

import py_trees
import pytest
from py_trees.blackboard import Blackboard
from py_trees.composites import Sequence

from robokudo.cas import CASViews
from robokudo.identifier import BBIdentifier
from robokudo.pipeline import Pipeline
from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
from robokudo.tree_components.task_scheduler import (
    TaskSchedulerBase,
    IterativeTaskScheduler,
)
from robokudo_msgs.action import Query


class DummyBehaviour(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self._setup = False

    def setup(self, **kwargs: Any):
        self._setup = True

    def update(self):
        return py_trees.common.Status.SUCCESS


class TestTreeComponents:
    def test_iterative_task_scheduler(self):
        leaf_behaviour1 = DummyBehaviour("Behaviour1")
        par1 = py_trees.composites.Parallel(
            "Par1", policy=py_trees.common.ParallelPolicy.SuccessOnAll()
        )
        par1.add_child(leaf_behaviour1)

        leaf_behaviour2 = DummyBehaviour("Behaviour2")
        par2 = py_trees.composites.Parallel(
            "Par2", policy=py_trees.common.ParallelPolicy.SuccessOnAll()
        )
        par2.add_child(leaf_behaviour2)

        leaf_behaviour3 = DummyBehaviour("Behaviour3")
        par3 = py_trees.composites.Parallel(
            "Par3", policy=py_trees.common.ParallelPolicy.SuccessOnAll()
        )
        par3.add_child(leaf_behaviour3)

        scheduler = IterativeTaskScheduler(tree_list=[par1, par2, par3])
        pipe = Pipeline("Seq", memory=True, children=[scheduler])

        scheduler.initialise()

        scheduler.setup(timeout=1.0)
        assert all(
            [leaf_behaviour1._setup, leaf_behaviour2._setup, leaf_behaviour3._setup]
        )

        for job in [par1, par2, par3, par1]:
            status = scheduler.update()
            assert status == py_trees.common.Status.SUCCESS
            task = scheduler.parent.children[1]
            assert task.children[0].name == job.name

    def test_query_based_scheduler(self):
        leaf_behaviour1 = DummyBehaviour("Behaviour1")
        par1 = py_trees.composites.Parallel(
            "Par1", policy=py_trees.common.ParallelPolicy.SuccessOnAll()
        )
        par1.add_child(leaf_behaviour1)

        leaf_behaviour2 = DummyBehaviour("Behaviour2")
        par2 = py_trees.composites.Parallel(
            "Par2", policy=py_trees.common.ParallelPolicy.SuccessOnAll()
        )
        par2.add_child(leaf_behaviour2)

        scheduler = QueryBasedScheduler(
            tasks={
                "detect": par1,
                "track": par2,
            },
            filter_fn=lambda query: query["query"],
        )
        pipe = Pipeline("Seq", memory=True, children=[scheduler])

        scheduler.initialise()

        scheduler.setup(timeout=1.0)
        assert all([leaf_behaviour1._setup, leaf_behaviour2._setup])

        query_goal = {"query": "detect"}
        scheduler.get_cas().set(CASViews.QUERY, query_goal)

        status = scheduler.update()
        assert status == py_trees.common.Status.SUCCESS
        task = scheduler.parent.children[1]
        assert task.children[0].name == par1.name

        query_goal = {"query": "track"}
        scheduler.get_cas().set(CASViews.QUERY, query_goal)

        status = scheduler.update()
        assert status == py_trees.common.Status.SUCCESS
        task = scheduler.parent.children[1]
        assert task.children[0].name == par2.name

        query_goal = {"query": "non-existent-query-goal"}
        scheduler.get_cas().set(CASViews.QUERY, query_goal)

        status = scheduler.update()
        assert status == py_trees.common.Status.FAILURE
        assert Blackboard().get(BBIdentifier.BLACKBOARD_EXCEPTION_NAME)
