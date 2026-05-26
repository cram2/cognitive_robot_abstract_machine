"""
Tree initialization utilities for RoboKudo.

This module provides functionality to initialize and set up behavior trees in RoboKudo.
It handles the creation of visualization structures, blackboard initialization, and
proper tree setup.

The module is responsible for:

* Setting up the blackboard with required data structures
* Initializing visualization components when needed
* Supporting single-shot execution for testing
* Configuring proper tree structure with or without GUI
"""

from __future__ import annotations

from py_trees.blackboard import Blackboard
from py_trees.common import Duration, ParallelPolicy, OneShotPolicy
from py_trees.composites import Parallel
from py_trees.decorators import OneShot
from py_trees_ros.trees import BehaviourTree
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.vis.visualizer_manager import VisualizationManager

if TYPE_CHECKING:
    from rclpy.node import Node
    from py_trees.behaviour import Behaviour


def grow_tree(
    tree: Behaviour,
    node: Node,
    include_gui: bool = True,
    run_once: bool = False,
) -> BehaviourTree:
    """Initialize a behavior tree with all required RoboKudo data structures.

    Sets up the blackboard with visualization maps and creates proper tree structure.
    Can optionally wrap the tree in a OneShot decorator for testing or add GUI
    visualization support.

    :param tree: The behavior tree to initialize
    :param node: The ROS node to use for tree setup
    :param include_gui: Whether to include visualization support, defaults to True
    :param run_once: Whether to wrap tree in OneShot for single execution, defaults to False
    :return: The initialized behavior tree
    """
    blackboard = Blackboard()
    annotator_output_pipeline_map_visualized = AnnotatorOutputPerPipelineMap()
    annotator_output_pipeline_map_buffer = AnnotatorOutputPerPipelineMap()
    blackboard.set(
        "annotator_output_pipeline_map_visualized",
        annotator_output_pipeline_map_visualized,
    )
    blackboard.set(
        "annotator_output_pipeline_map_buffer", annotator_output_pipeline_map_buffer
    )
    blackboard.set("pipeline_trigger", False)

    # In unit tests we would like to parametrize a behaviour tree and check a single-shot of that pipeline for errors
    if run_once:
        new_tree = OneShot(
            "OneShotTree",
            tree,
            policy=OneShotPolicy.ON_COMPLETION,
        )
        tree = new_tree

    if include_gui:
        root = Parallel(
            "RootNodeWithGUI",
            policy=ParallelPolicy.SuccessOnAll(synchronise=False),
        )
        root.add_child(VisualizationManager("VisManager"))
        root.add_child(tree)

        behavior_tree = BehaviourTree(root)
    else:
        behavior_tree = BehaviourTree(tree)
    behavior_tree.setup(node=node, timeout=Duration.INFINITE)

    return behavior_tree
