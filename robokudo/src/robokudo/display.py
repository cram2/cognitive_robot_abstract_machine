#!/usr/bin/env python
#
# License: BSD
#   https://raw.githubusercontent.com/stonier/py_trees/devel/LICENSE
#
##############################################################################
# Documentation
##############################################################################

"""
Visualization utilities for behavior trees in RoboKudo.

This module provides functionality to visualize behavior trees using dot graphs.
It is adapted from py_trees and enhanced with RoboKudo-specific features like
timing information for annotators.

The module supports:

* Generating dot graphs from behavior trees
* Rendering trees to various output formats (dot, png)
* Adding timing information for annotators
* Customizing node appearance based on node type and status
"""

##############################################################################
# Imports
##############################################################################
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor

from py_trees.common import BlackBoxLevel, VisibilityLevel, Status
from py_trees.composites import Selector, Sequence, Parallel
from py_trees.decorators import Decorator
from pydot import Node, Edge
from typing_extensions import TYPE_CHECKING, Dict, Optional, Tuple

from robokudo.pipeline import Pipeline
from robokudo.annotators.core import BaseAnnotator

if TYPE_CHECKING:
    from pydot import Dot
    from py_trees.behaviour import Behaviour


##############################################################################
# Methods
##############################################################################


def generate_pydot_graph(
    root: Behaviour,
    visibility_level: VisibilityLevel,
    collapse_decorators: bool = False,
) -> Tuple[Dot, Dict]:
    """
    Generate a pydot graph representation of a behavior tree.

    Creates a visual representation of a behavior tree using pydot. The graph includes:

    * Nodes for each behavior tree component
    * Edges showing parent-child relationships
    * Visual attributes (shape, color) based on node type and status
    * Timing information for annotator nodes

    The graph can be customized by:

    * Collapsing subtrees based on visibility level
    * Collapsing decorator nodes
    * Using different visual attributes for different node types
    * Including timing information for annotators

    :param root: The root node of the tree or subtree to visualize
    :param visibility_level: Level at which to collapse subtrees
    :param collapse_decorators: Whether to collapse decorator nodes, defaults to False
    :return: Tuple of (pydot graph object, timing information dictionary)
    """

    def get_node_attributes_legacy(
        node: Behaviour,
        visibility_level: VisibilityLevel,
    ) -> Tuple[str, str, str]:
        """
        Get visual attributes for a node in legacy mode.

        Determines the shape, color and font color for a node based on its type
        and blackbox level. This is the legacy version maintained for backwards
        compatibility.

        :param node: The behavior tree node to get attributes for
        :param visibility_level: The visibility level to use for collapsing subtrees
        :return: Tuple of (shape, fill_color, font_color)
        """
        blackbox_font_colours = {
            BlackBoxLevel.DETAIL: "dodgerblue",
            BlackBoxLevel.COMPONENT: "lawngreen",
            BlackBoxLevel.BIG_PICTURE: "white",
        }
        # if isinstance(node, Chooser):
        # attributes = ("doubleoctagon", "cyan", "black")  # octagon
        if isinstance(node, Selector):
            attributes = ("octagon", "cyan", "black")  # octagon
        elif isinstance(node, Sequence):
            attributes = ("box", "orange", "black")
        elif isinstance(node, Parallel):
            attributes = ("parallelogram", "gold", "black")
        elif isinstance(node, Decorator):
            attributes = ("ellipse", "ghostwhite", "black")
        else:
            attributes = ("ellipse", "gray", "black")
        if node.blackbox_level != BlackBoxLevel.NOT_A_BLACKBOX:
            attributes = (
                attributes[0],
                "gray20",
                blackbox_font_colours[node.blackbox_level],
            )
        return attributes

    def get_node_attributes(
        node: Behaviour,
        visibility_level: VisibilityLevel,
    ) -> Tuple[str, str, str]:
        """Get visual attributes for a node.

        Determines the shape, color and font color for a node based on its type,
        status and blackbox level. This is the current version that includes
        RoboKudo-specific node types and status colors.

        :param node: The behavior tree node to get attributes for
        :param visibility_level: The visibility level to use for collapsing subtrees
        :return: Tuple of (shape, fill_color, font_color)
        """
        blackbox_font_colours = {
            BlackBoxLevel.DETAIL: "dodgerblue",
            BlackBoxLevel.COMPONENT: "lawngreen",
            BlackBoxLevel.BIG_PICTURE: "white",
        }
        shape = ""
        # if isinstance(node, Chooser):
        # shape = "doubleoctagon"
        if isinstance(node, Selector):
            shape = "octagon"
        elif isinstance(node, Pipeline):
            shape = "box3d"
        elif isinstance(node, Sequence):
            shape = "box"
        elif isinstance(node, Parallel):
            shape = "parallelogram"
        elif isinstance(node, Decorator):
            shape = "ellipse"
        else:
            shape = "ellipse"

        if node.status == Status.SUCCESS:
            attributes = (shape, "lightgreen", "black")
        elif node.status == Status.FAILURE:
            attributes = (shape, "red", "black")
        elif node.status == Status.RUNNING:
            attributes = (shape, "lightblue", "black")
        elif node.status == Status.INVALID:
            attributes = (shape, "yellow", "black")
        else:
            attributes = ("ellipse", "gray", "black")

        if node.blackbox_level != BlackBoxLevel.NOT_A_BLACKBOX:
            attributes = (
                attributes[0],
                "gray20",
                blackbox_font_colours[node.blackbox_level],
            )
        return attributes

    fontsize = 11
    graph = Dot(graph_type="digraph")
    graph.set_name(root.name.lower().replace(" ", "_"))
    # fonts: helvetica, times-bold, arial (times-roman is the default, but this helps some viewers, like kgraphviewer)
    graph.set_graph_defaults(fontname="times-roman")
    graph.set_node_defaults(fontname="times-roman")
    graph.set_edge_defaults(fontname="times-roman")
    node_shape, node_colour, node_font_colour = get_node_attributes(
        root, visibility_level
    )
    node_root = Node(
        root.name,
        shape=node_shape,
        style="filled",
        fillcolor=node_colour,
        fontsize=fontsize,
        fontcolor=node_font_colour,
    )
    graph.add_node(node_root)
    names = [root.name]
    timing_info = {root.name: 0}

    def add_edges(
        root: Behaviour,
        root_dot_name: str,
        visibility_level: VisibilityLevel,
        collapse_decorators: bool,
        timing_info_dict: Optional[Dict],
    ) -> None:
        """
        Add edges to the dot graph recursively.

        Traverses the behavior tree starting from the root node and adds edges between
        nodes to the dot graph. Also collects timing information for annotators.

        :param root: The root node to start traversing from
        :param root_dot_name: The dot graph node name for the root
        :param visibility_level: The visibility level to use for collapsing subtrees
        :param collapse_decorators: Whether to collapse decorator nodes
        :param timing_info_dict: Dictionary to store timing information for annotators
        :return: Dictionary with timing information for annotators
        """
        if timing_info_dict is None:
            timing_info_dict = {}
        if isinstance(root, Decorator) and collapse_decorators:
            return
        if visibility_level < root.blackbox_level:
            for c in root.children:
                node_shape, node_colour, node_font_colour = get_node_attributes(
                    c, visibility_level
                )
                proposed_dot_name = c.name
                while proposed_dot_name in names:
                    proposed_dot_name = proposed_dot_name + "*"
                timing_info_dict[proposed_dot_name] = 0
                if isinstance(c, BaseAnnotator):
                    # Check for the compute or update method timing
                    if hasattr(c, "_times"):
                        time_dict = c._times
                    else:
                        time_dict = {}

                    if "compute" in time_dict:
                        # proposed_dot_name += f"\n({time_dict['compute']:.2}s)"
                        timing_info_dict[proposed_dot_name] = time_dict["compute"]
                    elif "update" in time_dict:
                        timing_info_dict[proposed_dot_name] = time_dict["update"]
                        # proposed_dot_name += f"\n({time_dict['update']:.2}s)"

                names.append(proposed_dot_name)
                node = Node(
                    proposed_dot_name,
                    shape=node_shape,
                    style="filled",
                    fillcolor=node_colour,
                    fontsize=fontsize,
                    fontcolor=node_font_colour,
                )
                graph.add_node(node)
                edge = Edge(root_dot_name, proposed_dot_name)
                graph.add_edge(edge)
                if c.children != []:
                    add_edges(
                        c,
                        proposed_dot_name,
                        visibility_level,
                        collapse_decorators,
                        timing_info_dict,
                    )

        return timing_info_dict

    timing_info_dict = {
        **timing_info,
        **add_edges(
            root, root.name, visibility_level, collapse_decorators, timing_info
        ),
    }
    return (graph, timing_info_dict)


def stringify_dot_tree(root: Behaviour) -> str:
    """
    Generate dot tree graphs and return a string representation
    of the dot graph.

    :param root: The root of a tree, or subtree
    :return: Dot graph as a string
    """
    graph = generate_pydot_graph(root, visibility_level=VisibilityLevel.DETAIL)
    return graph.to_string()


def render_dot_tree(
    root: Behaviour,
    visibility_level: VisibilityLevel = VisibilityLevel.DETAIL,
    collapse_decorators: bool = False,
    name: Optional[str] = None,
    threadpool_executor: Optional[ThreadPoolExecutor] = None,
    path_prefix: Optional[str] = None,
) -> None:
    """
    Render the dot tree to .dot, .svg, .png files in the current
    working directory. These will be named with the root behaviour name.

    :param root: The root of a tree, or subtree
    :param visibility_level: Collapse subtrees at or under this level
    :param collapse_decorators: Only show the decorator (not the child)
    :param name: Name to use for the created files (defaults to the root behaviour name)
    :param threadpool_executor: ThreadPoolExecutor for file writing
    :param path_prefix: String to be put before the filename

    Example:

        Render a simple tree to dot/svg/png file:

        .. graphviz:: dot/sequence.dot

        .. code-block:: python

            root = py_trees.composites.Sequence("Sequence")
            for job in ["Action 1", "Action 2", "Action 3"]:
                success_after_two = py_trees.behaviours.Count(name=job,
                                                              fail_until=0,
                                                              running_until=1,
                                                              success_until=10)
                root.add_child(success_after_two)
            py_trees.display.render_dot_tree(root)

    .. tip::

        A good practice is to provide a command line argument for optional rendering
        of a program so users can quickly visualise what tree the program will execute.
    """
    graph, timing_info_dict = generate_pydot_graph(
        root, visibility_level, collapse_decorators
    )
    filename_wo_extension = (
        root.name.lower().replace(" ", "_") if name is None else name
    )
    print("Writing %s.dot/svg/png" % filename_wo_extension)
    if path_prefix:
        filename_wo_extension = f"{path_prefix}{filename_wo_extension}"

    if threadpool_executor:
        threadpool_executor.submit(graph.write_png, filename_wo_extension + ".png")
        threadpool_executor.submit(graph.write, filename_wo_extension + ".dot")
        threadpool_executor.submit(
            write_timing_info_to_csv, filename_wo_extension, timing_info_dict
        )
    else:
        graph.write_png(filename_wo_extension + ".png")
        graph.write(filename_wo_extension + ".dot")
        write_timing_info_to_csv(filename_wo_extension, timing_info_dict)
    # graph.write_svg(filename_wo_extension + '.svg')


def write_timing_info_to_csv(
    filename_wo_extension: str, timing_info_dict: Dict[str, float]
) -> None:
    """
    Write timing information for annotators to a CSV file.

    Creates a CSV file containing the runtime information for each annotator
    in the behavior tree. The CSV has two columns: AnnotatorName and Runtime.

    :param filename_wo_extension: Base filename without extension to write the CSV to
    :param timing_info_dict: Dictionary mapping annotator names to their runtime in seconds
    """
    # Open the CSV file with write permission
    with open(f"{filename_wo_extension}.csv", "w", newline="") as csvfile:
        # Create a CSV writer using the field/column names
        writer = csv.writer(csvfile)
        writer.writerow(["AnnotatorName", "Runtime"])
        for key, value in timing_info_dict.items():
            writer.writerow([str(key).replace("\n", "").replace("\r", ""), value])
