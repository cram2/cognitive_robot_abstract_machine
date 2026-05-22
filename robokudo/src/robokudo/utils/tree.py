"""Behavior tree utilities for RoboKudo.

This module provides helper functions for working with py_trees behavior trees.
It supports:

* Traversing trees while excluding certain node types
* Finding nodes by type or name
* Setting up trees and their descendants
* Managing parent-child relationships
* Generating scoped names for behaviors

Dependencies:

* py_trees for behavior tree functionality
* py_trees_ros for ROS integration
"""

from __future__ import annotations

import uuid

from py_trees.behaviour import Behaviour
from typing_extensions import TYPE_CHECKING, Optional, Generator, List, Type

if TYPE_CHECKING:
    from py_trees_ros.trees import BehaviourTree
    from py_trees.composites import Composite


def behavior_iterate_except_type(
    tree: Behaviour,
    child_type: Type,
    direct_descendants: bool = False,
    include_tree: bool = False,
) -> Generator[Behaviour, None, None]:
    """Generator similar to Behavior.iterate(). But this one doesn't traverse
    nodes of a specific type. It also traverses in a post-order manner like Behavior.iterate().
    @param include_tree will control if you want to include 'tree' itself in the enumeration,
    like it is done in Behavior.iterate()

    :param tree: Root behavior to start iteration from
    :param child_type: Node type to exclude from iteration
    :param direct_descendants: Only iterate over direct children, defaults to False
    :param include_tree: Include root node in iteration, defaults to False
    :return: Generator yielding behavior nodes
    """
    for child in tree.children:
        if isinstance(child, child_type):
            continue

        if not direct_descendants:
            for node in behavior_iterate_except_type(
                child, child_type, direct_descendants, True
            ):
                yield node
        else:
            yield child

    if include_tree:
        yield tree


def find_parent_of_type(behaviour: Behaviour, parent_type: Type) -> Optional[Behaviour]:
    """Traverse the given behaviour up until we either hit the top of the tree or find a node of type parent_type.

    :param behaviour: Starting behavior node
    :param parent_type: Type of parent to find
    :return: First parent of specified type, or None if not found
    """
    current_parent = behaviour.parent
    while current_parent is not None:
        if type(current_parent) == parent_type:
            return current_parent
        else:
            current_parent = current_parent.parent
    return None


def find_children_with_name(
    composite: Composite,
    name: str,
    direct_descendants: bool = False,
) -> Optional[Behaviour]:
    """Iterate() the given composite over its children and return the first child
    with child.name == name.

    :param composite: Parent node to search in
    :param name: Name to search for
    :param direct_descendants: Only search direct children, defaults to False
    :return: First child with matching name, or None if not found
    """

    for child in composite.iterate(direct_descendants=direct_descendants):
        if child.name == name:
            return child
    return None


def get_scoped_list_of_names(
    behaviour: Behaviour, scoping_behaviour_type: Type
) -> List[str]:
    """This method can generate behaviour names which represent the structure of the tree.
    For that, we can prefix (or scope) the name of the input behaviour with a scoping behaviour parent.
    Imagine a tree like this:
      Sequence ("Z")
       |
      Sequence ("Y")
       |
      Behaviour ("B")

    Then get_scoped_list(b,Sequence) would return a list of the instances [B,Y,Z]

    :param behaviour: Starting behavior node
    :param scoping_behaviour_type: Type of ancestors to include
    :return: List of behavior names from root to leaf
    """
    current_parent = behaviour.parent
    list_of_names = [behaviour.name]

    while current_parent is not None:
        if type(current_parent) == scoping_behaviour_type:
            list_of_names.append(current_parent.name)
        current_parent = current_parent.parent

    list_of_names.reverse()

    return list_of_names


def get_scoped_name(
    behaviour: Behaviour, scoping_behaviour_type: Type, delimiter: str = "/"
) -> str:
    """Get scoped name string for behavior.

    Joins ancestor names with delimiter to create a scoped name.

    :param behaviour: Starting behavior node
    :param scoping_behaviour_type: Type of ancestors to include
    :param delimiter: String to join names with, defaults to "/"
    :return: Delimited string of ancestor names
    """
    return delimiter.join(get_scoped_list_of_names(behaviour, scoping_behaviour_type))


def setup_with_descendants_rk(tree: BehaviourTree, setup_timeout: float = 0) -> None:
    """Set up ROS behavior tree and all descendants.

    Call setup(0) on all children of tree.root

    :param tree: A BehaviourTree which already constains all trees that should be setup'ed
    during startup.
    :param setup_timeout: Timeout value that is passed to the setup of each child.
    :return: Result of setup operation
    """
    return setup_with_descendants_on_behavior(tree.root, setup_timeout=setup_timeout)


def setup_with_descendants_on_behavior(
    tree: Behaviour, setup_timeout: float = 0
) -> None:
    """Set up behavior tree node and all descendants.

    Call setup(0) on all children of tree

    :param tree: A Behaviour, which might also be a composition
    :param setup_timeout: Timeout value that is passed to the setup of each child.
    """
    for child in tree.children:
        for node in child.iterate():
            node.setup(timeout=setup_timeout, node=None, visitor=None)
    tree.setup(timeout=setup_timeout, node=None, visitor=None)


def fix_parent_relationship_of_childs(behavior: Behaviour) -> None:
    """Fix parent references in behavior tree.

    Iterate top-down over this tree and reset the parent relationship for all childs.
    This may be required if you dynamically assign the same instance of a Behavior Tree node
    into different parts of the BT or also have the same instance in multiple individual Behavior Trees.

    :param behavior: The behavior to start with. This is typically the root node of the tree you need to "repair"
    """
    for child in behavior.children:
        child.parent = behavior
        fix_parent_relationship_of_childs(child)


def find_root(behavior: Behaviour) -> Behaviour:
    """Find the root of this behavior tree

    :param behavior: The behaviour to start searching from.
    :return: The root node of the tree
    """
    current_parent = behavior.parent
    previous_node = behavior

    while current_parent is not None:
        previous_node = current_parent
        current_parent = current_parent.parent

    return previous_node


def add_child_to_parent(parent: Composite, child: Behaviour) -> uuid.UUID:
    """Adds the given child to the parent composite node.

    Add a child. In contrast to the standard add_child method of py_trees,
    this version does not check if the parent is already set.
    This is currently required in dynamic PPT selection during runtime,
    where subtrees are (re-)introduces based on runtime queries.

    :param parent: the composite node to add child to
    :param child: child to add
    :raises TypeError: if the child is not an instance of `Behaviour`
    :return: unique id of the child
    """
    if not isinstance(child, Behaviour):
        raise TypeError(
            "children must be behaviours, but you passed in {}".format(type(child))
        )
    parent.children.append(child)
    child.parent = parent
    return child.id


def add_children_to_parent(parent: Composite, children: List[Behaviour]) -> Behaviour:
    """Append a list of children to the current list.

    :param parent: the composite node to add child to
    :param children: list of children to add
    :return: parent node
    """
    for child in children:
        add_child_to_parent(parent, child)
    return parent
