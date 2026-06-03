"""
Error handling behavior for RoboKudo behavior trees.

This module provides a behavior that clears error states stored on the behavior
tree's blackboard. It is typically used:

* At the start of a pipeline to ensure a clean error state
* After error recovery to reset error flags
* Before starting new processing sequences

The behavior integrates with RoboKudo's error handling system to manage
error states across the behavior tree.
"""

from py_trees.behaviour import Behaviour
from py_trees.common import Status

from robokudo.utils.error_handling import clear_blackboard_exception


class ClearErrors(Behaviour):
    """
    A behavior that clears error states from the blackboard.

    This behavior is used to reset error conditions that may have been
    recorded on the behavior tree's blackboard. It is commonly used:

    * As part of pipeline initialization
    * After error recovery sequences
    * Before starting new processing tasks

    The behavior always returns SUCCESS after clearing errors.
    """

    def __init__(self, name: str = "ClearErrors") -> None:
        """
        Initialize the ClearErrors behavior.

        :param name: Name of the behavior node, defaults to "ClearErrors"
        """
        super().__init__(name=name)

    def initialise(self) -> None:
        """
        Initialize the behavior.

        This method is called when the behavior is first ticked.
        No initialization is needed for this behavior.
        """
        pass

    def update(self) -> Status:
        """
        Clear any error states from the blackboard.

        This method:
        * Calls the error handling utility to clear blackboard exceptions
        * Always returns SUCCESS since clearing errors cannot fail

        :return: Always returns SUCCESS
        """
        clear_blackboard_exception()

        return Status.SUCCESS
