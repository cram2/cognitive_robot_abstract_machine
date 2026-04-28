"""
Goal cancellation behavior for RoboKudo behavior trees.

This module provides a behavior that checks for goal cancellation by monitoring
error states on the behavior tree's blackboard. It is used to:

* Detect when goals should be canceled due to errors
* Provide clean termination of action sequences
* Support error handling in action servers

The behavior integrates with RoboKudo's error handling system to detect
conditions that should trigger goal cancellation.
"""

from py_trees.behaviour import Behaviour
from py_trees.common import Status

from robokudo.utils.error_handling import get_blackboard_exception


class GoalCanceled(Behaviour):
    """
    A behavior that detects when goals should be canceled due to errors.

    This behavior monitors the blackboard for error states and returns:
    * SUCCESS if an error is detected, indicating the goal should be canceled
    * FAILURE if no errors are present, indicating the goal can continue

    It is typically used in action server implementations to handle error
    conditions and provide clean goal cancellation.
    """

    def __init__(self, name: str = "GoalCanceled") -> None:
        """
        Initialize the GoalCanceled behavior.

        :param name: Name of the behavior node, defaults to "GoalCanceled"
        """
        super().__init__(name=name)

    def initialise(self) -> None:
        """
        Initialize the behavior.

        This method is called when the behavior is first ticked.
        No initialization is needed for this behavior.
        """
        # self.rk_logger.debug("%s.initialise()" % self.__class__.__name__)
        pass

    def update(self) -> Status:
        """
        Check if the current goal should be canceled.

        This method:
        * Checks for error states on the blackboard
        * Returns SUCCESS if an error is found (goal should be canceled)
        * Returns FAILURE if no errors are present (goal can continue)

        :return: SUCCESS if goal should be canceled, FAILURE otherwise
        """
        # self.rk_logger.debug("%s.update()" % (self.__class__.__name__))
        if get_blackboard_exception() is not None:
            return Status.SUCCESS

        return Status.FAILURE
