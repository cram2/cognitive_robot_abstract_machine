"""
Action server monitoring behaviors for RoboKudo behavior trees.

This module provides behaviors for monitoring and controlling ROS action servers
in RoboKudo. It includes behaviors for:

* Checking action server activity state
* Handling preemption requests
* Managing goal abortion
* Exception handling

These behaviors are typically used in pipelines that interact with ROS action
servers to ensure proper state management and error handling.
"""

import logging

from py_trees.common import Status
from py_trees.behaviour import Behaviour
from py_trees.blackboard import Blackboard

from robokudo.defs import PACKAGE_NAME
from robokudo.identifier import BBIdentifier
from robokudo.utils.error_handling import (
    catch_and_raise_to_blackboard,
    has_blackboard_exception,
    get_blackboard_exception,
)


class ActionServerActive(Behaviour):
    """A behavior that checks if an action server is active.

    This behavior monitors the action server's state and returns:
    * SUCCESS if the server is active and processing a goal
    * FAILURE if the server is not found or not active
    """

    def __init__(self, name: str = "ActionServerActive") -> None:
        """Initialize the ActionServerActive behavior.

        :param name: Name of the behavior node, defaults to "ActionServerActive"
        """
        super().__init__(name=name)

    def update(self) -> Status:
        """
        Check if the action server is active.

        This method:
        * Retrieves the action server from the blackboard
        * Checks if it exists and is active
        * Returns appropriate status based on server state

        :return: SUCCESS if server is active, FAILURE otherwise
        """
        blackboard = Blackboard()
        action_server = blackboard.get(BBIdentifier.QUERY_SERVER)
        if action_server is None:
            return Status.FAILURE

        if action_server.is_active():
            return Status.SUCCESS

        return Status.FAILURE


class ActionServerCheck(Behaviour):
    """
    A behavior that monitors action server state transitions.

    This behavior is used to ensure proper pipeline synchronization with
    action server state. It returns:
    * RUNNING if the server is active (processing a goal)
    * FAILURE if the server is not found
    * SUCCESS if the server is no longer active

    This prevents pipelines from proceeding until action server processing
    is complete.
    """

    def __init__(self, name: str = "ActionServerCheck") -> None:
        """Initialize the ActionServerCheck behavior.

        :param name: Name of the behavior node, defaults to "ActionServerCheck"
        """
        super().__init__(name=name)

    def update(self) -> Status:
        """Check action server state.

        This method:
        * Retrieves the action server from the blackboard
        * Returns RUNNING if server is active
        * Returns FAILURE if server not found
        * Returns SUCCESS if server is no longer active

        :return: Status based on server state
        """
        blackboard = Blackboard()
        action_server = blackboard.get(BBIdentifier.QUERY_SERVER)
        if action_server is None:
            return Status.FAILURE

        if action_server.is_active():
            return Status.RUNNING

        return Status.SUCCESS


class ActionServerNoPreemptRequest(Behaviour):
    """
    A behavior that handles action server preemption requests.

    This behavior monitors for preemption requests and handles them by:
    * Checking if a preempt request exists on the blackboard
    * Acknowledging the request if found
    * Returning appropriate status to trigger preemption

    :ivar name: Name of the behavior node
    """

    def __init__(self, name: str = "ActionServerNoPreemptRequest") -> None:
        """Initialize the ActionServerNoPreemptRequest behavior.

        :param name: Name of the behavior node, defaults to "ActionServerNoPreemptRequest"
        """
        super().__init__(name=name)

    def update(self) -> Status:
        """Check for and handle preemption requests.

        This method:
        * Checks the blackboard for preemption requests
        * Acknowledges requests if found
        * Returns FAILURE to trigger preemption on request
        * Returns SUCCESS if no preemption requested

        :return: FAILURE if preemption requested, SUCCESS otherwise
        """
        blackboard = Blackboard()
        if blackboard.get(BBIdentifier.QUERY_PREEMPT_REQUESTED):
            blackboard.set(BBIdentifier.QUERY_PREEMPT_ACK, True)
            return Status.FAILURE
        else:
            return Status.SUCCESS


class AbortGoal(Behaviour):
    """
    A behavior that aborts the current goal with an error message.

    This behavior raises an exception that is caught and stored on the
    blackboard to trigger goal abortion. It is used to explicitly
    terminate goals with a specified error message.
    """

    def __init__(
        self, name: str = "AbortGoal", msg: str = "Goal has been aborted"
    ) -> None:
        """Initialize the AbortGoal behavior.

        :param name: Name of the behavior node, defaults to "AbortGoal"
        :param msg: Error message for goal abortion, defaults to "Goal has been aborted"
        """
        super().__init__(name=name)

        self.msg: str = msg
        """Error message for goal abortion"""

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """Abort the current goal.

        This method raises an exception with the specified message.
        The exception is caught by the decorator and stored on the
        blackboard.

        :raises Exception: Always raises an exception with the abort message
        :return: Never returns due to exception
        """
        raise Exception(self.msg)


class RunningUntilExceptionHandled(Behaviour):
    """A behavior that waits for exception handling to complete.

    This behavior monitors the blackboard for active exceptions and
    returns RUNNING until they are cleared. It is used to prevent
    further processing until error conditions are resolved.
    """

    def __init__(
        self,
        name: str = "RunningUntilExceptionHandled",
        msg: str = "Running Until Exception Handled",
    ) -> None:
        """Initialize the RunningUntilExceptionHandled behavior.

        :param name: Name of the behavior node, defaults to "RunningUntilExceptionHandled"
        :param msg: Status message for waiting state, defaults to "Running Until Exception Handled"
        """
        super().__init__(name=name)

        self.msg: str = msg
        """Status message for waiting state"""

    def update(self) -> Status:
        """Check if exception handling is complete.

        This method:
        * Checks for active exceptions on the blackboard
        * Returns SUCCESS if no exceptions are present
        * Returns RUNNING if exceptions still need handling

        :return: SUCCESS if no exceptions, RUNNING otherwise
        """
        # No exception set yet? All good.
        if not has_blackboard_exception():
            return Status.SUCCESS

        # Exception was already initialized but is None? All good.
        if get_blackboard_exception() is None:
            return Status.SUCCESS

        self.feedback_message = "Waiting for exception handling"
        return Status.RUNNING


class ActionServerPresentAndDone(Behaviour):
    """A behaviour that checks whether an action server is present and represents its state."""

    def __init__(self, name: str = "ActionServerPresentAndDone") -> None:
        """Initialize the ActionServerPresentAndDone behaviour.

        :param name: Name of the behaviour node, defaults to "ActionServerPresentAndDone"
        """
        super().__init__(name=name)

        self.rk_logger: logging.Logger = logging.getLogger(PACKAGE_NAME)
        """Logger for this behaviour."""

    def update(self) -> Status:
        blackboard = Blackboard()
        if not blackboard.exists(BBIdentifier.QUERY_SERVER):
            # print("No ActionServer(AS) found - Skipping all following checks that are AS related")
            return Status.SUCCESS

        if not blackboard.get(BBIdentifier.QUERY_SERVER_IN_PIPELINE):
            # print("ActionServer(AS) found, but not used in Pipeline")
            return Status.SUCCESS

        action_server = blackboard.get(BBIdentifier.QUERY_SERVER)

        self.rk_logger.debug("ActionServer in pipeline!")

        # The Action Server has to pick up the exception and deliver it to the client. Wait until that happened.
        if has_blackboard_exception() and get_blackboard_exception() is not None:
            self.rk_logger.info("ActionServer needs to deliver exception")
            return Status.RUNNING

        # If the Action Server is still processing the current goal, wait.
        # It's the responsibility of the rest of the tree to send the goal or re-iterate without
        # calling *this* Behavior again.
        if action_server.is_active():
            self.rk_logger.info("ActionServer needs to finish current goal.")
            return Status.RUNNING

        return Status.SUCCESS
