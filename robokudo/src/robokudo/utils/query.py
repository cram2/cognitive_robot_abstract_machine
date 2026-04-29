from queue import Queue
from typing_extensions import Any

from py_trees.blackboard import Blackboard

from robokudo_msgs.action import Query
from robokudo.identifier import BBIdentifier


class QueryHandler(object):
    """
    QueryHandler provides an interface to interact with the ROS Action-based query interface.
    This wrapper eases the use of the various Blackboard variables devoted to the communication with the query interface.
    """

    @staticmethod
    def init_feedback_queue() -> None:
        """Initializes the feedback queue on the Blackboard."""
        blackboard = Blackboard()
        try:
            feedback_queue = blackboard.get(BBIdentifier.QUERY_FEEDBACK)
        except KeyError:
            feedback_queue = None
        if feedback_queue is None:
            feedback_queue = Queue()
            blackboard.set(BBIdentifier.QUERY_FEEDBACK, feedback_queue)

    @staticmethod
    def get_feedback_queue() -> Queue:
        """Retrieves and returns the feedback queue from the Blackboard."""
        QueryHandler.init_feedback_queue()

        blackboard = Blackboard()
        return blackboard.get(BBIdentifier.QUERY_FEEDBACK)

    @staticmethod
    def send_feedback(feedback: Query.Feedback) -> None:
        """Add a feedback part of the Query msg to the feedback queue, ready to be sent.

        :param feedback: The feedback message to send.
        """
        feedback_queue = QueryHandler.get_feedback_queue()
        feedback_queue.put(feedback)

    @staticmethod
    def send_feedback_str(feedback_str: str) -> None:
        """Add a simple string to the feedback to the feedback queue, ready to be sent.

        :param feedback_str: The string to send as feedback.
        """
        feedback_msg = Query.Feedback()
        feedback_msg.feedback = feedback_str
        QueryHandler.send_feedback(feedback_msg)

    @staticmethod
    def send_answer(result: Query.Result) -> None:
        """Raise a standard RoboKudo Query Result as a query answer to the blackboard.

        :param result: The result to raise to the blackboard.
        """
        if not isinstance(result, Query.Result):
            raise TypeError(
                f"Expected standard RoboKudo Query Result type. Got {type(result)}."
                "If you want to send other results, use QueryHandler.send_arbitrary_answer"
            )

        QueryHandler.send_arbitrary_answer(result)

    @staticmethod
    def send_arbitrary_answer(result: Any) -> None:
        """Raise any data as a query answer to the blackboard.

        :param result: The data to raise to the blackboard.
        """
        blackboard = Blackboard()
        blackboard.set(BBIdentifier.QUERY_ANSWER, result)

    @staticmethod
    def preempt_requested() -> bool:
        """Checks whether a preempt request is pending on the blackboard.

        :return: True if a preempt request is pending, False otherwise.
        """
        blackboard = Blackboard()
        try:
            is_requested = blackboard.get(BBIdentifier.QUERY_PREEMPT_REQUESTED)
        except KeyError:
            is_requested = False
        return is_requested

    @staticmethod
    def acknowledge_preempt_request() -> None:
        """Acknowledges the preempt request on the blackboard."""
        blackboard = Blackboard()
        blackboard.set(BBIdentifier.QUERY_PREEMPT_ACK, True)
