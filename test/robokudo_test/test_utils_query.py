import queue

import pytest
from py_trees.blackboard import Blackboard

from robokudo.identifier import BBIdentifier
from robokudo.utils.query import QueryHandler
from robokudo_msgs.action import Query


class TestUtilsQuery:
    def test_init_feedback_queue(self):
        blackboard = Blackboard()

        QueryHandler.init_feedback_queue()

        fb_queue = blackboard.get(BBIdentifier.QUERY_FEEDBACK)
        assert fb_queue is not None
        assert isinstance(fb_queue, queue.Queue)

        QueryHandler.init_feedback_queue()

        fb_queue2 = blackboard.get(BBIdentifier.QUERY_FEEDBACK)
        assert fb_queue2 is not None
        assert isinstance(fb_queue2, queue.Queue)
        assert fb_queue2 == fb_queue, f"feedback queue should not be reinitialized"

    def test_get_feedback_queue(self):
        fb_queue = QueryHandler.get_feedback_queue()
        assert fb_queue is not None
        assert isinstance(fb_queue, queue.Queue)

    def test_send_feedback(self):
        fb = Query.Feedback()
        fb.feedback = "Test Feedback"
        QueryHandler.send_feedback(fb)

        fb_queue = Blackboard().get(BBIdentifier.QUERY_FEEDBACK)
        fb_item = fb_queue.get()
        assert fb_item == fb
        assert fb_item.feedback == fb.feedback

    def test_send_feedback_str(self):
        fb = "Test Feedback"
        QueryHandler.send_feedback_str(feedback_str=fb)

        fb_queue = Blackboard().get(BBIdentifier.QUERY_FEEDBACK)
        fb_item = fb_queue.get()
        assert fb_item.feedback == fb

    def test_send_answer(self):
        answer = Query.Result()
        answer.text_result = "Test Answer"
        QueryHandler.send_answer(answer)

        answer_item = Blackboard().get(BBIdentifier.QUERY_ANSWER)
        assert answer_item == answer
        assert answer_item.text_result == answer.text_result

    def test_send_answer_invalid_type(self):
        answer = "Test Answer"
        assert pytest.raises(TypeError, QueryHandler.send_answer, answer)

    @pytest.mark.parametrize("answer", ["Test Answer", 42, 42.0])
    def test_send_arbitrary_answer(self, answer):
        QueryHandler.send_arbitrary_answer(answer)

        answer_item = Blackboard().get(BBIdentifier.QUERY_ANSWER)
        assert answer_item == answer

    def test_preempt_requested(self):
        blackboard = Blackboard()

        # Works uninitialized
        is_requested = QueryHandler.preempt_requested()
        assert is_requested == False

        blackboard.set(BBIdentifier.QUERY_PREEMPT_REQUESTED, True)

        # Works with preempt requested
        is_requested = QueryHandler.preempt_requested()
        assert is_requested == True

        blackboard.set(BBIdentifier.QUERY_PREEMPT_REQUESTED, False)

        # Works with no preempt requested
        is_requested = QueryHandler.preempt_requested()
        assert is_requested == False

    def test_acknowledge_preempt_request(self):
        blackboard = Blackboard()

        assert pytest.raises(KeyError, blackboard.get, BBIdentifier.QUERY_PREEMPT_ACK)

        QueryHandler.acknowledge_preempt_request()

        assert blackboard.get(BBIdentifier.QUERY_PREEMPT_ACK) == True
