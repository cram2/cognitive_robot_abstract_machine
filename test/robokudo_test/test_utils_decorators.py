"""
Tests of the annotator decorators in :mod:`robokudo.utils.decorators`.
"""

import logging
import time

import py_trees
import pytest

from robokudo.annotators.core import BaseAnnotator
from robokudo.utils.decorators import timer_decorator, record_time, publish_variables


class DummyTimerClass:
    def __init__(self):
        super().__init__()
        self.published = False

    @timer_decorator
    def timer(self) -> None:
        time.sleep(0.1)

    @publish_variables
    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS

    def publish_variables(self) -> None:
        self.published = True


class DummyAnnotator(BaseAnnotator):
    def __init__(self):
        super().__init__()
        self.published = False

    @record_time
    def compute(self, duration: float) -> None:
        time.sleep(duration)

    @publish_variables
    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS

    def publish_variables(self) -> None:
        self.published = True


class TestUtilsDecorators(object):
    """
    Timing, time recording and variable publishing of decorated methods.
    """

    def test_timer_decorator_logs_function_to_its_module_logger(self, *, caplog):
        """
        The run time of a decorated function is logged to its module's logger.
        """

        @timer_decorator
        def timer() -> None:
            """
            Sleep briefly.
            """
            time.sleep(0.1)

        with caplog.at_level(logging.INFO, logger=__name__):
            timer()

        assert [record.name for record in caplog.records] == [__name__]
        assert (
            caplog.records[0]
            .getMessage()
            .startswith(f"Function '{timer.__qualname__}' took")
        )

    def test_timer_decorator_names_class_of_method(self, *, caplog):
        """
        The run time of a decorated method is logged with its class name.
        """
        dummy = DummyTimerClass()

        with caplog.at_level(logging.INFO, logger=__name__):
            dummy.timer()

        assert len(caplog.records) == 1
        assert (
            caplog.records[0]
            .getMessage()
            .startswith(f"Function '{DummyTimerClass.__name__}.timer' took")
        )

    def test_record_time(self):
        """
        The decorated method's latest run time is stored under its name.
        """
        duration = 0.1
        dummy = DummyAnnotator()

        dummy.compute(duration)

        assert hasattr(dummy, "_times")
        assert dummy._times["compute"] == pytest.approx(duration, abs=0.01), (
            "Timing was not updated"
        )

        dummy.compute(duration * 2)
        assert hasattr(dummy, "_times")
        assert dummy._times["compute"] == pytest.approx(duration * 2, abs=0.01), (
            "Timing was not updated"
        )

    def test_publish_variables(self):
        dummy = DummyAnnotator()

        assert dummy.published == False
        dummy.update()
        assert dummy.published == True

    def test_publish_variables_non_annotator(self):
        dummy = DummyTimerClass()

        assert dummy.published == False
        assert pytest.raises(AssertionError, dummy.update)
        assert dummy.published == False
