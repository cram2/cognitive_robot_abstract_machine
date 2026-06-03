from typing import Generator

import py_trees
import pytest
from py_trees.blackboard import Blackboard

from robokudo.annotators.core import BaseAnnotator
from robokudo.identifier import BBIdentifier
from robokudo.utils.error_handling import (
    raise_to_blackboard,
    has_blackboard_exception,
    get_blackboard_exception,
    clear_blackboard_exception,
    catch_and_raise_to_blackboard,
)


class DummyAnnotator(BaseAnnotator):
    def __init__(self):
        super().__init__()
        self.published = False

    @catch_and_raise_to_blackboard
    def update(self, fail: bool = True) -> py_trees.common.Status:
        if fail:
            raise Exception("Test")
        else:
            return py_trees.common.Status.SUCCESS


class TestUtilsErrorHandling(object):

    @pytest.fixture(scope="function")
    def blackboard(self) -> Generator[Blackboard, None, None]:
        blackboard = Blackboard()
        yield blackboard
        blackboard.clear()

    def test_raise_to_blackboard(self, blackboard: Blackboard):
        exception = Exception("Test")
        try:
            raise exception
        except Exception as e:
            raise_to_blackboard(e)
        assert (
            blackboard.get(BBIdentifier.BLACKBOARD_EXCEPTION_NAME) == exception
        ), "Blackboard should contain exception"

    def test_has_blackboard_exception(self, blackboard: Blackboard):
        assert has_blackboard_exception() == False
        raise_to_blackboard(Exception("Test"))
        assert has_blackboard_exception() == True

    def test_get_blackboard_exception(self, blackboard: Blackboard):
        exception = Exception("Test")

        assert pytest.raises(
            KeyError, get_blackboard_exception
        ), "Blackboard should be empty"
        raise_to_blackboard(exception)
        assert (
            get_blackboard_exception() == exception
        ), "Blackboard should contain exception"

    def test_clear_blackboard_exception(self, blackboard: Blackboard):
        exception = Exception("Test")
        assert pytest.raises(
            KeyError, get_blackboard_exception
        ), "Blackboard should be empty"
        raise_to_blackboard(exception)
        assert (
            get_blackboard_exception() == exception
        ), "Blackboard should contain exception"

        clear_blackboard_exception()

        assert (
            get_blackboard_exception() is None
        ), "Blackboard should no longer contain exception"

    def test_catch_and_raise_to_blackboard(self, blackboard: Blackboard):
        dummy = DummyAnnotator()
        assert has_blackboard_exception() == False

        status = dummy.update(fail=False)
        assert status == py_trees.common.Status.SUCCESS
        assert (
            has_blackboard_exception() == False
        ), "No exception should be raised to blackboard"

        status = dummy.update(fail=True)
        assert has_blackboard_exception() == True
        assert (
            status == py_trees.common.Status.FAILURE
        ), "Exception should be raised to blackboard"

        status = dummy.update(fail=False)
        assert has_blackboard_exception() == True
        assert (
            status == py_trees.common.Status.FAILURE
        ), "Exception should persist until handled"

        clear_blackboard_exception()
        status = dummy.update(fail=False)
        assert has_blackboard_exception() == False
        assert (
            status == py_trees.common.Status.SUCCESS
        ), "Annotator should work fine after exception was handled"
