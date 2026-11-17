import pytest
from krrood.exceptions import DataclassException, InputError
from dataclasses import dataclass


@dataclass(kw_only=True)
class MockException(DataclassException):
    def error_message(self) -> str:
        pass

    def suggest_correction(self) -> str:
        pass


def test_raise_from():
    try:
        try:
            raise Exception()
        except Exception as e:
            raise MockException() from e
    except MockException as e:
        print(e.raised_from)
        assert isinstance(e.raised_from, Exception)


def test_raise():
    try:
        raise MockException()
    except MockException as e:
        print(e.raised_from)
        assert e.raised_from is None


def test_raised_while_handling():
    original = ValueError()
    with pytest.raises(MockException) as exception_info:
        try:
            raise original
        except ValueError:
            raise MockException()
    assert exception_info.value.raised_while_handling is original


def test_raised_while_handling_outside_of_handler():
    with pytest.raises(MockException) as exception_info:
        raise MockException()
    assert exception_info.value.raised_while_handling is None


def test_raised_from_given_at_construction():
    cause = ValueError()
    assert MockException(raised_from=cause).__cause__ is cause


def test_raised_while_handling_given_at_construction():
    context = ValueError()
    assert MockException(raised_while_handling=context).__context__ is context


def test_construction_keeps_context_in_traceback():
    with pytest.raises(MockException) as exception_info:
        try:
            raise ValueError()
        except ValueError:
            raise MockException()
    assert not exception_info.value.__suppress_context__
