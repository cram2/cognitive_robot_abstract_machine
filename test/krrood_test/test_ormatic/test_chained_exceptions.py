"""Persistence of the exceptions a
:class:`~krrood.exceptions.DataclassException` is chained to."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from krrood.ormatic.data_access_objects.helper import to_dao

from ..dataset.chained_exceptions import ExceptionWithOwnField
from ..dataset.ormatic_interface import ExceptionWithOwnFieldDAO


# %% fixtures


@pytest.fixture
def chained_exception() -> ExceptionWithOwnField:
    """
    An exception raised from a plain exception while handling a dataclass exception.
    """
    context = ExceptionWithOwnField(1)
    cause = ValueError("the cause")
    return ExceptionWithOwnField(2, raised_from=cause, raised_while_handling=context)


@pytest.fixture
def reloaded_exception(
    chained_exception: ExceptionWithOwnField, session, database
) -> ExceptionWithOwnField:
    """
    The chained exception after a round trip through the database.
    """
    session.add(to_dao(chained_exception))
    session.commit()
    session.expunge_all()
    return session.scalars(select(ExceptionWithOwnFieldDAO)).one().from_dao()


# %% round trip


def test_own_field_survives_round_trip(
    chained_exception: ExceptionWithOwnField,
    reloaded_exception: ExceptionWithOwnField,
):
    assert reloaded_exception.offending_value == chained_exception.offending_value


def test_plain_cause_survives_round_trip(
    chained_exception: ExceptionWithOwnField,
    reloaded_exception: ExceptionWithOwnField,
):
    assert type(reloaded_exception.__cause__) is type(chained_exception.__cause__)
    assert str(reloaded_exception.__cause__) == str(chained_exception.__cause__)


def test_dataclass_context_survives_round_trip(
    chained_exception: ExceptionWithOwnField,
    reloaded_exception: ExceptionWithOwnField,
):
    assert (
        reloaded_exception.__context__.offending_value
        == chained_exception.__context__.offending_value
    )
