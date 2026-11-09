"""Persistence of the exceptions a
:class:`~krrood.exceptions.DataclassException` is chained to."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from krrood.ormatic.data_access_objects.helper import to_dao

from ..dataset.chained_exceptions import ExceptionWithOwnField
from ..dataset.ormatic_interface import ExceptionMappingDAO, ExceptionWithOwnFieldDAO


# %% fixtures


@pytest.fixture
def chained_exception() -> ExceptionWithOwnField:
    """
    An exception raised from a plain exception, itself raised from another plain
    exception, while handling a dataclass exception.
    """
    context = ExceptionWithOwnField(1)
    cause = ValueError("the cause")
    cause.__cause__ = KeyError("the cause of the cause")
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
    query = select(ExceptionWithOwnFieldDAO).where(
        ExceptionWithOwnFieldDAO.offending_value == chained_exception.offending_value
    )
    return session.scalars(query).one().from_dao()


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


def test_cause_of_plain_cause_survives_round_trip(
    chained_exception: ExceptionWithOwnField,
    reloaded_exception: ExceptionWithOwnField,
):
    original = chained_exception.__cause__.__cause__
    reloaded = reloaded_exception.__cause__.__cause__
    assert type(reloaded) is type(original)
    assert str(reloaded) == str(original)


# %% tables


def test_every_exception_is_a_row(
    chained_exception: ExceptionWithOwnField, session, database
):
    session.add(to_dao(chained_exception))
    session.commit()
    chain = [
        chained_exception,
        chained_exception.__context__,
        chained_exception.__cause__,
        chained_exception.__cause__.__cause__,
    ]
    assert len(session.scalars(select(ExceptionMappingDAO)).all()) == len(chain)
