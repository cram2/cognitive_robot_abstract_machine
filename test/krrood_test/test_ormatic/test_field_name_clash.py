"""
Tests of mapping fields whose names clash with the columns ORMatic adds.
"""

from __future__ import annotations
import sqlalchemy


def test_duplicate_column():
    """
    Test that clashing field names do not cause duplicate columns or shadowing.
    """
    from ..dataset.ormatic_interface import ClashingEntityDAO, ClashingTargetDAO

    expected_columns = {
        "database_id",
        "clashing_target_id",
        "polymorphic_type",
        "_clashing_target_id",
    }
    actual_columns = set(ClashingEntityDAO.__table__.columns.keys())
    assert expected_columns.issubset(actual_columns), (
        f"Missing columns in ClashingEntityDAO. Expected at least {expected_columns}, found {actual_columns}"
    )

    expected_target_columns = {"database_id", "id", "polymorphic_type"}
    actual_target_columns = set(ClashingTargetDAO.__table__.columns.keys())
    assert expected_target_columns.issubset(actual_target_columns), (
        f"Missing columns in ClashingTargetDAO. Expected at least {expected_target_columns}, found {actual_target_columns}"
    )

    column = ClashingEntityDAO.__table__.columns["clashing_target_id"]
    assert not isinstance(column.type, sqlalchemy.Integer), (
        f"Column 'clashing_target_id' in ClashingEntityDAO has type {column.type}, "
        "which suggests it was shadowed by the automatically generated foreign key."
    )
