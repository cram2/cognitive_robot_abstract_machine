"""
Loading the results of a query for their conversion to domain objects.
"""

from __future__ import annotations

from sqlalchemy import event

from krrood.entity_query_language.factories import an, entity, variable
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.eql_interface import eql_to_sql
from krrood.ormatic.result_loading import RelationshipLoading, ResultLoading
from ..dataset.example_classes import KRROODOrientation, KRROODPose, KRROODPosition

# %% helpers


def store_poses(session, *, pose_count: int) -> None:
    """
    :param session: The session to store the poses with.
    :param pose_count: How many poses to store.
    """
    for index in range(pose_count):
        session.add(
            to_dao(
                KRROODPose(
                    position=KRROODPosition(x=float(index), y=0, z=0),
                    orientation=KRROODOrientation(x=0, y=0, z=0, w=1),
                )
            )
        )
    session.commit()
    session.expunge_all()


def count_statements_converting_poses(session) -> int:
    """
    :param session: The session holding the stored poses.
    :return: How many statements selecting every pose and converting it take, loaded
        for conversion.
    """
    statements = []

    def count_statement(*arguments) -> None:
        statements.append(arguments)

    pose = variable(KRROODPose, domain=None)
    event.listen(session.bind, "after_cursor_execute", count_statement)
    try:
        for dao in eql_to_sql(
            an(entity(pose)),
            session,
            result_loading=ResultLoading(
                relationship_loading=RelationshipLoading.FOR_CONVERSION
            ),
        ).evaluate():
            dao.from_dao()
    finally:
        event.remove(session.bind, "after_cursor_execute", count_statement)
    session.expunge_all()
    return len(statements)


# %% loading for conversion


def test_converting_results_loaded_for_conversion_takes_statements_per_relationship(
    session, database
):
    """
    Selecting and converting objects loaded for conversion takes as many statements for
    many objects as for one.
    """
    store_poses(session, pose_count=1)
    statements_for_one = count_statements_converting_poses(session)
    store_poses(session, pose_count=4)

    assert count_statements_converting_poses(session) == statements_for_one
