"""
Execution errors preserve native plan failure persistence.
"""

from sqlalchemy import select

from coraplex.language import SequentialNode
from coraplex.plans.failures import EmptyUnderspecified
from krrood.ormatic.data_access_objects.helper import to_dao


# %% execution failure persistence
def test_native_failure_survives_database_round_trip(coraplex_testing_session) -> None:
    """
    Persist structured failures while keeping arbitrary runtime exceptions local.
    """
    node = SequentialNode(reason=EmptyUnderspecified())
    node.execution_error = RuntimeError("runtime failure")
    persisted = to_dao(node)
    session = coraplex_testing_session
    session.add(persisted)
    session.commit()
    session.expire_all()

    restored = session.scalars(select(type(persisted))).one().from_dao()

    assert type(restored.reason) is type(node.reason)
    assert restored.execution_error is None
