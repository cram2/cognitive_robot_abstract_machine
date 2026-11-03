from sqlalchemy import select

from krrood.ormatic.data_access_objects.helper import to_dao
from coraplex.orm.ormatic_interface import *  # type: ignore
from coraplex.training_environments.training_environment import (
    MoveToReachTrainingEnvironment,
)


def test_move_to_reach(coraplex_testing_session):
    training_environment = MoveToReachTrainingEnvironment(visualize=False)

    training_environment.generate_episodes(2)

    assert len(training_environment.executed_plans) > 0

    coraplex_testing_session.add(to_dao(training_environment))
    coraplex_testing_session.commit()

    stored_reaches = coraplex_testing_session.scalars(select(MoveToReachDAO)).all()
    assert len(stored_reaches) == len(training_environment.tried_actions)
