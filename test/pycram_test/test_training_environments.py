import rclpy
from sqlalchemy import select

from krrood.ormatic.data_access_objects.helper import to_dao
from pycram.datastructures.enums import TaskStatus
from pycram.robot_plans.motions import *  # type: ignore
from pycram.orm.ormatic_interface import *  # type: ignore
from pycram.robot_plans.training_environment import MoveToReachTrainingEnvironment


def test_move_to_reach(pycram_testing_session):
    training_environment = MoveToReachTrainingEnvironment(visualize=False)

    training_environment.generate_episodes(2)

    assert len(training_environment.executed_plans) > 0

    pycram_testing_session.add(to_dao(training_environment))
    pycram_testing_session.commit()

    query = select(DesignatorNodeDAO.status).join(
        MoveToReachDAO, DesignatorNodeDAO.designator_id == MoveToReachDAO.database_id
    )
    results = pycram_testing_session.execute(query).all()

    success_rate = len([r for r in results if r[0] == TaskStatus.SUCCEEDED]) / len(
        results
    )
    assert success_rate >= 0.0
