"""
Conditioning a fitted relational circuit on a class-level enum attribute through an EQL
query.
"""

from __future__ import annotations

import numpy as np
import pytest

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from ..dataset import ormatic_interface  # type: ignore # noqa: F401
from ..dataset.example_classes import (
    GraspAttempt,
    RobotStation,
    StationedPickingRobot,
)


def _stationed_robots(
    random_state: np.random.Generator, station: RobotStation, count: int
) -> list[StationedPickingRobot]:
    return [
        StationedPickingRobot(
            station=station,
            skill=float(random_state.uniform(0, 1)),
            attempts=[
                GraspAttempt(
                    arm=float(random_state.uniform(0, 1)),
                    grasped=bool(random_state.integers(0, 2)),
                )
                for _ in range(3)
            ],
        )
        for _ in range(count)
    ]


@pytest.fixture
def stationed_robot_circuit() -> RelationalProbabilisticCircuit:
    random_state = np.random.default_rng(0)
    robots = _stationed_robots(random_state, RobotStation.LAB, 20) + _stationed_robots(
        random_state, RobotStation.FACTORY, 20
    )
    model = RelationalProbabilisticCircuit(StationedPickingRobot)
    model.fit([to_dao(robot) for robot in robots])
    return model


def test_query_conditioned_on_a_class_level_enum_yields_that_value(
    stationed_robot_circuit,
):
    """
    Regression test: a class-level enum attribute used to be stored in the fitted class
    circuit under a hash that had been rounded through a float, so conditioning on the
    enum member itself matched nothing and the query found no solution at all.
    """
    query = a(StationedPickingRobot)(
        station=RobotStation.LAB,
        skill=...,
        attempts=[a(GraspAttempt)(arm=..., grasped=...)],
    )
    backend = ProbabilisticBackend(
        model_registry=RelationalCircuitRegistry(
            relational_probabilistic_circuit=stationed_robot_circuit
        ),
        number_of_samples=5,
    )

    results = list(query.evaluate(backend=backend))

    assert len(results) == 5
    assert all(result.station == RobotStation.LAB for result in results)
