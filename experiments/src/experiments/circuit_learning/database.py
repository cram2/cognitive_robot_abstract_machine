"""
The database a run of the circuit learning benchmark keeps its records and measured rows
in, shared by every process of the run.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field
from pathlib import Path

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker
from typing_extensions import Any

from experiments.circuit_learning.records import Job, RunRecord
from experiments.orm.ormatic_interface import (
    Base,
    ComputationSettingDAO,
    JobDAO,
    RunRecordDAO,
    TrainedCircuitDAO,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

# %% database


@dataclass(kw_only=True)
class BenchmarkDatabase:
    """
    An SQLite database of the runs, circuits and rows of one benchmark.
    """

    path: Path
    """
    The file of the database, created with its tables on first use.
    """

    lock_timeout_seconds: float = 600.0
    """
    How long a process waits for another one to finish writing.
    """

    session_maker: sessionmaker = field(init=False)
    """
    The maker of the sessions on the database.
    """

    def __post_init__(self) -> None:
        """
        Create the tables that do not exist yet.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite:///{self.path}",
            connect_args={"timeout": self.lock_timeout_seconds},
        )
        Base.metadata.create_all(bind=engine)
        self.session_maker = sessionmaker(engine)

    def store(self, *, records: list[Any]) -> None:
        """
        :param records: Objects the ORM of the experiments maps, written in one
            transaction.
        """
        with self.session_maker() as session:
            session.add_all([to_dao(record) for record in records])
            session.commit()

    def run_records(self) -> list[RunRecord]:
        """
        :return: Every run that ended, in the order they were stored.
        """
        with self.session_maker() as session:
            return [
                dao.from_dao()
                for dao in session.scalars(
                    select(RunRecordDAO).order_by(RunRecordDAO.database_id)
                )
            ]

    def trained_circuit(self, *, job: Job) -> LayeredProbabilisticCircuit:
        """
        :param job: A training run that finished.
        :return: The circuit it learned.
        """
        with self.session_maker() as session:
            trained = session.scalars(
                select(TrainedCircuitDAO)
                .join(TrainedCircuitDAO.job)
                .join(JobDAO.setting)
                .where(
                    JobDAO.kind == job.kind,
                    JobDAO.dataset == job.dataset,
                    JobDAO.method == job.method,
                    ComputationSettingDAO.array_library == job.setting.array_library,
                    ComputationSettingDAO.device == job.setting.device,
                    ComputationSettingDAO.floating_point_type
                    == job.setting.floating_point_type,
                    ComputationSettingDAO.edge_representation
                    == job.setting.edge_representation,
                )
            ).one()
            return trained.from_dao().circuit
