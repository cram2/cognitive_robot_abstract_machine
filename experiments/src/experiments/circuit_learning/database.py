"""
The database a run of the circuit learning benchmark keeps its records and measured rows
in, shared by every process of the run.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from krrood.entity_query_language.factories import an, entity, the, variable
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.eql_interface import eql_to_sql
from krrood.ormatic.utils import create_engine
from sqlalchemy.orm import sessionmaker
from typing_extensions import Any

from experiments.circuit_learning.records import BenchmarkRun, RunRecord, TrainedCircuit
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

    lock_timeout: timedelta = timedelta(minutes=10)
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
        # the interface maps the classes of this package, so it only exists after them
        from experiments.orm.ormatic_interface import Base

        self.path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite:///{self.path}",
            connect_args={"timeout": self.lock_timeout.total_seconds()},
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
        record = variable(RunRecord, domain=None)
        with self.session_maker() as session:
            records = eql_to_sql(an(entity(record)), session).evaluate()
            return [
                dao.from_dao()
                for dao in sorted(records, key=lambda dao: dao.database_id)
            ]

    def trained_circuit(self, *, run: BenchmarkRun) -> LayeredProbabilisticCircuit:
        """
        :param run: A training run that finished.
        :return: The circuit it learned.
        """
        trained = variable(TrainedCircuit, domain=None)
        query = the(
            entity(trained).where(
                trained.training_run.kind == run.kind,
                trained.training_run.dataset == run.dataset,
                trained.training_run.method == run.method,
                trained.training_run.setting.array_library == run.setting.array_library,
                trained.training_run.setting.device == run.setting.device,
                trained.training_run.setting.floating_point_type
                == run.setting.floating_point_type,
                trained.training_run.setting.edge_representation
                == run.setting.edge_representation,
            )
        )
        with self.session_maker() as session:
            [dao] = eql_to_sql(query, session).evaluate()
            return dao.from_dao().circuit
