"""
The database a run of the circuit learning benchmark keeps its records and measured rows
in, shared by every process of the run.
"""

from __future__ import annotations

# %% imports
import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from krrood.adapters.json_serializer import from_json, to_json
from krrood.entity_query_language.factories import an, entity, variable
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.eql_interface import eql_to_sql
from krrood.ormatic.utils import create_engine
from sqlalchemy.orm import sessionmaker
from typing_extensions import Any

from experiments.circuit_learning.records import BenchmarkRun, RunRecord
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


# %% trained circuits


@dataclass(kw_only=True)
class TrainedCircuitFiles:
    """
    The circuits training runs learned, one JSON file per run, kept for the runs that
    measure their queries.
    """

    directory: Path
    """
    The directory of the files.
    """

    def path(self, *, training_run: BenchmarkRun) -> Path:
        """
        :param training_run: A training run.
        :return: The file of the circuit it learned.
        """
        return self.directory / f"{training_run.key}.json"

    def write(
        self, *, training_run: BenchmarkRun, circuit: LayeredProbabilisticCircuit
    ) -> None:
        """
        :param training_run: The training run that learned the circuit.
        :param circuit: The learned circuit, with host parameters.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path(training_run=training_run).open("w") as file:
            json.dump(to_json(circuit), file)

    def read(self, *, training_run: BenchmarkRun) -> LayeredProbabilisticCircuit:
        """
        :param training_run: A training run that finished.
        :return: The circuit it learned.
        """
        with self.path(training_run=training_run).open() as file:
            return from_json(json.load(file))
