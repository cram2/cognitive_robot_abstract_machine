"""
The ORM build every experiment test may need.

Each experiment keeps its own tests, and the fixtures they need, in a package of its own
under this one, so that tests needing ROS and tests needing nothing but the domain live
side by side.
"""

from __future__ import annotations

# Built before anything reads a mapped datastructure: pytest imports every conftest of a
# run before calling any hook, so a hook would fire too late. The build runs once per
# process and never on an xdist worker.
from ..orm_interface_build import regenerate_orm_interfaces

regenerate_orm_interfaces()


from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import pytest
from rclpy.action import get_action_names_and_types
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.sqltypes import NullType

from coraplex.demonstrations import RobotDemonstrationRosSession
from coraplex.perception import ROBOKUDO_QUERY_ACTION_NAME
from experiments.real_stretch_apartment_demo.demo import CEREAL_NAME
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.semantic_annotations.semantic_annotations import CheezeIt

from coraplex.testing import StandaloneProcess

# %% standalone controller process

WORLD_FETCH_SERVICE_SUFFIX = "fetch_world"
"""
Suffix of the service a controller announces once it is serving its world.
"""


@dataclass
class StandaloneControllerProcess:
    """
    A standalone Giskard controller running in its own process, as on the robot.

    Nothing is shared with the test's interpreter, so every exchange with the controller
    has to survive the middleware.
    """

    launcher_path: Path
    """
    Standalone controller script to run.
    """

    session: RobotDemonstrationRosSession | None = field(init=False, default=None)
    """
    Session watching the controller's services.

    It owns the ROS context for the whole test, so a demonstration running against this
    controller finds a context it did not create and leaves it alone.
    """

    process: StandaloneProcess | None = field(init=False, default=None)
    """
    The controller's process, once started.
    """

    def start(self) -> None:
        """
        Launch the controller and block until it serves its world.
        """
        self.session = RobotDemonstrationRosSession.start("controller_process_probe")
        self.process = StandaloneProcess(
            launcher_path=self.launcher_path, is_ready=self.is_serving_world
        )
        self.process.start()

    def is_serving_world(self) -> bool:
        """
        Whether the controller advertises its world-fetch service yet.
        """
        return any(
            name.endswith(WORLD_FETCH_SERVICE_SUFFIX)
            for name, _ in self.session.node.get_service_names_and_types()
        )

    def stop(self) -> None:
        """
        Stop the controller and release the ROS context.
        """
        self.process.stop()
        self.session.stop()

    def __enter__(self) -> StandaloneControllerProcess:
        self.start()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.stop()


@pytest.fixture()
def stretch_controller_process():
    """
    A standalone Giskard controller for the Stretch, running in its own process.
    """
    launcher = files("giskardpy").joinpath(
        "middleware/ros2/scripts/iai_robots/stretch/stretch_standalone.py"
    )
    with StandaloneControllerProcess(launcher_path=Path(str(launcher))) as controller:
        yield controller


# %% standalone perception pipeline


@pytest.fixture()
def cereal_perception_process(stretch_controller_process):
    """
    A perception pipeline reporting the demonstration's cereal, in its own process.

    The demonstration detects before it grasps, so a real run needs a pipeline answering
    queries. The cereal is reported at the origin of its own frame, so the pipeline sees
    it wherever it currently stands: a detection taken after the demonstration has
    carried it somewhere else confirms that pose instead of pulling it back to the one it
    was spawned at.
    """
    pytest.importorskip("robokudo_msgs")
    probe_node = stretch_controller_process.session.node

    def is_serving_queries() -> bool:
        return any(
            name.lstrip("/") == ROBOKUDO_QUERY_ACTION_NAME
            for name, _ in get_action_names_and_types(probe_node)
        )

    cereal_origin = (0.0, 0.0, 0.0)

    with StandaloneProcess(
        launcher_path=Path(__file__).parent.parent
        / "dataset"
        / "perception_pipeline_stand_in.py",
        is_ready=is_serving_queries,
        arguments=[
            "--class-label",
            CheezeIt.__name__,
            "--frame-id",
            CEREAL_NAME,
            "--position",
            *(str(coordinate) for coordinate in cereal_origin),
        ],
    ) as process:
        yield process


# %% montessori results ORM session


@pytest.fixture(scope="function")
def montessori_results_session():
    """
    An in-memory SQLite session against the montessori results schema (see
    :mod:`experiments.montessori.sorting_results`), mirroring
    ``coraplex_testing_session`` in ``test/coraplex_test/conftest.py``.

    Skips any table ORMatic could not assign a real column type to (surfaced as
    SQLAlchemy's ``NullType``, e.g. ``EpisodePlayerDAO.rdr_viewer`` for the
    ``RDRCaseViewer`` field it doesn't have a mapping for) rather than letting one
    unrelated, pre-existing gap in the huge generated ``experiments`` schema stop
    ``create_all`` from creating every other table.
    """
    from experiments.orm.ormatic_interface import Base

    engine = create_engine("sqlite:///:memory:")
    session = sessionmaker(engine)()
    creatable_tables = [
        table
        for table in Base.metadata.tables.values()
        if not any(isinstance(column.type, NullType) for column in table.columns)
    ]
    Base.metadata.create_all(bind=session.bind, tables=creatable_tables)
    yield session
    session.close()
