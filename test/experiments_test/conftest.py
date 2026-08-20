"""
Fixtures for running demonstrations against a controller in its own process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import pytest
from rclpy.action import get_action_names_and_types

from coraplex.demonstrations import RobotDemonstrationRosSession
from coraplex.perception import ROBOKUDO_QUERY_ACTION_NAME
from experiments.real_stretch_apartment_demo.demo import (
    CEREAL_SHELF_LAYER_NAME,
    CEREAL_SHELF_LAYER_T_CEREAL,
)
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
    queries. Reporting the cereal in its shelf layer's frame, at the offset the
    demonstration itself spawns it with, keeps the perceived pose and the scene from
    drifting apart as one or the other is retuned.
    """
    pytest.importorskip("robokudo_msgs")
    probe_node = stretch_controller_process.session.node

    def is_serving_queries() -> bool:
        return any(
            name.lstrip("/") == ROBOKUDO_QUERY_ACTION_NAME
            for name, _ in get_action_names_and_types(probe_node)
        )

    position = CEREAL_SHELF_LAYER_T_CEREAL.to_position().to_np().flatten()[:3]
    with StandaloneProcess(
        launcher_path=Path(__file__).parent.parent
        / "dataset"
        / "perception_pipeline_stand_in.py",
        is_ready=is_serving_queries,
        arguments=[
            "--class-label",
            CheezeIt.__name__,
            "--frame-id",
            CEREAL_SHELF_LAYER_NAME,
            "--position",
            *(str(coordinate) for coordinate in position),
        ],
    ) as process:
        yield process
