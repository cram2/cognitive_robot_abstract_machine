import argparse

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.daisy.configs import (
    DAiSyInteractiveMarkerChain,
    WorldWithDAiSyConfig,
)
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    InteractiveMarkerNode,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig


def create_giskard() -> Giskard:
    """
    Build a Giskard that simulates DAiSy, described by the file its robot class
    declares.
    """
    return Giskard.standalone(
        world_config=WorldWithDAiSyConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=33),
    )


def create_giskard_with_interactive_markers() -> Giskard:
    """
    Build the Giskard :func:`create_giskard` builds and start the interactive marker
    server that moves DAiSy's grippers through it from RViz.
    """
    giskard = create_giskard()
    InteractiveMarkerNode.start_in_background_thread(
        chains=[chain.value for chain in DAiSyInteractiveMarkerChain]
    )
    return giskard


def main() -> None:
    """
    Run the Giskard :func:`create_giskard` builds as a ros node until ROS shuts down,
    with the interactive markers if ``--interactive-marker`` is given.
    """
    parser = argparse.ArgumentParser(description="DAiSy Giskard standalone controller.")
    parser.add_argument(
        "--interactive-marker",
        action="store_true",
        help="Also start the interactive marker server for Cartesian control via RViz.",
    )
    # parse_known_args ignores ROS 2 arguments (--ros-args ...) that argparse does not know about.
    args, _ = parser.parse_known_args()
    if args.interactive_marker:
        Giskard.launch(create_giskard_with_interactive_markers)
    else:
        Giskard.launch(create_giskard)


if __name__ == "__main__":
    main()
