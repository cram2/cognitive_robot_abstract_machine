"""
A perception pipeline that reports one object it was told about on the command line.

Run as its own process so a test can reach it only through the middleware. The reported
label and position come from arguments, so the test that launches it owns the values it
asserts on instead of duplicating them here.
"""

from __future__ import annotations

import argparse
import signal
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from robokudo_msgs.action import Query
from robokudo_msgs.msg import ObjectDesignator
from std_msgs.msg import Header

QUERY_ACTION_NAME = "robokudo/query"
"""
Action this pipeline answers queries on.
"""


@dataclass
class CannedDetectionPipeline:
    """
    Answers every query with the same single object.
    """

    node: Node
    """
    Node the action server is created on.
    """

    class_label: str
    """
    Label to report the object under.

    Empty reproduces a pipeline that localizes without recognizing, which is what the
    plane-and-cluster annotators produce.
    """

    position: tuple[float, float, float]
    """
    Where to report the object.
    """

    frame_id: str
    """
    Frame the reported position is expressed in.
    """

    def __post_init__(self):
        ActionServer(self.node, Query, QUERY_ACTION_NAME, self.answer_query)

    def answer_query(self, goal_handle) -> Query.Result:
        """
        Report the configured object, whatever was asked for.

        :param goal_handle: The query to answer.
        :return: A result holding exactly one object designator.
        """
        goal_handle.succeed()
        pose_stamped = PoseStamped(header=Header(frame_id=self.frame_id))
        (
            pose_stamped.pose.position.x,
            pose_stamped.pose.position.y,
            pose_stamped.pose.position.z,
        ) = self.position
        pose_stamped.pose.orientation.w = 1.0
        return Query.Result(
            res=[ObjectDesignator(type=self.class_label, pose=[pose_stamped])]
        )


def main() -> None:
    """
    Serve queries until interrupted.

    SIGINT stops the executor rather than raising out of the spin, so the process exits
    cleanly when the test that launched it is done.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--class-label",
        default="",
        help="Label to report; omit for a pipeline that localizes without recognizing.",
    )
    parser.add_argument("--position", required=True, type=float, nargs=3)
    parser.add_argument("--frame-id", default="map")
    arguments = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("perception_pipeline_stand_in")
    CannedDetectionPipeline(
        node=node,
        class_label=arguments.class_label,
        position=tuple(arguments.position),
        frame_id=arguments.frame_id,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    signal.signal(signal.SIGINT, lambda signal_number, frame: executor.shutdown())
    executor.spin()


if __name__ == "__main__":
    main()
