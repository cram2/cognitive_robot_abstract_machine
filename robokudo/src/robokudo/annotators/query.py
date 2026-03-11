"""
Query handling annotator for RoboKudo.

This module provides an annotator that handles queries from external ROS nodes.
It supports:

* Spawning an action server for query handling
* Type-agnostic query processing
* Asynchronous query response
* Integration with ROS action system
* CAS annotation with query data

The module is used for:

* External system integration
* Query-based perception
* Interactive perception tasks
* Asynchronous data exchange
"""

import logging
import queue
import threading
import time
import geometry_msgs.msg
import py_trees
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from typing_extensions import Any, Optional

import robokudo
import robokudo.defs
import robokudo.types.annotation
import robokudo.types.cv
import robokudo.types.scene
import robokudo_msgs.msg
import robokudo.annotators.core
from robokudo.cas import CASViews
from robokudo.identifier import BBIdentifier
from robokudo.utils.annotation_conversion import (
    SemanticColor2ODConverter,
    Classification2ODConverter,
    StampedPose2ODConverter,
    BoundingBox3DForShapeSizeConverter,
    Pose2ODConverter,
    Position2ODConverter,
    StampedPosition2ODConverter,
    Shape2ODConverter,
    Cuboid2ODConverter,
    Sphere2ODConverter,
    Location2ODConverter,
)
from robokudo.utils.error_handling import (
    has_blackboard_exception,
    get_blackboard_exception,
    clear_blackboard_exception,
)
from robokudo.utils.query import QueryHandler
from robokudo_msgs.action import Query


class QueryAnnotator(robokudo.annotators.core.BaseAnnotator):
    """Handle external queries through ROS action server.

    This Annotator spawns an Action Server that listens for Queries from external ROS nodes.
    It will then annotate the CAS and put the Query into CASViews.QUERY.
    The Annotator and the Actionserver are type-agnostic, which means that you are not bound
    to a specific type of query. You can pass these from your AE to this QueryAnnotator.
    """

    def __init__(self, name: str = "QueryAnnotator") -> None:
        """Initialize the query annotator.

        :param name: Annotator name, defaults to "QueryAnnotator"
        """
        super().__init__(name=name)

        self.feedback_instance = Query.Feedback()
        self.result_instance = Query.Result()

        self.action_server = None
        """Action server placeholder."""

    def setup(self, **kwargs: Any) -> None:
        """Ensure that the Query Server is spawned early on, directly after PPT creation."""
        self.initialise()

    def initialise(self) -> None:
        """Initialize query handling.

        Sets up the action server if not already initialized.
        Stores server instance on blackboard for access by other nodes.
        """
        self.rk_logger.debug(f"{self.__class__.__name__}.initialise()")
        blackboard = py_trees.blackboard.Blackboard()
        if not blackboard.exists(BBIdentifier.QUERY_SERVER):
            query_action_server = QueryActionServer(name="query")
            blackboard.set(BBIdentifier.QUERY_SERVER, query_action_server)

        blackboard.set(BBIdentifier.QUERY_SERVER_IN_PIPELINE, True)

        self.action_server = blackboard.get(BBIdentifier.QUERY_SERVER)

    def update(self) -> py_trees.common.Status:
        """Process new queries and update CAS.

        Checks for new queries from action server and updates CAS if found.
        Provides feedback about query status.

        :return: SUCCESS if query processed, RUNNING if waiting
        """
        self.rk_logger.debug(f"{self.__class__.__name__}.update()")

        assert (
            self.action_server is not None
        ), "Action server should be initialized by now."
        query = self.action_server.new_query
        self.rk_logger.debug(f"self.action_server.new_query: {query}")

        if query:
            self.feedback_message = f"Query: {query}"
            self.get_cas().set(CASViews.QUERY, query)
            self.action_server.start_processing()
            # self.publish_feedback()

            return py_trees.common.Status.SUCCESS

        self.feedback_message = "Waiting for query"
        return py_trees.common.Status.RUNNING


class QueryFeedback(robokudo.annotators.core.BaseAnnotator):
    """A test class which simply generates a fixed-string feedback."""

    def __init__(self, name: str = "QueryFeedback", feedback_str: str = "") -> None:
        """Initialize query feedback generator.

        :param name: Annotator name
        :param feedback_str: Feedback string to send to the client
        """
        super().__init__(name=name)

        self.feedback_str = feedback_str
        """Feedback string to send to the client."""

    def update(self) -> py_trees.common.Status:
        self.rk_logger.debug(f"{self.__class__.__name__}.update()")
        QueryHandler.send_feedback_str(self.feedback_str)

        return py_trees.common.Status.SUCCESS


class QueryFeedbackAndCount(robokudo.annotators.core.BaseAnnotator):
    """
    A test class which simply counts up until a fixed number.
    Until this number is reached, a pre-defined status is returned.
    """

    def __init__(
        self,
        name: str = "QueryFeedback",
        count_until: int = 20,
        return_code: py_trees.common.Status = py_trees.common.Status.RUNNING,
    ) -> None:
        """Initialize query feedback generator.

        :param name: Annotator name
        :param count_until: Number until which to count before stopping
        :param return_code: The return code to return while still counting
        """
        super().__init__(name=name)

        self.i = 0
        """The current count"""

        self.count_until = count_until
        """The number until which to count"""

        self.return_code = return_code
        """The return code to return while still counting"""

    def update(self) -> py_trees.common.Status:
        self.rk_logger.debug(f"{self.__class__.__name__}.update()")
        QueryHandler.send_feedback_str(f"Count: {self.i}")
        self.i = self.i + 1
        if self.i > self.count_until:
            self.i = 0
            return py_trees.common.Status.SUCCESS
        else:
            return self.return_code


class QueryReply(robokudo.annotators.core.BaseAnnotator):
    """
    A test class which simply generates an empty Query Answer to check
    if the Action server can reply properly.
    Create a single, empty Object Designator that will be sent to the caller.
    """

    def __init__(self, name: str = "QueryReply"):
        """Initialize query reply generator.

        :param name: Annotator name, defaults to "QueryReply"
        """
        super().__init__(name=name)

    def initialise(self) -> None:
        """Initialize reply generator."""
        self.rk_logger.debug(f"{self.__class__.__name__}.initialise()")

    def update(self) -> py_trees.common.Status:
        """Generate test query response.

        Creates an empty ObjectDesignator with a test pose and adds it to blackboard.

        :return: SUCCESS after generating response
        """
        self.rk_logger.debug(f"{self.__class__.__name__}.update()")
        result = Query.Result()
        od = robokudo_msgs.msg.ObjectDesignator()

        import geometry_msgs

        pose_stamped = geometry_msgs.msg.PoseStamped()

        # Explicitly cast to float
        pose_stamped.pose.position.x = float(1)
        pose_stamped.pose.position.y = float(2)
        pose_stamped.pose.position.z = float(3)

        pose_stamped.pose.orientation.x = float(0)
        pose_stamped.pose.orientation.y = float(0)
        pose_stamped.pose.orientation.z = float(0)
        pose_stamped.pose.orientation.w = float(1)

        od.pose.append(pose_stamped)
        result.res = [od]

        QueryHandler.send_answer(result)

        return py_trees.common.Status.SUCCESS


class GenerateQueryResult(robokudo.annotators.core.BaseAnnotator):
    """
    This class reads in the annotations done by the previous Annotators
    and generates Object Designators from them.
    These will be placed into the Blackboard so that a running Query Action Server can pick the information
    up and send it as a query reply.
    """

    def __init__(self, name: str = "GenerateQueryResult"):
        """Initialize query result generator.

        :param name: Annotator name, defaults to "GenerateQueryResult"
        """
        self.rk_logger = logging.getLogger(robokudo.defs.PACKAGE_NAME)

        self.color_converter = SemanticColor2ODConverter()
        self.class_converter = Classification2ODConverter()
        self.position_converter = Position2ODConverter()
        self.stamped_position_converter = StampedPosition2ODConverter()
        self.pose_converter = Pose2ODConverter()
        self.stamped_pose_converter = StampedPose2ODConverter()
        self.shape_converter = Shape2ODConverter()
        self.cuboid_converter = Cuboid2ODConverter()
        self.sphere_converter = Sphere2ODConverter()
        self.location_converter = Location2ODConverter()
        self.bb_size_converter = BoundingBox3DForShapeSizeConverter()

        self.type_converter = {
            robokudo.types.annotation.SemanticColor: self.color_converter,
            robokudo.types.annotation.Classification: self.class_converter,
            robokudo.types.annotation.PoseAnnotation: self.pose_converter,
            robokudo.types.annotation.StampedPoseAnnotation: self.stamped_pose_converter,
            robokudo.types.annotation.PositionAnnotation: self.position_converter,
            robokudo.types.annotation.StampedPositionAnnotation: self.stamped_position_converter,
            robokudo.types.annotation.Shape: self.shape_converter,
            robokudo.types.annotation.Cuboid: self.cuboid_converter,
            robokudo.types.annotation.Sphere: self.sphere_converter,
            robokudo.types.annotation.LocationAnnotation: self.location_converter,
            robokudo.types.cv.BoundingBox3D: self.bb_size_converter,
            robokudo.types.annotation.BoundingBox3DAnnotation: self.bb_size_converter,
        }

        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        """Generate query result from current CAS annotations.

        For each ObjectHypothesis in CAS:
        * Creates ObjectDesignator
        * Adds color information if available
        * Adds classification if available
        * Adds pose information if available
        * Packages into query result

        :return: SUCCESS after generating result
        """
        if QueryHandler.preempt_requested():
            QueryHandler.acknowledge_preempt_request()
            self.rk_logger.warning("Acknowledge preempt")
            return py_trees.common.Status.FAILURE

        cas = self.get_cas()
        annotations = cas.annotations
        object_hypotheses_count = 0
        query_result = []
        result = Query.Result()

        for annotation in annotations:
            if not isinstance(annotation, robokudo.types.scene.ObjectHypothesis):
                continue

            object_designator = robokudo_msgs.msg.ObjectDesignator()

            for oh_annotation in annotation.annotations:
                converter = self.type_converter.get(type(oh_annotation), None)
                if converter is None:
                    self.rk_logger.warning(
                        f"no converter available for annotation type {type(oh_annotation)}, skipping annotation."
                    )
                    continue
                if not converter.can_convert(oh_annotation):
                    self.rk_logger.warning(
                        f"converter for {type(oh_annotation)} available but cannot convert, skipping annotation."
                    )
                    continue

                converter.convert(oh_annotation, cas, object_designator)
                self.rk_logger.info(
                    f"converted {type(oh_annotation)} on OH {object_hypotheses_count}"
                )

            query_result.append(object_designator)
            object_hypotheses_count += 1

        result.res = query_result
        QueryHandler.send_answer(result)

        self.feedback_message = (
            f"Send result for {object_hypotheses_count} object hypotheses"
        )
        return py_trees.common.Status.SUCCESS


class QueryActionServer(Node):
    """ROS action server for handling perception queries.

    Action server that listens for queries and executes them by checking blackboard
    for results generated by QueryAnnotator and QueryReply.
    """

    def __init__(
        self,
        name: str,
        feedback_instance: Query.Feedback = Query.Feedback(),
        result_instance: Query.Result = Query.Result(),
        action_type: Query = Query,
    ) -> None:
        super().__init__(name, namespace="robokudo")
        self._action_name = name
        """Name of the ROS action"""

        self._as = ActionServer(
            self,
            action_type,
            self._action_name,
            execute_callback=self.execute_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
        )
        """Action server instance"""

        self.feedback_instance: Query.Feedback = feedback_instance or Query.Feedback()
        """Latest feedback message"""

        self.result_instance: Query.Result = result_instance or Query.Result()
        """Latest result message"""

        self.new_query: Optional[robokudo_msgs.action.Query.Goal] = None
        """Latest received query"""

        self.query: Optional[robokudo_msgs.action.Query.Goal] = None
        """Currently processing query"""

        self.reset_bookkeeping_vars()

        self.query_processed_event: threading.Event = threading.Event()
        """Event to signal query processing completion"""

        self.logger = logging.getLogger(robokudo.defs.LOGGING_IDENTIFIER_QUERY)

    def reset_bookkeeping_vars(self) -> None:
        """Reset internal state variables.

        Clears query state and blackboard variables.
        """
        self.query = None
        self.new_query = None
        py_trees.blackboard.Blackboard().set(BBIdentifier.QUERY_ANSWER, None)
        py_trees.blackboard.Blackboard().set(BBIdentifier.QUERY_FEEDBACK, queue.Queue())
        py_trees.blackboard.Blackboard().set(
            BBIdentifier.QUERY_PREEMPT_REQUESTED, False
        )
        py_trees.blackboard.Blackboard().set(BBIdentifier.QUERY_PREEMPT_ACK, False)

    def goal_cb(self, goal_request: Query.Goal) -> GoalResponse:
        self.logger.info(f"Received new goal: {goal_request}")
        return GoalResponse.ACCEPT

    def cancel_cb(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        self.logger.info(f"Received cancel request:{goal_handle}")
        py_trees.blackboard.Blackboard().set(BBIdentifier.QUERY_PREEMPT_REQUESTED, True)
        return CancelResponse.ACCEPT

    def start_processing(self) -> None:
        """Start processing new query.

        Tell the ActionServer that we are now starting the execution and it can start the monitoring/response process.
        """
        self.logger.info("start_processing called, setting new_query to None.")
        self.new_query = None

    def is_active(self) -> bool:
        """Check if query is being processed.

        :return: True if query active, False otherwise
        """
        return self.query is not None

    async def execute_cb(self, goal_handle: ServerGoalHandle) -> Optional[Query.Result]:
        """Action server execution callback.

        Handles:
        * Query reception and validation
        * Processing status monitoring
        * Preemption requests
        * Error handling
        * Result generation and sending

        :param goal_handle: Query goal from client
        """
        self.logger.info(f"Received query: {goal_handle.request}")
        self.new_query = goal_handle.request
        self.query = goal_handle.request

        self.logger.info("Begin waiting for new_query")
        self.logger.info("Processing query...")

        feedback_queue = py_trees.blackboard.Blackboard().get(
            BBIdentifier.QUERY_FEEDBACK
        )
        self.logger.info("Start watching")

        while rclpy.ok():
            time.sleep(1.0 / 50.0)

            # At least one node in the Tree has to acknowledge the preempt request. This allows the tree
            # to properly shutdown.
            preempt_acknowledge = py_trees.blackboard.Blackboard().get(
                BBIdentifier.QUERY_PREEMPT_ACK
            )
            if goal_handle.is_cancel_requested and preempt_acknowledge:
                self.logger.info("Goal cancel acknowledged by PPT.")
                goal_handle.canceled()
                cancel_result = Query.Result()
                cancel_result.text_result = "Canceled"
                self.reset_bookkeeping_vars()
                return cancel_result

            if has_blackboard_exception():
                exception_text = str(get_blackboard_exception())
                self.logger.error(f"Aborting due to error: {exception_text}")
                clear_blackboard_exception()
                goal_handle.abort()
                abort_result = Query.Result()
                abort_result.text_result = "Aborted"
                self.reset_bookkeeping_vars()
                return abort_result

            try:
                feedback_msg = feedback_queue.get_nowait()
                if isinstance(feedback_msg, Query.Feedback):
                    self.feedback_instance.feedback = (
                        feedback_msg.feedback
                    )  # Adjust based on actual field
                    goal_handle.publish_feedback(self.feedback_instance)
                    self.logger.info(
                        f"Published feedback: {self.feedback_instance.feedback}"
                    )
            except queue.Empty:
                pass

            answer = py_trees.blackboard.Blackboard().get(BBIdentifier.QUERY_ANSWER)
            if answer is not None:
                goal_handle.succeed()
                self.logger.info(f"Send: {answer}")
                self.reset_bookkeeping_vars()
                return answer

        goal_handle.abort()
        return None
