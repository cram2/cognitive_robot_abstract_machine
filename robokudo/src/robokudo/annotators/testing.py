"""
Testing annotators for RoboKudo.

This module provides annotators for testing and debugging purposes.
It supports:

* Empty annotator for baseline testing
* Failing annotator for error handling testing
* Simulated processing delays
* Visual output generation
* Status feedback testing

The module is used for:

* System testing
* Error handling verification
* Performance testing
* Debug visualization
"""

import copy
import time

import cv2
import numpy
import py_trees

import robokudo
import robokudo.types.scene
import robokudo.annotators.core
from robokudo.cas import CASViews
from robokudo.utils.error_handling import catch_and_raise_to_blackboard


class SlowAnnotator(robokudo.annotators.core.ThreadedAnnotator):
    """
    A slow annotator that demonstrates long-running processing using ThreadedAnnotator.

    This annotator simulates a time-consuming process by adding a configurable delay
    and generating visual output. It is useful for:

    * Testing thread handling
    * Performance monitoring
    * Timeout behavior verification
    * Visual feedback testing

    :ivar sleep_in_s: Duration to sleep in seconds
    :type sleep_in_s: float
    """

    def __init__(self, name: str = "SlowAnnotator", sleep_in_s: float = 1.0) -> None:
        """
        Initialize the slow annotator.

        :param name: Name of the annotator instance
        :param sleep_in_s: Sleep duration in seconds
        """
        super().__init__(name)

        self.sleep_in_s = sleep_in_s
        """Sleep duration in seconds"""

    def compute(self) -> py_trees.common.Status:
        """
        Perform the main computation with artificial delay.

        This method:

        * Retrieves the color image from CAS
        * Adds timestamp and visual markers
        * Sleeps for configured duration
        * Sets feedback message

        :return: Success status after completion
        """
        self.rk_logger.debug(
            "%s.compute(): Start doing the heavy stuff" % self.__class__.__name__
        )

        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)
        vis = copy.deepcopy(self.color)

        import datetime

        vis = cv2.putText(
            vis,
            str(datetime.datetime.now()),
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        vis = cv2.rectangle(vis, (400, 400), (650, 650), (0, 0, 255), 2)
        self.get_annotator_output_struct().set_image(vis)

        time.sleep(self.sleep_in_s)  # in s

        self.rk_logger.debug(
            "%s.compute(): Stop doing the heavy stuff" % self.__class__.__name__
        )
        self.feedback_message = "SlowAnnotator was successful."

        return py_trees.common.Status.SUCCESS


class EmptyAnnotator(robokudo.annotators.core.BaseAnnotator):
    """
    Basic annotator that generates empty visual output.

    This annotator creates a black image and sets it as output.
    Used for testing visualization and output handling.
    """

    def __init__(self, name: str = "EmptyAnnotator", sleep_in_s: float = 1) -> None:
        """Initialize the empty annotator.

        :param name: Annotator name, defaults to "EmptyAnnotator"
        :param sleep_in_s: Sleep duration in seconds, defaults to 1
        """
        super().__init__(name)

    def update(self) -> py_trees.common.Status:
        """
        Generate empty visual output.

        Creates a black image and sets it as the annotator output.

        :return: SUCCESS status
        """
        vis = numpy.zeros((640, 480, 3), dtype="uint8")
        self.get_annotator_output_struct().set_image(vis)

        return py_trees.common.Status.SUCCESS


class FailingAnnotator(robokudo.annotators.core.ThreadedAnnotator):
    """
    Annotator that simulates failures for testing with a long-running annotator using the ThreadedAnnotator class.

    This annotator alternates between success and failure states,
    with configurable delays. Used for testing error handling.
    """

    def __init__(self, name: str = "FailingAnnotator") -> None:
        """Initialize the failing annotator.

        :param name: Annotator name, defaults to "FailingAnnotator"
        """
        super().__init__(name)

        self.counter = 0
        """Counter for alternating between success and failure states"""

    def initialise(self) -> None:
        """Initialize the annotator state.

        Called on first tick and whenever status changes from non-running.
        """
        super().initialise()

    def compute(self) -> py_trees.common.Status:
        """Simulate processing with alternating success/failure.

        Sleeps for a fixed duration and alternates between success and failure
        states based on an internal counter.

        :return: SUCCESS or FAILURE status
        :rtype: py_trees.Status
        """
        self.rk_logger.debug(
            "%s.compute(): Start doing the heavy stuff" % self.__class__.__name__
        )
        time.sleep(0.5)  # in s

        if self.counter == 1:
            self.feedback_message.put("I failed horribly!")
            self.counter = 0
            return py_trees.Status.FAILURE
        else:
            self.counter += 1
            self.feedback_message_queue.put("Action successful")
            return py_trees.Status.SUCCESS

        self.rk_logger.debug(
            "%s.compute(): Stop doing the heavy stuff" % self.__class__.__name__
        )


class FakeCollectionReaderAnnotator(robokudo.annotators.core.BaseAnnotator):
    """
    A simulated collection reader for testing pipeline behavior.

    This annotator simulates a collection reader by:

    * Waiting for a configurable number of iterations
    * Creating synthetic CAS data
    * Managing feedback messages
    * Testing pipeline flow control

    Used for:

    * Pipeline integration testing
    * Flow control verification
    * Feedback message handling
    * CAS creation testing
    """

    def __init__(self, name: str = "FakeCollectionReader") -> None:
        """
        Initialize the fake collection reader.

        :param name: Name of the annotator instance, defaults to "FakeCollectionReader"
        """
        super().__init__(name)
        self.rk_logger.debug("%s.__init__()" % (self.__class__.__name__))

        self.collection_readers = [self.descriptor]
        """List of collection reader descriptors"""
        # self.rk_logger.debug("%s" % (get_line_context(self)))

    def setup(self, timeout: float) -> bool:
        """
        Set up the collection reader.

        :param timeout: Maximum time to wait for setup completion
        :return: True if setup successful
        """
        self.rk_logger.debug("%s.setup()" % self.__class__.__name__)
        return True

    def initialise(self) -> None:
        """
        Initialize the reader state.

        Called on first tick and whenever status changes from non-running.
        Resets counter and clears feedback messages for all children in sequence.
        """
        self.rk_logger.debug("%s.initialise()" % (self.__class__.__name__))
        self.counter = 0

        # Clear all feedback messages when Collection Reader starts over
        assert isinstance(self.parent, py_trees.composites.Sequence)
        for child in self.parent.children:
            child.feedback_message = ""

    def update(self) -> py_trees.common.Status:
        """
        Update the reader state and generate synthetic data.

        This method:

        * Increments internal counter
        * Creates new CAS data after 3 iterations
        * Sets appropriate feedback messages
        * Updates processing status

        :return: Current processing status (RUNNING or SUCCESS)
        """
        self.counter += 1
        new_status = (
            py_trees.common.Status.SUCCESS
            if self.counter == 3
            else py_trees.common.Status.RUNNING
        )
        if new_status == py_trees.common.Status.SUCCESS:
            self.feedback_message = "Got sensor data!"
            # Create a new CAS
            blackboard = py_trees.blackboard.Blackboard()
            cas = robokudo.cas.CAS()
            cas.percepts = [time.time()]
            blackboard.set("CAS", cas)

        else:
            self.feedback_message = "Waiting for incoming sensor data ..."
        self.rk_logger.debug(
            "%s.update()[%s->%s][%s]"
            % (self.__class__.__name__, self.status, new_status, self.feedback_message)
        )
        return new_status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """
        Clean up when transitioning to non-running state.

        :param new_status: New status being transitioned to
        """
        self.rk_logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )


class ScopedAnnotator(robokudo.annotators.core.BaseAnnotator):
    """
    Demonstrates the usage of analysis scopes in annotators.

    This annotator shows how to:

    * Define and use analysis scopes
    * Handle different data types in the scope
    * Process object hypotheses and scene data
    * Manage scope parameters

    The annotator can operate in two modes:

    * Object Hypothesis Analysis - processes individual object hypotheses
    * Scene Analysis - processes scene-level data

    .. note::
        This is primarily a demonstration annotator for educational purposes.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Descriptor class defining the annotator's parameters."""

        class Parameters:
            """Parameter class for the ScopedAnnotator."""

            def __init__(self) -> None:
                """Initialize parameters with default analysis scope."""
                self.analysis_scope = [CASViews.COLOR_IMAGE, CASViews.CLOUD]
                """ List of data types to analyze """

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "ScopedAnnotator",
        descriptor: "ScopedAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        """
        Initialize the scoped annotator.

        :param name: Name of the annotator instance, defaults to "ScopedAnnotator"
        :param descriptor: Descriptor instance with parameters, defaults to Descriptor()
        """
        super().__init__(name, descriptor)

    @catch_and_raise_to_blackboard
    def update(self) -> py_trees.common.Status:
        """
        Process data based on the current analysis scope.

        This method:

        * Retrieves data based on the analysis scope
        * Determines the processing mode (OH or Scene analysis)
        * Processes object hypotheses if present
        * Handles scene-level analysis otherwise

        :return: Processing status
        :raises: Any exceptions are caught and raised to the blackboard
        """
        # self.descriptor.parameters.analysis_scope = [robokudo.types.scene.ObjectHypothesis]  # test overwrite
        # self.descriptor.parameters.analysis_scope = [CASViews.COLOR_IMAGE, CASViews.CLOUD]  # test overwrite
        self.descriptor.parameters.analysis_scope = []  # test overwrite => error case

        data_to_analyze = self.get_data_from_analysis_scope(
            self.descriptor.parameters.analysis_scope
        )

        # Your Annotator now has to decide which kinds of data it expects and how it would process them
        if robokudo.types.scene.ObjectHypothesis in data_to_analyze:
            print(f"Annotator is in OH-Analysis mode")
            object_hypotheses = data_to_analyze[robokudo.types.scene.ObjectHypothesis]
            for object_hypothesis in object_hypotheses:
                assert isinstance(
                    object_hypothesis, robokudo.types.scene.ObjectHypothesis
                )
                print(f"OH ID: {object_hypothesis.id}")
        else:
            print(f"Annotator is in Scene-Analysis mode")
            cloud = data_to_analyze[CASViews.CLOUD]
            print(f"Cloud points: {len(cloud.points)}")

        return py_trees.common.Status.SUCCESS
