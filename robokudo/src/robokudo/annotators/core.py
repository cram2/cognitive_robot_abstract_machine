"""Core annotator classes for RoboKudo.

This module provides base classes for implementing annotators in RoboKudo.

:Classes:
    * BaseAnnotator - Core functionality for CAS access, tree structure, and GUI result handling
    * ThreadedAnnotator - Extends BaseAnnotator for long-running operations that should not block the main thread

:Features:
    * Common Analysis Structure (CAS) access
    * Tree structure navigation
    * GUI result handling
    * Thread-safe operation
    * Variable publishing
    * Action client integration
"""

import logging
import threading
import time
from collections import defaultdict  # noqa: B027ons import defaultdict
from concurrent.futures import Future

from py_trees.behaviour import Behaviour
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from sensor_msgs.msg import JointState
from typing_extensions import Union, List, Dict, Callable, Tuple, Type, Optional, Any

from robokudo.annotator_parameters import AnnotatorPredefinedParameters
from robokudo.annotators.outputs import (
    AnnotatorOutputs,
    AnnotatorOutputPerPipelineMap,
    AnnotatorOutputStruct,
)
from robokudo.cas import CAS, CASViews
from robokudo.defs import PACKAGE_NAME
import robokudo.pipeline  # Work around circular import
from robokudo.types.core import Annotation
from robokudo.utils.tree import find_parent_of_type

"""
This module contains superclasses for annotators.  the two main superclasses are Base- and ThreadedAnnotator.
The first superclass provides convenience methods to access the CAS, 
the tree structure as well as GUI result related functionality.
The threaded annotator provides functionality to call long-running annotators without blocking the main thread.
"""


class BaseAnnotator(Behaviour):
    """Base class for all RoboKudo annotators.

    This class provides core functionality for CAS access, tree structure navigation,
    and GUI result handling. It serves as the foundation for implementing annotators
    in the RoboKudo framework.
    """

    class Descriptor:
        """Configuration descriptor for annotator parameters and capabilities.

        This class defines the structure for specifying annotator parameters
        and capabilities. It provides access to global parameters through
        the Parameters class.
        """

        class Parameters:
            """Container for annotator parameters.

            This class provides access to both annotator-specific and global
            parameters. Global parameters are accessed through the global_
            prefix.
            """

            None

        class Capabilities:
            """Defines the input and output capabilities of an annotator."""

            def __init__(self) -> None:
                """Initialize capabilities with empty input/output definitions."""

                self.inputs = None
                """Input types accepted by the annotator"""

                self.outputs = None
                """Output types produced by the annotator"""

        def __init__(self) -> None:
            """Initialize the descriptor with parameters and capabilities."""
            param_class = self.Parameters

            # overwrite __getattr__ in current 'Parameters' class
            def get_attr_with_predefined_parameters(self, name: str) -> Any:
                """Custom attribute getter for accessing global parameters.

                :param name: Name of the parameter to access
                :return: Value of the global parameter
                :raises AttributeError: If parameter name is unknown
                """
                if name.startswith("global_"):
                    return getattr(
                        AnnotatorPredefinedParameters,
                        name,
                    )
                else:
                    raise AttributeError(f"Unknown parameter with name <{name}>.")

            param_class.__getattr__ = get_attr_with_predefined_parameters

            self.parameters = param_class()
            self.capabilities = self.Capabilities()

    def __init__(
        self,
        name: str = "Annotator",
        descriptor: "BaseAnnotator.Descriptor" = Descriptor(),
        ros_pkg_name: str = PACKAGE_NAME,
    ) -> None:
        """Initialize the BaseAnnotator.

        Performs minimal one-time initialization of the annotator.

        :param name: Name of the annotator instance
        :param descriptor: Configuration descriptor
        :param ros_pkg_name: Name of the ROS package
        """
        super().__init__(name)
        # self.rk_logger = get_logger(ros_pkg_name)
        self.rk_logger = logging.getLogger(ros_pkg_name)

        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)
        # self.rk_logger.debug("%s" % (get_line_context(self)))
        self.descriptor = descriptor

    def setup(self, **kwargs: Any) -> bool:
        """Perform delayed initialization.

        This method is called for initialization tasks after the constructor has been called.
        It can be used for setup calls that require ROS to be running,
        such as setting up publishers/subscribers or drivers.
        """
        self.rk_logger.debug("%s.setup()" % self.__class__.__name__)
        return True

    def initialise(self) -> None:
        """Initialize the annotator state.

        Called when first tick is received and anytime status is not running.
        """
        self.rk_logger.debug("%s.initialise()" % self.__class__.__name__)

    def update(self) -> Status:
        """Update the annotator state.

        Called every time the behavior is ticked.

        :return: Status of the behavior after update
        """
        return Status.SUCCESS

    def terminate(self, new_status: Status) -> None:
        """Handle annotator termination.

        Called whenever behavior switches to a non-RUNNING state.

        :param new_status: New status of the behavior (SUCCESS, FAILURE, or INVALID)
        """
        self.rk_logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )

    def key_callback(self, key: int) -> None:
        """Handle keyboard input events.

        :param key: Key code of the pressed key
        """
        self.rk_logger.debug("%s.keyCallback() got %i" % (self.__class__.__name__, key))

    def mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: int
    ) -> None:
        """Handle mouse interaction events in the visualizer window.

        :param event: The event type (see cv2.EVENT_* constants)
        :param x: The x coordinate of the mouse click
        :param y: The y coordinate of the mouse click
        :param flags: Flags passed by OpenCV
        :param param: Parameters passed by OpenCV
        """
        self.rk_logger.debug(
            "%s.mouseCallback() got %i at (%i,%i)"
            % (self.__class__.__name__, event, x, y)
        )

    def get_cas(self) -> CAS:
        """Get the CAS from the parent RoboKudo pipeline.

        :return: The Common Analysis Structure (CAS) instance or None if not found
        """
        return self.get_parent_pipeline().cas

    def get_parent_pipeline(self) -> Optional["robokudo.pipeline.Pipeline"]:
        """Get the pipeline containing this annotator.

        :return: The parent pipeline instance or None if not found
        """
        return find_parent_of_type(self, robokudo.pipeline.Pipeline)

    def get_annotator_outputs(self) -> AnnotatorOutputs:
        """Get the outputs container for all annotators in the current pipeline.

        :return: Container for all annotator outputs in the pipeline
        :raises AssertionError: If output container types are invalid
        """
        blackboard = Blackboard()
        annotator_output_pipeline_map_buffer = blackboard.get(
            "annotator_output_pipeline_map_buffer"
        )
        assert isinstance(
            annotator_output_pipeline_map_buffer,
            AnnotatorOutputPerPipelineMap,
        )

        pipeline = self.get_parent_pipeline()
        annotator_outputs = annotator_output_pipeline_map_buffer.map[pipeline.name]
        assert isinstance(annotator_outputs, AnnotatorOutputs)
        return annotator_outputs

    def get_annotator_output_struct(
        self,
    ) -> AnnotatorOutputStruct:
        """Get the output structure for this specific annotator.

        Dynamically adds the annotator to the output structure if not already present.

        :return: Output structure for this annotator
        """
        annotator_outputs = self.get_annotator_outputs()
        # This check adds dynamically added annotators to the output struct that have
        # not been in the tree during the setup() of it.
        if self.name not in annotator_outputs.outputs:
            self.rk_logger.debug(
                f"Annotator {self.name} was not in AnnotatorOutput struct. Adding it now."
            )
            self.add_self_to_annotator_output_struct()
        return self.get_annotator_outputs().outputs[self.name]

    def add_self_to_annotator_output_struct(self) -> None:
        """Add this annotator to the output structure.

        Called when the annotator is not yet present in the output structure.
        """
        blackboard = Blackboard()
        annotator_output_pipeline_map_buffer = blackboard.get(
            "annotator_output_pipeline_map_buffer"
        )
        annotator_output_pipeline_map_buffer.map[
            self.get_parent_pipeline().name
        ].init_annotator(self.name)

    def get_class_name(self) -> str:
        """Get the class name of this annotator.

        Unlike self.name which can be customized, this always returns the actual class name.

        :return: The class name of this annotator
        """
        return type(self).__name__

    def get_data_from_analysis_scope(
        self, analysis_scope: List[Union[str, Type]]
    ) -> Dict[Union[str, Type], Any]:
        """Look up data to analyze based on a list of Types or String constants.

        Retrieves data from the CAS based on specified types or CASView constants.
        Can be used to analyze data at different levels (Scene, ObjectHypothesis, etc.)
        based on the current perception task context.

        Example::

            get_data_from_analysis_scope([CASViews.COLOR_IMAGE])
            =>
            {
                CASViews.COLOR_IMAGE : numpy.array([.. image data...])
            }

            get_data_from_analysis_scope([robokudo.types.scene.ObjectHypothesis])
            =>
            {
                robokudo.types.scene.ObjectHypothesis : [ObjectHypothesis1, ... ObjectHypothesisN]
            }

        :param analysis_scope: List of data types to target (CASView attributes or Annotation subclasses)
        :return: Dict mapping requested types to corresponding data
        :raises ValueError: If result would be empty
        """
        result: Dict[Union[str, Type], Any] = {}

        for content in analysis_scope:
            # CASViews are string values
            if isinstance(content, str):
                if content not in CASViews.__dict__.values():
                    self.rk_logger.warning(
                        "You have passed a string that is not defining a valid CASView from robokudo.cas.CASViews"
                    )
                result[content] = self.get_cas().get(content)
            elif issubclass(content, Annotation):
                # We assume that typically only few annotations are requested here
                # Otherwise we should iterate over all annotations and filter once to save resources
                result[content] = self.get_cas().filter_annotations_by_type(content)
            else:
                self.rk_logger.error(
                    f"Couldn't lookup data {content} - Unsupported type"
                )

        if len(result) == 0:
            raise ValueError(
                f"Result list after scope look up is empty. Can't continue processing.Input was: {analysis_scope}"
            )

        return result

    def init_time_recording(self) -> None:
        """Initialize the time recording dictionary for performance measurement."""
        if not hasattr(self, "_times"):
            setattr(self, "_times", defaultdict(list))

    def set_time_recording(self, value: float, func: str = "update") -> None:
        """Record timing information for a function.

        :param value: The timing value to record
        :param func: The function name to associate with the timing, defaults to "update"
        """
        timing_dict = getattr(self, "_times")
        timing_dict[func] = value

    #########################################################
    # Data publishing handling                               #
    #                                                        #
    # Collect float variables in your Annotators             #
    # that you want to analyze with external tools.          #
    # The following methods handle to publish these via ROS. #
    ##########################################################

    def shall_publish_variables(self) -> bool:
        """Check if variables should be published based on descriptor settings.

        :return: True if variables should be published, False otherwise
        """
        return (
            hasattr(self.descriptor.parameters, "publish_variables")
            and self.descriptor.parameters.publish_variables
        )

    def setup_published_variables(self) -> None:
        """Setup the ROS Publisher and internal data structure if data shall be published

        :return: None
        """
        if self.shall_publish_variables():
            # TODO Rewrite so that the pipeline name is also included and updated on-the-fly
            self.variable_publisher = self.create_publisher(
                f"~{self.name}/published_variables",
                JointState,
                queue_size=10,
            )
            self.published_variables: Dict[str, float] = dict()

    def update_published_variable(self, name: str, value: float) -> None:
        """Store a value that shall be published whenever self.publish_variables() is called.

        :param name: Identifier of the variable
        :param value: The actual value (must be a float)
        """
        try:
            self.published_variables[name] = value
        except Exception as e:
            pass

    def setup_done_for_published_variables(self) -> bool:
        """Check if variable publishing setup is complete.

        :return: True if setup is complete, False otherwise
        """
        return hasattr(self, "published_variables") and isinstance(
            self.published_variables, dict
        )

    def publish_variables(self) -> None:
        """Publish all collected variable updates at once.

        Use this function as a decorator (i.e. @robokudo.utils.decorators.publish_variables) on your update or
        compute method to automatically publish variables whenever the method is returning.

        This method will only work if you've set self.publish_variables = True in the Descriptor of your Annotator
        """
        if self.shall_publish_variables():
            if not self.setup_done_for_published_variables():
                self.setup_published_variables()

            # Create a joint state message to have a common data structure for all values
            msg = JointState()
            msg.header.stamp = (
                self.get_cas().get(CASViews.CAM_INFO).header.stamp
            )  # This is always the time of recording the data.

            # msg.header.stamp = rospy.Time(self.get_cas().timestamp) # This might be the current time!
            for key, value in self.published_variables.items():
                msg.name.append(key)
                msg.position.append(value)

            try:
                self.variable_publisher.publish(msg)
            except Exception as e:
                self.rk_logger.warning(f"Caught exception {e} while publish_variables")


# This class is based on https://stackoverflow.com/a/32913813
class Worker(object):
    """Worker class for executing functions asynchronously.

    This class provides a way to run functions in a separate thread and
    optionally execute a callback when the function completes.
    """

    def __init__(self, fn: Callable, args: Tuple = ()) -> None:
        """Initialize the worker.

        :param fn: The function to execute
        :param args: Arguments to pass to the function, defaults to ()
        """
        self.future: Future = Future()
        self._fn: Callable = fn
        self._args: Tuple = args

    def start(self, callback: Optional[Callable] = None) -> threading.Thread:
        """Start executing the function in a separate thread.

        :param callback: Function to call when execution completes, defaults to None
        """
        self._callback = callback
        self.future.set_running_or_notify_cancel()
        thread = threading.Thread(target=self.run, args=())
        thread.start()
        return thread

    def run(self) -> None:
        """Execute the function and store its result in the future object."""
        try:
            self.future.set_result(self._fn(*self._args))
        except BaseException as e:
            self.future.set_exception(e)

        if self._callback:
            self._callback(self.future.result())


class ThreadedAnnotator(BaseAnnotator):
    """Base class for annotators that perform long-running operations.

    This class extends BaseAnnotator to support operations that should not block
    the main thread. It provides functionality to run computations asynchronously
    and handle their results.
    """

    def __init__(
        self,
        name: str = "ThreadedAnnotator",
        descriptor: BaseAnnotator.Descriptor = BaseAnnotator.Descriptor(),
    ) -> None:
        """Initialize the ThreadedAnnotator.

        :param name: Name of the annotator instance, defaults to "ThreadedAnnotator"
        :param descriptor: Configuration descriptor, defaults to BaseAnnotator.Descriptor()
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)
        self.compute_worker: Optional[Worker] = None
        self.compute_worker_started: bool = False
        self.compute_worker_thread: Optional[threading.Thread] = None

    def initialise(self) -> None:
        """Initialize the annotator state.

        Called when first tick is received and anytime status is not running.
        """
        self.rk_logger.debug("%s.initialise()" % self.__class__.__name__)
        self.compute_worker = Worker(self.compute)

    def update(self) -> Status:
        """Update the annotator state.

        Manages the asynchronous computation and returns appropriate status:
        - RUNNING if computation is ongoing
        - SUCCESS/FAILURE based on computation result
        - INVALID if computation failed with an exception

        :return: Status of the behavior after update
        """
        self.rk_logger.debug("%s.update()" % self.__class__.__name__)
        if self.compute_worker.future.running():
            return Status.RUNNING
        elif not self.compute_worker_started:
            self.compute_worker = Worker(self.compute)
            self.compute_worker_thread = self.compute_worker.start()
            self.compute_worker_started = True
            return Status.RUNNING

        self.compute_worker_thread.join()
        self.compute_worker_started = False
        # exc = self.compute_worker.future.exception()
        result = self.compute_worker.future.result()

        if result is None:
            self.rk_logger.error(
                "Your Annotator didn't return a Status. Please fix. Will return SUCCESS."
            )
            return Status.SUCCESS

        return result

    def compute(self) -> Status:
        """Perform the main computation of the annotator.

        This method should be overridden by subclasses to implement the actual
        computation logic.
        """
        self.rk_logger.debug(
            "%s.compute(): Start doing the heavy stuff" % self.__class__.__name__
        )
        time.sleep(0.5)  # in seconds
        self.rk_logger.debug(
            "%s.compute(): Stop doing the heavy stuff" % self.__class__.__name__
        )
        return Status.SUCCESS

    def terminate(self, new_status: Status) -> None:
        """Handle annotator termination.

        Called whenever behavior switches to !RUNNING state.
        new_status can be SUCCESS, FAILURE, or INVALID

        :param new_status: New status of the behavior (SUCCESS, FAILURE, or INVALID)
        """
        self.rk_logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )
