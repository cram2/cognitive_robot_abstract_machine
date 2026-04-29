"""Base visualization interface for RoboKudo pipelines.

This module provides the base visualization interface and shared state management
for RoboKudo pipeline visualization. It implements:

* Observer pattern for state updates
* Shared visualization state
* Visualizer lifecycle management
* Pipeline data access
* Common visualization utilities
"""

import logging

from py_trees.blackboard import Blackboard
from typing_extensions import List, Any, Optional, Set, Type

from robokudo.annotators.outputs import AnnotatorOutputs
from robokudo.defs import PACKAGE_NAME
from robokudo.pipeline import Pipeline


class Visualizer(object):
    """Base class for RoboKudo pipeline visualizers.

    This class provides the foundation for implementing pipeline visualizers. It includes:

    * Observer pattern for state updates
    * Shared visualization state
    * Visualizer lifecycle management
    * Pipeline data access
    * Common visualization utilities

    .. note::
    Do not instantiate this class directly. Use :meth:`new_visualizer_instance`
    to properly register visualizers.
    """

    # Observer pattern from https://en.wikipedia.org/wiki/Observer_pattern
    class Observable:
        """Base class for objects that can be observed.

        This class implements the Observable part of the Observer pattern.
        """

        def __init__(self) -> None:
            """Initialize the observable object."""
            self._observers: List["Visualizer.Observer"] = []

        def register_observer(self, observer: "Visualizer.Observer") -> None:
            """Register an observer to receive notifications.

            :param observer: The observer to register
            """
            self._observers.append(observer)

        def notify_observers(self, *args: Any, **kwargs: Any) -> None:
            """Notify all registered observers."""
            for obs in self._observers:
                obs.notify(self, *args, **kwargs)

    class Observer:
        """Base class for objects that observe state changes.

        This class implements the Observer part of the Observer pattern.
        """

        def notify(
            self, observable: "Visualizer.Observable", *args: Any, **kwargs: Any
        ) -> None:
            """Handle notification of state changes.

            :param observable: The object that sent the notification
            :raises Exception: If not implemented by subclass
            """
            print("Got", args, kwargs, "From", observable)
            # Show the passed data, but terminate. Make this an ABC in the future.
            raise Exception(
                "You need to implement the notify method in your Visualizer to catch update requests."
            )

    class SharedState(Observable):
        """Shared state for single-view visualizers.

        This class manages state shared between visualizers, particularly for
        single-view visualizers that switch between annotators. It uses the
        Observer pattern to notify visualizers of state changes.

        .. note::
        The Observer pattern is used for state changes, not for new data.
        """

        def __init__(self) -> None:
            """Initialize shared state."""
            super().__init__()

            self.active_annotator: Optional[robokudo.annotators.core.BaseAnnotator] = (
                None
            )
            """ Currently active annotator."""

            self.active_annotator_i: int = 0
            """ Index of currently active annotator """

    instances: List["Visualizer"] = []
    """List of all active visualizer instances"""

    def __init__(
        self,
        pipeline: Pipeline,
        shared_visualizer_state: Optional["Visualizer.SharedState"] = None,
    ) -> None:
        """Initialize the visualizer.

        Do not use this constructor directly. Use :meth:`new_visualizer_instance`
        to properly register visualizers.

        :param pipeline: Pipeline to visualize
        :param shared_visualizer_state: Shared state object for coordinating between visualizers
        """
        self.pipeline: Pipeline = pipeline
        """Pipeline being visualized"""

        self.indicate_termination_var: bool = False
        """Flag indicating if visualization should terminate"""

        self.shared_visualizer_state: Optional["Visualizer.SharedState"] = (
            shared_visualizer_state
        )
        """Shared state object for coordinating between visualizers"""

        self.update_output: bool = True
        """Indicate that the output of this Visualizer needs to be renewed/redrawn"""

        self.new_data: bool = False
        """Flag indicating if new data is available"""

        # for now we assume that every annotator outputs an image
        if (
            self.shared_visualizer_state
            and not self.shared_visualizer_state.active_annotator
        ):
            self.shared_visualizer_state.active_annotator = (
                self.pipeline.get_annotators()[0]
            )
            self.shared_visualizer_state.active_annotator_i = 0

        self.rk_logger: logging.Logger = logging.getLogger(PACKAGE_NAME)
        """Logger instance"""

    def identifier(self) -> str:
        """Get a unique identifier for this visualizer."""
        return "RoboKudo/" + self.pipeline.name

    def pre_tick(self) -> None:
        """Prepare for visualization update.

        Called before :meth:`tick`. Override to implement pre-update logic.
        """
        pass

    def tick(self) -> None:
        """Update the visualization display.

        This is the main method for visualizers. Override to implement
        visualization logic.
        """
        pass

    def post_tick(self) -> None:
        """Clean up after visualization update.

        Called after :meth:`tick`. Override to implement post-update logic.
        """
        pass

    @staticmethod
    def static_post_tick() -> None:
        """Perform static post-update operations.

        This method is called once per visualizer type, regardless of how many
        instances exist. Override to implement static post-update logic.
        """
        pass

    @classmethod
    def new_visualizer_instance(
        cls,
        pipeline: Pipeline,
        shared_visualizer_state: Optional["Visualizer.SharedState"] = None,
    ) -> "Visualizer":
        """Create and register a new visualizer instance.

        :param pipeline: Pipeline to visualize
        :param shared_visualizer_state: Shared state object for coordinating between visualizers

        :returns: New visualizer instance
        """
        vis = cls(pipeline=pipeline, shared_visualizer_state=shared_visualizer_state)
        Visualizer.instances.append(vis)
        return vis

    @staticmethod
    def clear_visualizer_instances() -> None:
        """Remove all registered visualizer instances."""
        Visualizer.instances.clear()

    def insert_input(self) -> None:
        """Handle input insertion.

        Override to implement input handling logic.
        """
        pass

    def activate_update_output(self) -> None:
        """Mark visualizer for update.

        Sets the update flag to trigger a redraw.
        """
        self.update_output = True

    def new_data_available(self) -> None:
        """Signal that new data is available.

        Called by external code to indicate new data is ready for visualization.
        """
        self.new_data = True

    def indicate_termination(self) -> bool:
        """Check if visualization should terminate.

        :returns: True if visualization should terminate, False otherwise
        """
        return self.indicate_termination_var

    @staticmethod
    def get_unique_types_of_visualizer_instances() -> Set[Type]:
        """Get set of unique visualizer types.

        :returns: Set of visualizer classes that have instances
        """
        return set([type(x) for x in Visualizer.instances])

    def update_output_flag_for_new_data(self) -> None:
        """Update flags when new data arrives.

        Sets update flag and clears new data flag when new data is available.
        """
        if self.new_data:
            self.update_output = True
            self.new_data = False

    def get_visualized_annotator_outputs_for_pipeline(
        self,
    ) -> AnnotatorOutputs:
        """Get annotator outputs for visualization.

        :returns: Annotator outputs for the current pipeline
        :raises AssertionError: If outputs are not of the expected type
        """
        blackboard = Blackboard()
        annotator_output_pipeline_map_visualized = blackboard.get(
            "annotator_output_pipeline_map_visualized"
        )
        annotator_outputs = annotator_output_pipeline_map_visualized.map[
            self.pipeline.name
        ]
        assert isinstance(annotator_outputs, AnnotatorOutputs)

        return annotator_outputs
