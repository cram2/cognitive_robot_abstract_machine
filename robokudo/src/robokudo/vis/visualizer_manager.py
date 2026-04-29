"""Visualization management for RoboKudo pipelines.

This module provides a behavior tree node that manages visualization of pipeline data
using different visualization backends. It coordinates:

* OpenCV-based visualization
* Open3D visualization
* ROS visualization
* Shared visualization state
* Pipeline data synchronization
"""

import copy
import logging
from timeit import default_timer

from py_trees.common import Status
from py_trees.behaviour import Behaviour
from py_trees.blackboard import Blackboard
from typing_extensions import Dict, List, Type

from robokudo.defs import PACKAGE_NAME
from robokudo.pipeline import Pipeline
from robokudo.vis.cv_visualizer import CVVisualizer
from robokudo.vis.o3d_visualizer import O3DVisualizer
from robokudo.vis.ros_visualizer import SharedROSVisualizer, AllAnnotatorROSVisualizer
from robokudo.vis.visualizer import Visualizer


class VisualizationManager(Behaviour):
    """Behavior tree node for managing pipeline visualizations.

    This class coordinates visualization of pipeline data across multiple
    visualization backends. It handles:

    * Pipeline data buffering and synchronization
    * Visualizer creation and lifecycle management
    * Shared visualization state
    * Visualization timing and performance monitoring

    .. note::
        The manager operates in three phases:

        1. Pre-tick: Prepare visualizers for new data
        2. Tick: Update visualizations
        3. Post-tick: Cleanup and synchronization
    """

    def __init__(self, name: str) -> None:
        """Initialize the visualization manager.

        :param name: Name of the behavior tree node
        """
        super().__init__(name=name)

        self.pipelines: Dict[str, Pipeline] = {}
        """Mapping of pipeline names to Pipeline objects"""

        self.rk_logger: logging.Logger = logging.getLogger(PACKAGE_NAME)
        """Logger instance for this class"""

        self.visualizer_types: List[Type] = [
            CVVisualizer,
            O3DVisualizer,
            SharedROSVisualizer,
            AllAnnotatorROSVisualizer,
        ]
        """List of available visualizer classes"""

        self.visualizers: Dict[str, List[Visualizer]] = {}
        """Mapping of pipeline name to list of Visualizers"""

    def create_visualizers_for_pipeline(self, pipeline: Pipeline) -> None:
        """Create visualizer instances for a pipeline.

        :param pipeline: Pipeline to create visualizers for

        .. note::
            Creates one instance of each visualizer type and associates them
            with the pipeline. All visualizers share a common visualization state.
        """
        # TODO Handle shared visualization context - Not all Visualizers need one!
        shared_state = Visualizer.SharedState()
        visualizers = [
            visclass.new_visualizer_instance(
                pipeline=pipeline, shared_visualizer_state=shared_state
            )
            for visclass in self.visualizer_types
        ]
        self.visualizers[pipeline.name] = visualizers

    @staticmethod
    def visualizer_instances() -> List[Visualizer]:
        """Get all active visualizer instances.

        :returns: List of all active Visualizer instances
        """
        return Visualizer.instances

    def initialise(self) -> None:
        """Initialize the behavior tree node.

        This method:

        * Clears existing pipeline references
        * Finds all Pipeline nodes in the behavior tree
        * Stores references to found pipelines

        .. note::
            The VisualizationManager should be placed one level below the top node
            (Parallel) in the behavior tree.
        """
        self.rk_logger.debug("%s.initialise()" % self.__class__.__name__)
        self.pipelines = {}
        # The GUIManager should live one level below the top-node (Parallel)
        # Initially fill a list of all Pipelines
        for node in self.parent.iterate():
            if isinstance(node, Pipeline):
                self.pipelines[node.name] = node

    def update(self) -> Status:
        """Update visualizations for all pipelines.

        This method:

        1. Copies new data from buffer to visualization map
        2. Creates visualizers for new pipelines
        3. Notifies visualizers of new data
        4. Executes pre-tick, tick, and post-tick phases
        5. Checks for termination signals

        :returns: FAILURE if any visualizer indicates termination, RUNNING otherwise

        .. note::
            The update cycle ensures synchronized visualization across all backends
            while maintaining separation between data buffers and visualization state.
        """
        self.rk_logger.debug("%s.update()" % self.__class__.__name__)
        start_timer = default_timer()

        blackboard = Blackboard()
        annotator_output_pipeline_map_buffer = blackboard.get(
            "annotator_output_pipeline_map_buffer"
        )  # Working buffer
        annotator_output_pipeline_map_visualized = blackboard.get(
            "annotator_output_pipeline_map_visualized"
        )  # Vis only

        # Create Visualized Buffer if new data is available
        for pipeline_name in annotator_output_pipeline_map_buffer.map:
            if annotator_output_pipeline_map_buffer.map[pipeline_name].redraw:
                self.rk_logger.debug(
                    "%s.update(): Redraw flag on Pipeline %s is true. Copying data"
                    % (self.__class__.__name__, pipeline_name)
                )

                annotator_output_pipeline_map_visualized.map[pipeline_name] = (
                    copy.deepcopy(
                        annotator_output_pipeline_map_buffer.map[pipeline_name]
                    )
                )

                annotator_output_pipeline_map_buffer.map[pipeline_name].redraw = False

        # Create new Visualizers based on the visualization data
        for pipeline_name in annotator_output_pipeline_map_visualized.map:
            assert pipeline_name in self.pipelines

            if pipeline_name not in self.visualizers:
                self.create_visualizers_for_pipeline(self.pipelines[pipeline_name])

            # Ping Visualizers if there is new data for this Pipeline
            if annotator_output_pipeline_map_visualized.map[pipeline_name].redraw:
                for visualizer in self.visualizers[pipeline_name]:
                    visualizer.new_data_available()

                # Visualizers have been informed
                annotator_output_pipeline_map_visualized.map[pipeline_name].redraw = (
                    False
                )

        # Pre-tick
        for pipeline_name, visualizer_list in self.visualizers.items():
            for visualizer in visualizer_list:
                visualizer.pre_tick()

        # Tick
        for pipeline_name, visualizer_list in self.visualizers.items():
            for visualizer in visualizer_list:
                visualizer.tick()
        # Post Tick
        for pipeline_name, visualizer_list in self.visualizers.items():
            for visualizer in visualizer_list:
                visualizer.post_tick()

        # Static Post Tick
        for visualizer_type in self.visualizer_types:
            visualizer_type.static_post_tick()

        # Check if one Visualizer returned a termination signal
        for vis in VisualizationManager.visualizer_instances():
            if vis.indicate_termination():
                return Status.FAILURE

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"

        return Status.RUNNING
