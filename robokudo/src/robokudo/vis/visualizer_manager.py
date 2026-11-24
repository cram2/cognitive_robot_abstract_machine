"""
Showing the annotator outputs of RoboKudo pipelines in the OpenCV, Open3D and ROS
visualizers.
"""

from __future__ import annotations

import copy
import logging
from timeit import default_timer

from py_trees.common import Status
from py_trees.behaviour import Behaviour
from py_trees.blackboard import Blackboard

from robokudo.pipeline import Pipeline
from robokudo.vis.cv_visualizer import CVVisualizer
from robokudo.vis.o3d_visualizer import O3DVisualizer
from robokudo.vis.ros_visualizer import SharedROSVisualizer, AllAnnotatorROSVisualizer
from robokudo.vis.visualizer import Visualizer

logger = logging.getLogger(__name__)


class VisualizationManager(Behaviour):
    """
    Behaviour tree node that gives every pipeline of its tree one visualizer of each
    type and shows the pipeline's latest annotator outputs in them.
    """

    def __init__(self, name: str, visualizer_types: list[type] | None = None) -> None:
        """
        Initialize the visualization manager.

        :param name: Name of the behavior tree node
        :param visualizer_types: Visualizer backends to instantiate for every pipeline.
            Defaults to all of them (CV, Open3D and both ROS visualizers).
        """
        super().__init__(name=name)

        self.pipelines: dict[str, Pipeline] = {}
        """
        Mapping of pipeline names to Pipeline objects.
        """
        self.visualizer_types: list[type] = visualizer_types or [
            CVVisualizer,
            O3DVisualizer,
            SharedROSVisualizer,
            AllAnnotatorROSVisualizer,
        ]
        """
        List of available visualizer classes.
        """
        self.visualizers: dict[str, list[Visualizer]] = {}
        """
        Mapping of pipeline name to list of Visualizers.
        """

    def create_visualizers_for_pipeline(self, pipeline: Pipeline) -> None:
        """
        Create visualizer instances for a pipeline.

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
    def visualizer_instances() -> list[Visualizer]:
        """
        Get all active visualizer instances.

        :returns: List of all active Visualizer instances
        """
        return Visualizer.instances

    def initialise(self) -> None:
        """
        Collect the pipelines in the subtree of this node's parent.

        .. note::
            Place the manager directly below the top node of the tree, so that it finds
            every pipeline.
        """
        logger.debug("%s.initialise()" % self.__class__.__name__)
        self.pipelines = {}
        # The GUIManager should live one level below the top-node (Parallel)
        # Initially fill a list of all Pipelines
        for node in self.parent.iterate():
            if isinstance(node, Pipeline):
                self.pipelines[node.name] = node

    def update(self) -> Status:
        """
        Show the latest annotator outputs of every pipeline in its visualizers, creating
        the visualizers of a pipeline the first time it has outputs.

        :returns: FAILURE once a visualizer asks to shut down, RUNNING otherwise
        """
        logger.debug("%s.update()" % self.__class__.__name__)
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
                logger.debug(
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
                annotator_output_pipeline_map_visualized.map[
                    pipeline_name
                ].redraw = False

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
