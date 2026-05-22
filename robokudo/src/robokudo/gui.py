"""
GUI-related behavior tree components for RoboKudo.

This module provides behaviors that control GUI rendering and visualization
in RoboKudo pipelines. The behaviors can be used to trigger redrawing of
annotator outputs and manage pipeline visualization state.
"""

import logging

from py_trees.common import Status
from py_trees.behaviour import Behaviour
from py_trees.blackboard import Blackboard

from robokudo.pipeline import Pipeline
from robokudo.defs import PACKAGE_NAME


class SetPipelineRedraw(Behaviour):
    """A behavior that triggers GUI redraw of pipeline outputs.

    This behavior indicates that the GUI output of the owning Pipeline
    should render its AnnotatorOutputs in the next tick(). Useful when
    you want to render current AnnotatorOutputs while the Pipeline is
    still running.

    .. note::
        This behavior must be added directly as a child to a
        :class:`robokudo.pipeline.Pipeline` node.
    """

    def __init__(self, name: str = "SetPipelineRedraw") -> None:
        """Initialize the behavior.

        :param name: Name of the behavior
        """
        super().__init__(name)

        self.rk_logger = logging.getLogger(PACKAGE_NAME)

    def update(self) -> Status:
        """Execute the behavior's update step.

        Checks if the parent is a Pipeline and sets its redraw flag.
        Returns SUCCESS if redraw was set, FAILURE if parent is not
        a Pipeline.

        :return: Behavior execution status
        """
        self.rk_logger.debug("%s.update()" % self.__class__.__name__)

        parent_pipeline = self.parent
        if not isinstance(parent_pipeline, Pipeline):
            self.rk_logger.warning(
                "You've put a PipelineGUI behavior in your tree but your parent is not a Pipeline. Exiting..."
            )
            return Status.INVALID

        blackboard = Blackboard()
        annotator_output_pipeline_map_buffer = blackboard.get(
            "annotator_output_pipeline_map_buffer"
        )
        annotator_output_pipeline_map_buffer.map[parent_pipeline.name].redraw = True

        return Status.SUCCESS
