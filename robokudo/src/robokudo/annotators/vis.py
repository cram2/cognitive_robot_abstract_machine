"""
Visualization control for RoboKudo annotators.

This module provides annotators for controlling visualization updates.
It supports:

* Manual visualization redraw triggering
* Pipeline-specific visualization control
* Output buffer management
* Visualization state management
* Display synchronization

The module is used for:

* Visualization control
* Display updates
* GUI synchronization
* Output buffer management
"""

from py_trees.blackboard import Blackboard
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator


class Redraw(BaseAnnotator):
    """
    Annotator for triggering visualization updates.

    This annotator sets the redraw flag on the pipeline to make
    visualizers show the latest visualization output.
    """

    def __init__(self, name: str = "Redraw") -> None:
        """Initialize the redraw annotator.

        :param name: Annotator name, defaults to "Redraw"
        """
        super().__init__(name=name)

    def update(self) -> Status:
        """Set redraw flag to trigger visualization update.

        Gets the annotator output pipeline map and sets the redraw
        flag for the current pipeline.

        :return: SUCCESS status
        """
        blackboard = Blackboard()
        annotator_output_pipeline_map_buffer = blackboard.get(
            "annotator_output_pipeline_map_buffer"
        )
        annotator_output_pipeline_map_buffer.map[
            self.get_parent_pipeline().name
        ].redraw = True

        return Status.SUCCESS
