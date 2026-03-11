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

import py_trees

import robokudo.annotators.core


class Redraw(robokudo.annotators.core.BaseAnnotator):
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

    def update(self) -> py_trees.common.Status:
        """Set redraw flag to trigger visualization update.

        Gets the annotator output pipeline map and sets the redraw
        flag for the current pipeline.

        :return: SUCCESS status
        """
        blackboard = py_trees.blackboard.Blackboard()
        annotator_output_pipeline_map_buffer = blackboard.get(
            "annotator_output_pipeline_map_buffer"
        )
        annotator_output_pipeline_map_buffer.map[
            self.get_parent_pipeline().name
        ].redraw = True

        return py_trees.common.Status.SUCCESS
