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

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from py_trees.blackboard import Blackboard
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator


@dataclass(eq=False)
class Redraw(BaseAnnotator):
    """
    Annotator for triggering visualization updates.

    This annotator sets the redraw flag on the pipeline to make visualizers show the
    latest visualization output.
    """

    name: str = __qualname__
    """
    Name of this node in the behaviour tree.
    """

    def update(self) -> Status:
        """
        Set redraw flag to trigger visualization update.

        Gets the annotator output pipeline map and sets the redraw flag for the current
        pipeline.

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
