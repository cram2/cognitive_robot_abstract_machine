"""Pipeline implementation for RoboKudo.

This module provides the Pipeline class, which is a specialized version of py_trees'
Sequence behavior. A Pipeline manages its own Common Analysis Structure (CAS) and
provides visualization capabilities for its annotators.

Key features:

* Automatic CAS management for each pipeline iteration
* Visualization support through annotator outputs
* Helper methods for accessing annotators and CAS
* Nested pipeline support with proper scoping
"""

from __future__ import annotations

import logging
import timeit
import typing

from py_trees.blackboard import Blackboard
from py_trees.common import Status
from py_trees.composites import Sequence
from typing_extensions import TYPE_CHECKING, Optional, List, Any

import robokudo.annotators.core
import robokudo.utils.tree
from . import cas
from . import defs
import robokudo.annotators.outputs

# from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap, AnnotatorOutputs

if TYPE_CHECKING:
    from py_trees.behaviour import Behaviour


class Pipeline(Sequence):
    """A RoboKudo pipeline composed of annotators with its own CAS.

    A Pipeline is a specialized sequence behavior that manages a Common Analysis
    Structure (CAS) for its annotators. It provides visualization capabilities
    and helper methods for accessing annotators and CAS data.
    """

    def __init__(
        self,
        name: str = "Sequence",
        children: Optional[List[Behaviour]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, memory=True, children=children)
        """Initialize the pipeline.

        :param name: Name of the pipeline, defaults to "Sequence"
        :param children: List of child behaviors, defaults to None
        """

        self.cas: cas.CAS = None
        """The Common Analysis Structure for this pipeline."""

        self._cas_counter: int = 0
        """Monotonic CAS counter that is local to this pipeline instance."""

        self.cas_start_timer: timeit.default_timer = None
        """Timer for measuring pipeline execution time."""

        self.rk_logger = logging.getLogger(defs.PACKAGE_NAME)
        """Logger instance for this pipeline"""

        self.create_new_cas()
        # self.update_gui_for_this_pipeline = False
        # self.annotator_outputs = AnnotatorOutputs()

    def create_new_cas(self) -> None:
        """Create a new Common Analysis Structure (CAS) for this pipeline. Also resets the execution timer."""
        self._cas_counter += 1
        self.cas = cas.CAS()
        self.cas.cas_id = self._cas_counter
        self.cas_start_timer = timeit.default_timer()

    def setup(self, **kwargs: Any) -> bool:
        """Set up the pipeline and its annotator outputs.

        Creates output structures for all annotators in the pipeline and
        marks them for initial rendering.
        """
        blackboard = Blackboard()
        annotator_output_pipeline_map_buffer = blackboard.get(
            "annotator_output_pipeline_map_buffer"
        )
        assert isinstance(
            annotator_output_pipeline_map_buffer,
            robokudo.annotators.outputs.AnnotatorOutputPerPipelineMap,
        )
        if self.name not in annotator_output_pipeline_map_buffer.map:
            annotator_output_pipeline_map_buffer.map[self.name] = (
                robokudo.annotators.outputs.AnnotatorOutputs()
            )

        annotators = self.get_annotators()
        for a in annotators:
            a.add_self_to_annotator_output_struct()
        annotator_output_pipeline_map_buffer.map[self.name].redraw = True
        return True

    def get_cas(self) -> cas.CAS:
        """Get the pipeline's Common Analysis Structure.

        :return: The pipeline's CAS
        """
        return self.cas

    def get_annotators(self) -> List["robokudo.annotators.core.BaseAnnotator"]:
        """Get all annotators in this pipeline.

        Returns a list of all annotators in this pipeline, including those
        nested in other behaviors (except other pipelines).

        :return: List of annotators
        """
        return [
            child
            for child in self.pipeline_children()
            if isinstance(child, robokudo.annotators.core.BaseAnnotator)
        ]

    def pipeline_children(self) -> List[Behaviour]:
        """Get all children of this pipeline.

        Returns a list of all children in this pipeline, stopping at nested
        Pipeline objects. The order is given by utils.tree.behavior_iterate_except_type.

        :return: List of child behaviors
        """
        return [
            child
            for child in robokudo.utils.tree.behavior_iterate_except_type(
                self, Pipeline
            )
        ]

    def update(self) -> None:
        """Update the pipeline's state.

        Handles pipeline execution status and triggers GUI updates on failure.
        """
        super().update()
        self.rk_logger.debug("%s.update()" % self.__class__.__name__)
        if self.status == Status.FAILURE:
            self.rk_logger.debug(
                "%s: Previous execution was a failure. Update GUI." % self.status
            )
            blackboard = Blackboard()
            annotator_output_pipeline_map_buffer = blackboard.get(
                "annotator_output_pipeline_map_buffer"
            )
            annotator_output_pipeline_map_buffer.map[self.name].redraw = True

    def terminate(self, new_status: Status) -> None:
        """Handle pipeline termination.

        Called when the pipeline switches to a non-RUNNING state. Updates GUI
        on success and logs execution time statistics.

        :param new_status: New execution status
        """
        self.rk_logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )

        if new_status == Status.SUCCESS:
            blackboard = Blackboard()
            annotator_output_pipeline_map_buffer = blackboard.get(
                "annotator_output_pipeline_map_buffer"
            )
            annotator_output_pipeline_map_buffer.map[self.name].redraw = True

        if new_status == Status.SUCCESS or new_status == Status.FAILURE:
            self.feedback_message = f"Processing took {(timeit.default_timer() - self.cas_start_timer):.4f}s"
            self.rk_logger.debug(
                f"Pipeline Processing took in total {(timeit.default_timer() - self.cas_start_timer):.4f}s"
            )
            self.cas_start_timer = timeit.default_timer()
