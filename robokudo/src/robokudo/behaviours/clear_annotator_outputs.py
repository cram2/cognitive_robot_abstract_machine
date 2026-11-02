"""
Behaviour that resets the visualization outputs of a pipeline's annotators.
"""

import logging
from dataclasses import dataclass, field

from py_trees.behaviour import Behaviour
from py_trees.blackboard import Blackboard
from py_trees.common import Status

import robokudo.defs as defs
import robokudo.utils.tree
from robokudo.exceptions import ParentPipelineMissing
from robokudo.pipeline import Pipeline


@dataclass(eq=False)
class ClearAnnotatorOutputs(Behaviour):
    """
    Clears the annotator outputs of the nearest pipeline it is part of.
    """

    name: str = "ClearAnnotatorOutputs"
    """
    The name of this behaviour in the behaviour tree.
    """

    rk_logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(defs.PACKAGE_NAME),
        init=False,
        repr=False,
    )
    """
    The logger of the RoboKudo package.
    """

    def __post_init__(self) -> None:
        super().__init__(self.name)

    def update(self) -> Status:
        self.rk_logger.debug("%s.update()" % (self.__class__.__name__))
        blackboard = Blackboard()
        annotator_output_pipeline_map_buffer = blackboard.get(
            "annotator_output_pipeline_map_buffer"
        )
        pipeline = robokudo.utils.tree.find_parent_of_type(self, Pipeline)
        if pipeline is None:
            raise ParentPipelineMissing(behaviour_name=self.name)

        annotator_outputs = annotator_output_pipeline_map_buffer.map[pipeline.name]
        annotator_outputs.clear_outputs()

        return Status.SUCCESS
