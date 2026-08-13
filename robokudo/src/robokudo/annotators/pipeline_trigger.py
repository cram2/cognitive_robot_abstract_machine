"""Pipeline execution control through triggers.

This module provides an annotator for:

* Controlling pipeline execution with triggers
* Waiting for external signals via blackboard
* Resetting trigger state after activation

.. note::
   The pipeline will remain in RUNNING state until triggered.
"""

from py_trees.blackboard import Blackboard
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator


class PipelineTrigger(BaseAnnotator):
    """Pipeline execution controller using triggers.

    This annotator basically gives you a way to wait for a 'trigger'
    before a pipeline should start.
    It basically just returns RUNNING until it received the trigger from the blackboard.

    .. note::
       The trigger is accessed via "pipeline_trigger" on the blackboard.
    """

    def __init__(self, name: str = "PipelineTrigger") -> None:
        """Initialize the pipeline trigger.

        :param name: Name of this annotator instance, defaults to "PipelineTrigger"
        """
        super().__init__(name)

    def update(self) -> Status:
        """Check trigger state and control pipeline execution.

        The method:

        * Reads trigger state from blackboard
        * Returns SUCCESS if triggered
        * Resets trigger to False
        * Returns RUNNING if not triggered

        :return: SUCCESS if triggered, RUNNING otherwise
        """
        self.rk_logger.debug("%s.update()" % (self.__class__.__name__))
        blackboard = Blackboard()
        pipeline_trigger = blackboard.get("pipeline_trigger")

        if pipeline_trigger:
            blackboard.set("pipeline_trigger", False)
            return Status.SUCCESS

        return Status.RUNNING
