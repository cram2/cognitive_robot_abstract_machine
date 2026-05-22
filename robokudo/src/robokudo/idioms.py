"""Common behavior patterns for RoboKudo pipelines.

This module provides reusable behavior patterns that are commonly used in
RoboKudo pipelines. These patterns encapsulate standard combinations of
annotators and behaviors that are frequently needed together.

The module provides:

* Standard initialization sequences for pipelines
* Common error handling patterns
* Reusable behavior combinations
"""

from py_trees.composites import Sequence

from robokudo.annotators.outputs import ClearAnnotatorOutputs
from robokudo.behaviours.action_server_checks import ActionServerPresentAndDone
from robokudo.behaviours.clear_errors import ClearErrors


def non_query_pipeline_init() -> Sequence:
    """Same as pipeline_init(), but without waiting for the query-related actions to complete.
    This is the behavior which was used before Dec'25 in RK by default.

    :return: A sequence behavior containing initialization steps
    """
    seq = Sequence(name="Initialization Sequence", memory=True)
    seq.add_children(
        [
            ClearAnnotatorOutputs(),
            ClearErrors(),
        ]
    )

    return seq


def pipeline_init() -> Sequence:
    """
    This method returns a Behaviour/Composite node which hosts different nodes to reset and prepare internal
    datastructures properly before the next iteration can start.
    It will reset the visualization outputs (Annotator Outputs), waits until all pending operations
    on the query server (if present) are done (abort goal operations, send goal results, etc.).
    Finally, it will clear any previous exceptions present in the error variable.

    :return: A sequence behavior containing initialization steps
    """
    seq = Sequence(name="Initialization Sequence", memory=True)
    seq.add_children(
        [
            ClearAnnotatorOutputs(),
            ActionServerPresentAndDone(),
            ClearErrors(),
        ]
    )

    return seq
