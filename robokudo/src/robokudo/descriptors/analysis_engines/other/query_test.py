"""
This script defines a Robokudo-based behavior tree pipeline for processing queries related to numbers.

Features:
- Implements a PyTrees behavior tree to handle query processing.
- Includes a PrintNumbers behavior that sequentially processes numbers from 1 to 10.
- Utilizes a CheckQueryType annotator to validate if the query type is 'numbers'.
- Incorporates a conditional selector to manage task execution with preemption handling.
- Integrates with the Robokudo analysis engine and pipeline framework.
- Uses a feedback mechanism to report progress dynamically.
"""

from queue import Queue

from py_trees.blackboard import Blackboard
from py_trees.common import Status
from py_trees.composites import Selector
from py_trees.composites import Sequence
from py_trees.decorators import Inverter
from robokudo_msgs.action import Query
from typing_extensions import List

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.query import QueryAnnotator
from robokudo.behaviours.action_server_checks import ActionServerNoPreemptRequest
from robokudo.cas import CASViews
from robokudo.identifier import BBIdentifier
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


class PrintNumbers(BaseAnnotator):
    def __init__(self, name: str = "PrintNumbers") -> None:
        super().__init__(name)
        self.current_number: int = 1
        self.completed: bool = False
        self.result: List[int] = []
        self.result_string: str = ""

    def initialise(self) -> None:
        self.result = []
        self.result_string = ""
        self.current_number = 1
        self.completed = False

    def update(self) -> Status:
        blackboard = Blackboard()
        feedback_queue = blackboard.get(BBIdentifier.QUERY_FEEDBACK)
        if feedback_queue is None:
            feedback_queue = Queue()
            blackboard.set(BBIdentifier.QUERY_FEEDBACK, feedback_queue)

        if self.current_number <= 100:
            # Append number and update result
            print(f"Current Number: {self.current_number}")
            self.result.append(self.current_number)
            self.result_string += f"{self.current_number}, "

            # Add feedback message
            feedback_msg = Query.Feedback()
            feedback_msg.feedback = f"Processing number: {self.result_string}"
            feedback_queue.put(feedback_msg)

            self.current_number += 1
            return Status.RUNNING
        else:
            self.completed = True
            blackboard.set(BBIdentifier.QUERY_ANSWER, self.result_string)
            return Status.SUCCESS


class CheckQueryType(BaseAnnotator):
    """
    Checks if the query type on the CAS is 'numbers'.
    """

    def __init__(self, name: str = "CheckQueryType") -> None:
        super().__init__(name)

    def update(self) -> Status:
        # Retrieve the query from the CAS
        query = self.get_cas().get(CASViews.QUERY)

        # Check if the query type is 'numbers'
        if query and getattr(query.obj, "type", None) == "numbers":
            self.logger.info("Query type is 'numbers'. Proceeding.")
            return Status.SUCCESS
        else:
            self.logger.warning(
                f"Query type is not 'numbers' or query is missing: {query}"
            )
            return Status.FAILURE


class AnalysisEngine(AnalysisEngineInterface):
    def name(self) -> str:
        return "test_pipeline"

    def implementation(self) -> Pipeline:
        # Define the sequence that runs the task
        task_sequence = Sequence(name="TaskSequence", memory=True)
        task_sequence.add_children(
            [
                CheckQueryType(),  # Check if the query type is 'numbers'
                PrintNumbers(),  # Execute the task if the query is valid
            ]
        )

        # Combine preemption handling and task execution in a selector
        conditional_selector = Selector(name="ConditionalSelector", memory=False)
        conditional_selector.add_children(
            [
                Inverter(
                    name="Invert Preempt Request", child=ActionServerNoPreemptRequest()
                ),
                task_sequence,  # Run task sequence only if no preemption
            ]
        )

        # Main Pipeline
        seq = Pipeline("OPExperiments")
        seq.add_children(
            [
                pipeline_init(),  # Initialize the pipeline
                QueryAnnotator(),  # Handle incoming queries
                conditional_selector,  # Preemption-aware task execution
            ]
        )

        return seq
