from pycram.robot_plans import ActionDescription
from typing_extensions import Type, List, Optional
from queue import Queue

from pycram.plan import Plan, ActionNode
from ..episode_segmenter import AgentEpisodeSegmenter
from ..players.cram_player import CRAMPlayer


class CRAMSegmenter(AgentEpisodeSegmenter):
    """
    The CRAMSegmenter class is used to segment the CRAMs motion replay data by using event detectors, such as contact,
    loss of contact, and pick up events.
    """

    def __init__(self, world, detectors_to_start=None, annotate_events=False, **kwargs):
        """
        Initializes the CRAMSegmenter class.

        :param world: The world object used to query the CRAMs motion replay data.
        :param detectors_to_start: An optional list of event detectors to start.
        :param annotate_events: A boolean value that indicates whether the events should be annotated.
        """
        self.cram_player_thread = CRAMPlayer(world)
        super().__init__(self.cram_player_thread,
                         detectors_to_start=detectors_to_start,
                         annotate_events=annotate_events, **kwargs)
        self.action_types: List[Optional[Type[ActionDescription]]] = [detector.action_type()
                                                                      for detector in self.detectors_to_start]
        self.action_types.append(None)
        self.start_action_queue: Queue = Queue()
        self.end_action_queue: Queue = Queue()
        for action_type in self.action_types:
            self.add_callback(action_type)

    def add_callback(self, action_type: Optional[Type[ActionDescription]] = None):
        """
        Add a callback for the given action type.

        :param action_type: The action type to add the callback for.
        """
        Plan.add_on_start_callback(self.start_action_callback, action_type=action_type)
        Plan.add_on_end_callback(self.end_action_callback, action_type=action_type)

    def start_action_callback(self, action_node: ActionNode[ActionDescription]):
        """
        The action callback method that is called when an action is performed.

        :param action_node: The node in the task tree representing the action that was performed.
        """
        # Maybe create an Event for the given action.
        # One could use that for supervising the RDRs of action detection.
        # Maybe fit_rdr_case here :D.
        # print(f"Action Started: {action_node}")
        self.start_action_queue.put(action_node)

    def end_action_callback(self, action_node: ActionNode[ActionDescription]):
        """
        The action callback method that is called when an action is performed.

        :param action_node: The node in the task tree representing the action that was performed.
        """
        # print(f"Action Ended: {action_node}")
        self.end_action_queue.put(action_node)

    def start(self) -> None:
        """
        Query CRAMs for motion replay data and run the event detectors on the CRAMs motion replay data.
        """
        super().start()
