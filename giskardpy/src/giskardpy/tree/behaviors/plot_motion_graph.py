from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy.tree.blackboard_utils import catch_and_raise_to_blackboard

from giskardpy.tree.behaviors.plugin import GiskardBehavior


class PlotMotionGraph(GiskardBehavior):

    def __init__(self, name: str = "plot task graph"):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        # file_name = (
        #     god_map.tmp_folder
        #     + f"task_graphs/goal_{GiskardBlackboard().move_action_server.goal_id}.pdf"
        # )
        # execution_state = giskard_state_to_execution_state()
        # parser = ExecutionStateToDotParser(execution_state)
        # graph = parser.to_dot_graph()
        # create_path(file_name)
        # graph.write_pdf(file_name)
        # graph.write_dot(file_name + '.dot')
        # rospy.node.get_logger().info(f"Saved task graph at {file_name}.")
        return Status.SUCCESS
