import traceback

from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import GiskardBlackboard


class PlotGanttChart(GiskardBehavior):

    def __init__(self, name: str = "plot gantt chart"):
        super().__init__(name)

    @record_time
    def update(self):
        if not GiskardBlackboard().motion_statechart.history:
            return Status.SUCCESS
        try:
            file_name = (
                GiskardBlackboard().executor.tmp_folder
                + f"gantt_charts/goal_{GiskardBlackboard().move_action_server.goal_id}.pdf"
            )
            GiskardBlackboard().motion_statechart.plot_gantt_chart(
                file_name,
                context=GiskardBlackboard().executor.context,
                second_length_in_cm=1.5,
            )
        except Exception as e:
            rospy.node.get_logger().warning(f"Failed to create goal gantt chart: {e}.")
            traceback.print_exc()

        return Status.SUCCESS
