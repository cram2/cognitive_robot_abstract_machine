from py_trees.decorators import FailureIsRunning, SuccessIsRunning

from giskardpy.executor import SimulationPacer
from giskardpy.tree.behaviors.goal_canceled import GoalCanceled
from giskardpy.tree.behaviors.instantaneous_controller import ControllerPlugin
from giskardpy.tree.behaviors.time import RosTime
from giskardpy.tree.branches.publish_state import PublishState
from giskardpy.tree.composites.async_composite import AsyncBehavior
from giskardpy.utils.decorators import toggle_on, toggle_off
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from giskardpy.tree.branches.send_controls import SendControls
from giskardpy.tree.branches.synchronization import Synchronization


class ControlLoop(AsyncBehavior):
    publish_state: PublishState
    projection_synchronization: Synchronization
    closed_loop_synchronization: Synchronization
    debug_added: bool = False
    in_projection: bool
    # controller_active: bool = True

    ros_time: RosTime
    send_controls: SendControls
    controller_plugin: ControllerPlugin

    def __init__(self, name: str = "control_loop", log_traj: bool = True):
        pacer = GiskardBlackboard().giskard.executor.pacer
        if isinstance(pacer, SimulationPacer) and pacer.real_time_factor is None:
            control_dt = None
        else:
            control_dt = 1 / pacer.target_frequency
        if control_dt is not None:
            max_hz = 1 / control_dt
        else:
            max_hz = None
        name = f"{name}\nmax_hz -- {max_hz}"
        super().__init__(name, max_hz=max_hz)
        self.publish_state = PublishState("publish state 2")
        self.publish_state.add_publish_feedback()
        self.projection_synchronization = Synchronization()
        self.projection_synchronization_sir = SuccessIsRunning(
            f"sir {self.projection_synchronization.name}",
            self.projection_synchronization,
        )
        # projection plugins

        self.ros_time = RosTime()
        self.send_controls = SendControls()
        self.closed_loop_synchronization = Synchronization()
        self.closed_loop_synchronization_sir = SuccessIsRunning(
            f"sir {self.closed_loop_synchronization.name}",
            self.closed_loop_synchronization,
        )

        goal_canceled = GoalCanceled(GiskardBlackboard().move_action_server)

        self.add_child(
            FailureIsRunning(
                f"failure is running\n{goal_canceled.name}", goal_canceled
            ),
            success_is_running=False,
        )

        self.controller_plugin = ControllerPlugin("controller")
        self.add_child(self.controller_plugin, success_is_running=False)

        self.add_child(self.publish_state)

    @toggle_on("in_projection")
    def switch_to_projection(self):
        self.remove_closed_loop_behaviors()
        self.add_projection_behaviors()

    @toggle_off("in_projection")
    def switch_to_closed_loop(self):
        assert GiskardBlackboard().tree_config.is_closed_loop()
        self.remove_projection_behaviors()
        self.add_closed_loop_behaviors()

    # @toggle_on("controller_active")
    # def add_qp_controller(self):
    #     self.insert_behind(self.controller_plugin, self.evaluate_monitors)
    #
    # @toggle_off("controller_active")
    # def remove_qp_controller(self):
    #     self.remove_child(self.controller_plugin)
    #     self.remove_child(self.kin_sim)

    def remove_projection_behaviors(self):
        self.remove_child(self.projection_synchronization_sir)
        # self.publish_state.remove_visualization_marker_behavior()

    def remove_closed_loop_behaviors(self):
        self.remove_child(self.closed_loop_synchronization_sir)
        self.remove_child(self.ros_time)
        self.remove_child(self.send_controls)

    def add_projection_behaviors(self):
        # self.publish_state.add_visualization_marker_behavior()
        self.insert_child(self.projection_synchronization_sir, 1)
        self.in_projection = True

    def add_closed_loop_behaviors(self):
        self.insert_child(self.closed_loop_synchronization_sir, 1)
        self.insert_child(
            SuccessIsRunning(f"sir {self.ros_time.name}", self.ros_time), -2
        )
        self.insert_child(self.send_controls, -2)
        self.in_projection = False

    def add_evaluate_debug_expressions(self, log_traj: bool):
        if not self.debug_added:
            self.debug_added = True
