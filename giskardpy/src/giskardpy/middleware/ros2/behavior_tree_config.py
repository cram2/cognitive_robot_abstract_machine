from dataclasses import dataclass, field

from py_trees.decorators import FailureIsSuccess
from giskardpy.middleware.ros2.utils.utils import is_in_github_workflow
from giskardpy.tree.behaviors.publish_debug_expressions import QPDataPublisherConfig
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from giskardpy.tree.branches.giskard_bt import GiskardBT
from giskardpy.tree.branches.send_trajectories import ExecuteTraj


@dataclass
class BehaviorTreeConfig:
    tree: GiskardBT = field(init=False)
    tree_tick_rate: float = 0.05
    debug_mode: bool = False

    add_gantt_chart_plotter: bool = False
    add_goal_graph_plotter: bool = False
    add_trajectory_plotter: bool = False
    add_debug_trajectory_plotter: bool = False
    add_debug_marker_publisher: bool = False
    add_trajectory_visualizer: bool = False
    add_debug_trajectory_visualizer: bool = False
    add_qp_data_publisher: QPDataPublisherConfig = field(
        default_factory=QPDataPublisherConfig
    )

    def __post_init__(self):
        if is_in_github_workflow():
            self.debug_mode = False

    def is_closed_loop(self) -> bool:
        return isinstance(self, ClosedLoopBTConfig)

    def is_standalone(self) -> bool:
        return isinstance(self, StandAloneBTConfig)

    def is_open_loop(self) -> bool:
        return isinstance(self, OpenLoopBTConfig)

    def setup(self):
        """
        Implement this method to configure the behavior tree using it's self. methods.
        """
        GiskardBlackboard().tree_config = self
        self.tree = GiskardBT()
        self.add_tf_publisher()
        if self.debug_mode:
            if self.add_trajectory_plotter:
                self._add_trajectory_plotter(wait=True)
            if self.add_gantt_chart_plotter:
                self._add_gantt_chart_plotter()
            if self.add_goal_graph_plotter:
                self._add_goal_graph_plotter()
            if self.add_qp_data_publisher.any():
                self._add_qp_data_publisher(publish_config=self.add_qp_data_publisher)

    def switch_to_projection_mode(self):
        """Override this method to define projection mode behavior for each config type."""
        raise NotImplementedError()

    def switch_to_execution_mode(self):
        """Override this method to define execution mode behavior for each config type."""
        raise NotImplementedError()

    def _add_qp_data_publisher(self, publish_config: QPDataPublisherConfig):
        """
        QP data is streamed and can be visualized in e.g. plotjuggler. Useful for debugging.
        """
        self.add_evaluate_debug_expressions()
        if GiskardBlackboard().tree_config.is_open_loop():
            self.tree.execute_traj.base_closed_loop.publish_state.add_qp_data_publisher(
                publish_config=publish_config
            )
        else:
            self.tree.control_loop_branch.publish_state.add_qp_data_publisher(
                publish_config=publish_config
            )

    def _add_trajectory_plotter(
        self, normalize_position: bool = False, wait: bool = False
    ):
        """
        Plots the generated trajectories.
        :param normalize_position: Positions are centered around zero.
        :param wait: True: Behavior tree waits for this plotter to finish.
                     False: Plot is generated in a separate thread to not slow down Giskard.
        """
        self.tree.cleanup_control_loop.add_plot_trajectory(normalize_position, wait)

    def _add_gantt_chart_plotter(self):
        self.add_evaluate_debug_expressions()
        self.tree.cleanup_control_loop.add_plot_gantt_chart()

    def _add_goal_graph_plotter(self):
        self.add_evaluate_debug_expressions()
        self.tree.prepare_control_loop.add_plot_goal_graph()

    def add_tf_publisher(self):
        """
        Publishes tf for Giskard's internal state.
        """
        self.tree.wait_for_goal.publish_state.add_tf_publisher()
        if GiskardBlackboard().tree_config.is_standalone():
            self.tree.control_loop_branch.publish_state.add_tf_publisher()

    def add_evaluate_debug_expressions(self):
        self.tree.prepare_control_loop.add_compile_debug_expressions()
        if GiskardBlackboard().tree_config.is_closed_loop():
            self.tree.control_loop_branch.add_evaluate_debug_expressions(log_traj=False)
        else:
            self.tree.control_loop_branch.add_evaluate_debug_expressions(log_traj=True)
        if GiskardBlackboard().tree_config.is_open_loop() and hasattr(
            GiskardBlackboard().tree.execute_traj, "prepare_base_control"
        ):
            GiskardBlackboard().tree.execute_traj.prepare_base_control.add_compile_debug_expressions()
            GiskardBlackboard().tree.execute_traj.base_closed_loop.add_evaluate_debug_expressions(
                log_traj=False
            )

    def add_world_state_publisher(self):
        """
        Publishes joint states for Giskard's internal state.
        """
        GiskardBlackboard().tree.wait_for_goal.publish_state.add_joint_state_publisher()
        # GiskardBlackboard().tree.control_loop_branch.publish_state.add_joint_state_publisher()


@dataclass
class StandAloneBTConfig(BehaviorTreeConfig):
    """
    The default behavior tree for Giskard in standalone mode. Make sure to set up the robot interface accordingly.
    :param publish_world_state: publish current world state.
    :param publish_tf: publish all link poses in tf.
    """

    publish_world_state: bool = True

    def __post_init__(self):
        super().__post_init__()
        if is_in_github_workflow():
            self.publish_tf = True

    def setup(self):
        super().setup()
        self.tree.control_loop_branch.add_projection_behaviors()
        self.add_evaluate_debug_expressions()
        if self.publish_world_state:
            self.add_world_state_publisher()

    def switch_to_projection_mode(self):
        # StandAlone specific projection logic
        pass

    def switch_to_execution_mode(self):
        # StandAlone specific execution logic
        pass


@dataclass
class OpenLoopBTConfig(BehaviorTreeConfig):
    """
    The default behavior tree for Giskard in open-loop mode. It will first plan the trajectory in simulation mode
    and then publish it to connected joint trajectory followers. The base trajectory is tracked with a closed-loop
    controller.
    """

    def setup(self):
        super().setup()
        self.tree.control_loop_branch.add_projection_behaviors()
        self.tree.execute_traj = ExecuteTraj()
        self.tree.execute_traj_failure_is_success = FailureIsSuccess(
            "ignore failure", self.tree.execute_traj
        )

    def switch_to_projection_mode(self):
        self.tree.root.remove_child(self.tree.execute_traj_failure_is_success)

    def switch_to_execution_mode(self):
        self.tree.root.insert_child(self.tree.execute_traj_failure_is_success, -2)


@dataclass
class ClosedLoopBTConfig(BehaviorTreeConfig):
    """
    The default configuration for Giskard in closed loop mode. Make use to set up the robot interface accordingly.
    """

    def setup(self):
        super().setup()
        self.tree.control_loop_branch.add_closed_loop_behaviors()
        self.add_world_state_publisher()

    def switch_to_projection_mode(self):
        self.tree.control_loop_branch.switch_to_projection()

    def switch_to_execution_mode(self):
        self.tree.control_loop_branch.switch_to_closed_loop()
