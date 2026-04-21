from dataclasses import dataclass

import numpy as np
from py_trees.common import Status
from sensor_msgs.msg import JointState

from giskardpy.qp.qp_controller import QPController
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy.middleware.ros2 import rospy


@dataclass
class QPDataPublisherConfig:
    publish_lb: bool = False
    publish_ub: bool = False
    publish_lbA: bool = False
    publish_ubA: bool = False
    publish_bE: bool = False
    publish_Ax: bool = False
    publish_Ex: bool = False
    publish_xdot: bool = False
    publish_weights: bool = False
    publish_g: bool = False
    publish_debug: bool = False

    def any(self) -> bool:
        return any(
            [
                self.publish_lb,
                self.publish_ub,
                self.publish_lbA,
                self.publish_ubA,
                self.publish_bE,
                self.publish_Ax,
                self.publish_Ex,
                self.publish_xdot,
                self.publish_weights,
                self.publish_g,
                self.publish_debug,
            ]
        )


class PublishDebugExpressions(GiskardBehavior):

    def __init__(
        self, publish_config: QPDataPublisherConfig, name: str = "publish qp data"
    ):
        super().__init__(name)
        self.config = publish_config

    def setup(self, timeout):
        self.publisher = rospy.node.create_publisher(
            JointState, f"{rospy.node.get_name()}/qp_data", 10
        )
        return super().setup(timeout)

    def create_msg(self, qp_controller: QPController):
        msg = JointState()
        msg.header.stamp = rospy.node.get_clock().now().to_msg()

        weights, g, lb, ub, E, bE, A, lbA, ubA, weight_filter, bE_filter, bA_filter = (
            qp_controller.qp_solver.get_problem_data()
        )
        free_variable_names = qp_controller.free_variable_bounds.names[weight_filter]
        equality_constr_names = qp_controller.equality_bounds.names[bE_filter]
        inequality_constr_names = qp_controller.inequality_bounds.names[bA_filter]

        if self.config.publish_debug:
            for (
                name,
                value,
            ) in debug_expression_manager.evaluated_debug_expressions.items():
                if isinstance(value, np.ndarray):
                    if len(value) > 1:
                        if len(value.shape) == 2:
                            for x in range(value.shape[0]):
                                for y in range(value.shape[1]):
                                    tmp_name = f"{name}|{x}_{y}"
                                    msg.name.append(tmp_name)

                                    msg.position.append(value[x, y])
                        else:
                            for x in range(value.shape[0]):
                                tmp_name = f"{name}|{x}"
                                msg.name.append(tmp_name)
                                msg.position.append(value[x])
                    else:
                        msg.name.append(name)
                        msg.position.append(value.flatten())
                else:
                    msg.name.append(name)
                    msg.position.append(value)

        if self.config.publish_lb:
            names = [f"lb/{entry_name}" for entry_name in free_variable_names]
            msg.name.extend(names)
            msg.position.extend(lb.tolist())

        if self.config.publish_ub:
            names = [f"ub/{entry_name}" for entry_name in free_variable_names]
            msg.name.extend(names)
            msg.position.extend(ub.tolist())

        if self.config.publish_lbA:
            names = [f"lbA/{entry_name}" for entry_name in inequality_constr_names]
            msg.name.extend(names)
            msg.position.extend(lbA.tolist())

        if self.config.publish_ubA:
            names = [f"ubA/{entry_name}" for entry_name in inequality_constr_names]
            msg.name.extend(names)
            msg.position.extend(ubA.tolist())

        if self.config.publish_bE:
            names = [f"bE/{entry_name}" for entry_name in equality_constr_names]
            msg.name.extend(names)
            msg.position.extend(bE.tolist())

        if self.config.publish_weights:
            names = [f"weights/{entry_name}" for entry_name in free_variable_names]
            msg.name.extend(names)
            msg.position.extend(weights.tolist())

        if self.config.publish_g:
            names = [f"g/{entry_name}" for entry_name in free_variable_names]
            msg.name.extend(names)
            msg.position.extend(g.tolist())

        if self.config.publish_xdot:
            names = [f"xdot/{entry_name}" for entry_name in free_variable_names]
            msg.name.extend(names)
            msg.position.extend(qp_controller.xdot_full.tolist())

        if self.config.publish_Ax or self.config.publish_Ex:
            num_vel_constr = len(qp_controller.derivative_constraints) * (
                qp_controller.prediction_horizon - 2
            )
            num_neq_constr = len(qp_controller.inequality_constraints)
            num_eq_constr = len(qp_controller.equality_constraints)
            num_constr = num_vel_constr + num_neq_constr + num_eq_constr

            pure_xdot = qp_controller.xdot_full.copy()
            pure_xdot[-num_constr:] = 0

            if self.config.publish_Ax:
                names = [f"Ax/{entry_name}" for entry_name in inequality_constr_names]
                msg.name.extend(names)
                Ax = np.dot(A, pure_xdot)
                # Ax[-num_constr:] /= sample_period
                msg.position.extend(Ax.tolist())
            if self.config.publish_Ex:
                names = [f"Ex/{entry_name}" for entry_name in equality_constr_names]
                msg.name.extend(names)
                Ex = np.dot(E, pure_xdot)
                # Ex[-num_constr:] /= sample_period
                msg.position.extend(Ex.tolist())

        return msg

    @record_time
    def update(self):
        raise NotImplementedError("needs fixing")
        qp_controller: QPController = GiskardBlackboard().executor.qp_controller
        msg = self.create_msg(qp_controller)
        self.publisher.publish(msg)
        return Status.SUCCESS
