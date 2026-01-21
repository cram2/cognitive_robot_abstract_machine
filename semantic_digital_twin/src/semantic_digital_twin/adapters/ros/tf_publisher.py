from collections import OrderedDict
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.publisher import Publisher
from tf2_msgs.msg import TFMessage

from krrood.symbolic_math.symbolic_math import (
    Matrix,
    VariableParameters,
    CompiledFunction,
)
from semantic_digital_twin.callbacks.callback import (
    StateChangeCallback,
    ModelChangeCallback,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass
class TfPublisherModelCallback(ModelChangeCallback):
    node: Node
    ignored_kinematic_structure_entities: set[KinematicStructureEntity] = field(
        default_factory=set
    )
    connections_to_expression: dict[tuple[UUID, UUID], Matrix] = field(
        init=False, default_factory=OrderedDict
    )
    size: int = field(init=False, default=0)
    tf_message: TFMessage = field(init=False)
    compiled_tf: CompiledFunction = field(init=False)

    def build_tf(self):
        for connection in self.world.connections:
            if (
                connection.parent in self.ignored_kinematic_structure_entities
                or connection.child in self.ignored_kinematic_structure_entities
            ):
                continue
            self.connections_to_expression[
                (connection.parent.id, connection.child.id)
            ] = connection.origin_as_position_quaternion()
        self.size = len(self.connections_to_expression)

    def _notify(self):
        self.build_tf()
        self.compile_tf_expression()
        self.init_tf_message()

    def compile_tf_expression(self):
        tf = Matrix.vstack([pose for pose in self.connections_to_expression.values()])
        params = [v.variables.position for v in self.world.degrees_of_freedom]
        self.compiled_tf = tf.compile(parameters=VariableParameters.from_lists(params))
        self.compiled_tf.bind_args_to_memory_view(0, self.world.state.positions)

    def compute_tf(self) -> np.ndarray:
        return self.compiled_tf.evaluate()

    def init_tf_message(self):
        self.tf_message = TFMessage()
        self.tf_message.transforms = [TransformStamped() for _ in range(self.size)]
        for i, (parent_link_id, child_link_id) in enumerate(
            self.connections_to_expression
        ):
            parent_link = self.world.get_kinematic_structure_entity_by_id(
                parent_link_id
            )
            child_link = self.world.get_kinematic_structure_entity_by_id(child_link_id)

            self.tf_message.transforms[i].header.frame_id = str(parent_link.name)
            self.tf_message.transforms[i].child_frame_id = str(child_link.name)

    def update_tf_message(self):
        tf_data = self.compute_tf()
        current_time = self.node.get_clock().now().to_msg()
        for i, (p_T_c, pose) in enumerate(zip(self.tf_message.transforms, tf_data)):
            p_T_c.header.stamp = current_time
            p_T_c.transform.translation.x = pose[0]
            p_T_c.transform.translation.y = pose[1]
            p_T_c.transform.translation.z = pose[2]
            p_T_c.transform.rotation.x = pose[3]
            p_T_c.transform.rotation.y = pose[4]
            p_T_c.transform.rotation.z = pose[5]
            p_T_c.transform.rotation.w = pose[6]


@dataclass
class TFPublisher(StateChangeCallback):
    """
    On state change, publishes the TF tree of the world.
    Puts a frame in every kinematic structure entity that is not in the ignored_bodies set.
    """

    node: Node
    world: World
    ignored_kinematic_structure_entities: set[KinematicStructureEntity] = field(
        default_factory=set
    )
    tf_topic: str = field(default="tf")
    tf_pub: Publisher = field(init=False)

    tf_model_cb: TfPublisherModelCallback = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.tf_pub = self.node.create_publisher(TFMessage, self.tf_topic, 10)
        self.tf_model_cb = TfPublisherModelCallback(
            node=self.node,
            world=self.world,
            ignored_kinematic_structure_entities=self.ignored_kinematic_structure_entities,
        )
        self.tf_model_cb.notify()
        self._notify()

    def _notify(self):
        self.tf_model_cb.update_tf_message()
        self.tf_pub.publish(self.tf_model_cb.tf_message)
