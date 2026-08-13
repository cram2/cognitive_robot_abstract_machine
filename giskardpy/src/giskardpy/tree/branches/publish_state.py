from py_trees.composites import Sequence

from giskardpy.tree.behaviors.publish_debug_expressions import (
    QPDataPublisherConfig,
    PublishDebugExpressions,
)
from giskardpy.tree.behaviors.publish_feedback import PublishFeedback
from giskardpy.tree.behaviors.publish_joint_states import PublishWorldState
from giskardpy.tree.behaviors.tf_publisher import TFPublisher


class PublishState(Sequence):

    def __init__(self, name: str = "publish state"):
        super().__init__(name, memory=True)

    def add_publish_feedback(self):
        self.add_child(PublishFeedback())

    def add_tf_publisher(self):
        node = TFPublisher("publish tf")
        self.add_child(node)

    def add_qp_data_publisher(self, publish_config: QPDataPublisherConfig):
        node = PublishDebugExpressions(publish_config=publish_config)
        self.add_child(node)

    def add_joint_state_publisher(self):
        node = PublishWorldState()
        self.add_child(node)
