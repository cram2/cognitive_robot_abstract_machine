from __future__ import annotations

import pytest
from rclpy.node import Node

from semantic_digital_twin.adapters.ros.input_synchronization import (
    TfFrameSynchronizer,
)
from semantic_digital_twin.adapters.ros.latest_message_subscriber import (
    LatestMessageSubscriber,
)
from semantic_digital_twin.adapters.ros.ros2_node import Ros2Node
from semantic_digital_twin.adapters.ros.tf_publisher import (
    TFPublisher,
    TfPublisherModelCallback,
)
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.collision_viz_marker import (
    CollisionVisualizationMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_publisher import (
    SpatialTypePublisher,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer

# %% the node field lives on Ros2Node alone

node_users = [
    LatestMessageSubscriber,
    TfFrameSynchronizer,
    TfPublisherModelCallback,
    TFPublisher,
    VizMarkerPublisher,
    CollisionVisualizationMarkerPublisher,
    SpatialTypePublisher,
    FetchWorldServer,
    TFWrapper,
]


@pytest.mark.parametrize("node_user", node_users)
def test_classes_communicating_over_a_node_are_ros2_nodes(node_user):
    assert issubclass(node_user, Ros2Node)


@pytest.mark.parametrize("node_user", node_users)
def test_node_field_is_declared_only_on_ros2_node(node_user):
    assert (
        node_user.__dataclass_fields__["node"] is Ros2Node.__dataclass_fields__["node"]
    )


def test_node_is_taken_by_keyword(rclpy_node: Node):
    assert TFWrapper(node=rclpy_node).node is rclpy_node
