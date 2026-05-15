import rclpy
from rclpy.node import Node
import os
import sys

# Get the project root directory relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Add the project's src directory to sys.path
sys.path.append(os.path.join(PROJECT_ROOT, "semantic_digital_twin", "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "krrood", "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "pycram", "src"))

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.visualization.interactive_marker import (
    InteractiveMarkerPublisher,
)


def main():
    rclpy.init()
    node = Node("interactive_marker_demo")

    # Load apartment world
    apartment_urdf_path = os.path.join(
        PROJECT_ROOT, "pycram", "resources", "worlds", "kitchen.urdf"
    )
    if not os.path.exists(apartment_urdf_path):
        print(f"Error: Could not find apartment URDF at {apartment_urdf_path}")
        return

    print(f"Loading apartment from {apartment_urdf_path}...")
    world = URDFParser.from_file(apartment_urdf_path).parse()

    # Setup publishers
    print("Setting up publishers...")
    tf_publisher = TFPublisher(_world=world, node=node)
    viz_publisher = VizMarkerPublisher(_world=world, node=node)
    interactive_publisher = InteractiveMarkerPublisher(_world=world, node=node)

    print("Interactive Marker Demo is running.")
    print("Please open RViz2 and add:")
    print(" - TF plugin")
    print(
        " - MarkerArray plugin (topic: /semworld/viz_marker, durability: Transient Local)"
    )
    print(
        " - InteractiveMarkers plugin (update topic: /semworld/interactive_markers/update)"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
