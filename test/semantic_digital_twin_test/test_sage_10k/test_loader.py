from krrood.adapters.json_serializer import from_json
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.adapters.sage_10k_dataset.schema import Sage10kScene


def test_loader(rclpy_node):
    loader = Sage10kDatasetLoader(
        scene_url="https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_020526_layout_84b703fb.zip",
    )
    target_path = loader._download_scene()
    unzipped = loader._unzip_scene(target_path)
    scene = loader._parse_json(unzipped)
    world = scene.create_world()
    # pub = VizMarkerPublisher(
    #     _world=world,
    #     node=rclpy_node,
    # )
    # pub.with_tf_publisher()
