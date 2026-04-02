import numpy as np

from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.adapters.sage_10k_dataset.schema import Sage10kScene
from semantic_digital_twin.world import World


def verify_scene(world: World, scene: Sage10kScene):
    """
    Verify that the object positions of the scene are the same as in the world.
    :param world:
    :param scene:
    :return:
    """
    for room in scene.rooms:
        for obj in room.objects:
            body = world.get_body_by_name(obj.id)
            global_position = body.global_pose.to_position()
            assert np.isclose(global_position.x, obj.position.x)
            assert np.isclose(global_position.y, obj.position.y)
            assert np.isclose(global_position.z, obj.position.z)


def test_loader(rclpy_node):
    loader = Sage10kDatasetLoader()

    scene = loader.create_scene(scene_url=Sage10kDatasetLoader.available_scenes()[0])

    world = scene.create_world()
    # pub = VizMarkerPublisher(
    #     _world=world,
    #     node=rclpy_node,
    # )
    # pub.with_tf_publisher()

    verify_scene(world, scene)
