import datetime
import time
from pathlib import Path
import pytest
import segmind
from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.players.json_player import JSONPlayer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture(scope="session")
def test_json_player_context():
    world = World()
    root = Body(name=PrefixedName(name="root", prefix="world"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)

    json_file = f"{Path(segmind.__file__).parent.parent.parent}/resources/fame_episodes/alessandro_with_ycp_objects_in_max_room_2/refined_poses.json"
    obj_id_to_name = {1: "obj_000001", 3: "obj_000003", 4: "obj_000004", 6: "obj_000006"}
    json_file_player = JSONPlayer(json_file, world=world,
                             time_between_frames=datetime.timedelta(milliseconds=1),
                             obj_id_to_name=obj_id_to_name)
    print("JSONPlayer symbol:", JSONPlayer)
    print("Constructed type:", type(json_file_player))
    print("MRO:", type(json_file_player).mro())
    context = MotionStatechartContext(world=world)
    episode_segmenter = EpisodeSegmenterExecutor(player=json_file_player, context=context)
    json_file_player.transform_to_stl(f"{Path(segmind.__file__).parent.parent.parent}/resources/fame_episodes/alessandro_sliding_bueno/models")
    episode_segmenter.spawn_scene(
        models_dir=f"{Path(segmind.__file__).parent.parent.parent}/resources/fame_episodes/alessandro_sliding_bueno/models/")

    return {"world": world, "json_file_player": json_file_player, "context": context, "episode_segmenter": episode_segmenter}


def test_json_player(test_json_player_context):
    context = test_json_player_context["context"]
    file_player = test_json_player_context["json_file_player"]
    file_player.start()
    assert file_player.is_alive()
    file_player.stop()
    time.sleep(0.5)
    assert len(context.world.bodies_with_collision) > 0
    assert not file_player.is_alive()