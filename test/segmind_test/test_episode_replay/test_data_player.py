import datetime
import time
from os.path import dirname

import pytest

from segmind.detectors.base import SegmindContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.players.csv_player import CSVEpisodePlayer
from segmind.players.json_player import JSONPlayer
from semantic_digital_twin.adapters.package_resolver import FileUriResolver
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture(scope="function")
def test_csv_player_context():
    world = World()
    root = Body(name=PrefixedName(name="root", prefix="world"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)

    multiverse_episodes_dir = (
        f"{dirname(__file__)}/../resources/multiverse_episodes"
    )
    file_player = CSVEpisodePlayer(
        file_path=f"{multiverse_episodes_dir}/icub_montessori_no_hands/data.csv",
        world=world,
        time_between_frames=datetime.timedelta(milliseconds=1),
        position_shift=Vector3(0, 0, 0),
    )
    context = SegmindContext(world=world)
    episode_executor = EpisodeSegmenterExecutor(
        context=context,
        player=file_player,
        ignored_objects=["iCub"],
        fixed_objects=["scene"],
    )
    episode_executor.spawn_scene(
        models_dir=f"{multiverse_episodes_dir}/icub_montessori_no_hands/models/",
        file_resolver=FileUriResolver(),
    )
    return {
        "world": world,
        "file_player": file_player,
        "context": context,
        "episode_executor": episode_executor,
    }


def test_replay_episode(test_csv_player_context):
    context = test_csv_player_context["context"]
    file_player = test_csv_player_context["file_player"]
    file_player.start()
    assert file_player.is_alive()
    file_player.stop()
    time.sleep(0.5)
    assert len(context.world.bodies_with_collision) > 0
    assert not file_player.is_alive()


@pytest.fixture(scope="function")
def test_json_player_context():
    world = World()
    root = Body(name=PrefixedName(name="root", prefix="world"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)

    json_file = f"{dirname(__file__)}/../resources/fame_episodes/alessandro_with_ycp_objects_in_max_room_2/refined_poses.json"
    obj_id_to_name = {1: "obj_000001", 3: "obj_000003", 4: "obj_000004", 6: "obj_000006"}
    json_file_player = JSONPlayer(json_file, world=world,
                             time_between_frames=datetime.timedelta(milliseconds=1),
                             obj_id_to_name=obj_id_to_name)
    print("JSONPlayer symbol:", JSONPlayer)
    print("Constructed type:", type(json_file_player))
    print("MRO:", type(json_file_player).mro())
    context = SegmindContext(world=world)
    episode_segmenter = EpisodeSegmenterExecutor(player=json_file_player, context=context)
    json_file_player.transform_to_stl(f"{dirname(__file__)}/../resources/fame_episodes/alessandro_sliding_bueno/models")
    episode_segmenter.spawn_scene(
        models_dir=f"{dirname(__file__)}/../resources/fame_episodes/alessandro_sliding_bueno/models/")

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