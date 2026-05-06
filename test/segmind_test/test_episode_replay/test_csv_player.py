import datetime
import time
from pathlib import Path
import pytest
import segmind
from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.players.csv_player import CSVEpisodePlayer
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
        f"{Path(segmind.__file__).parent.parent.parent}/resources/multiverse_episodes"
    )

    file_player = CSVEpisodePlayer(
        file_path=f"{multiverse_episodes_dir}/icub_montessori_no_hands/data.csv",
        world=world,
        time_between_frames=datetime.timedelta(milliseconds=1),
        position_shift=Vector3(0, 0, 0),
    )
    context = MotionStatechartContext(world=world)
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
        "episode_executor": episode_executor,
    }


def test_replay_episode(test_csv_player_context):
    file_player = test_csv_player_context["file_player"]
    episode_executor = test_csv_player_context["episode_executor"]
    file_player.start()
    assert file_player.is_alive()
    obj = episode_executor.context.world.bodies_with_collision[0]
    frozen_pose = obj.global_pose
    time.sleep(0.5)
    assert obj.global_pose != frozen_pose
    file_player.stop()
    time.sleep(0.5)
    assert len(episode_executor.context.world.bodies_with_collision) == 19
    assert not file_player.is_alive()