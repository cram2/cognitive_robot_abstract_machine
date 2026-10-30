import datetime
from os.path import dirname
from unittest import TestCase

import pytest
from segmind.detectors.base import SegmindContext
from cramph.executor import StatechartExecutor
from segmind.episode_segmenter import EpisodeSceneLoader, EpisodeSegmentation
from cramph.context import StatechartContext
from segmind.players.csv_player import CSVEpisodePlayer
from semantic_digital_twin.adapters.package_resolver import FileUriResolver
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from segmind.statecharts.segmind_statechart import DetectorStatechartBuilder


@pytest.fixture(scope="function")
def test_context():
    world = World()
    root = Body(name=PrefixedName(name="root", prefix="world"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)

    context = StatechartContext(world=world)
    multiverse_episodes_dir = f"{dirname(__file__)}/../resources/multiverse_episodes"
    file_player = CSVEpisodePlayer(
        file_path=f"{multiverse_episodes_dir}/icub_montessori_no_hands/data.csv",
        world=world,
        time_between_frames=datetime.timedelta(milliseconds=0.1),
        position_shift=Vector3(0, 0, 0),
    )
    episode_executor = StatechartExecutor(
        context=context, extensions=[EpisodeSegmentation(player=file_player)]
    )
    EpisodeSceneLoader(
        world=world, ignored_objects=["iCub"], fixed_objects=["scene"]
    ).spawn_scene(
        models_dir=f"{multiverse_episodes_dir}/icub_montessori_no_hands/models/",
        file_resolver=FileUriResolver(),
    )
    return {
        "world": world,
        "logger": context.require_extension(SegmindContext).logger,
        "context": context,
        "file_player": file_player,
        "episode_executor": episode_executor,
    }


@pytest.mark.skip(reason="This test takes too long to run.")
def test_replay_episode(test_context):
    context = test_context["context"]
    logger = test_context["logger"]
    executor = test_context["episode_executor"]
    executor.compile(DetectorStatechartBuilder().build(executor.context))
    assert executor.require_extension(EpisodeSegmentation).player.is_alive()
    executor.tick_until_end()
    try:
        while executor.require_extension(EpisodeSegmentation).player.is_alive():
            pass
    finally:
        assert len(logger.get_events()) > 0
