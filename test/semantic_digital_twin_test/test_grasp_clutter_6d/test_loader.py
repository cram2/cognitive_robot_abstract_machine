import re

import pytest
from huggingface_hub.errors import HfHubHTTPError
from requests import HTTPError

from semantic_digital_twin.adapters.grasp_clutter_6d_dataset.loader import (
    GraspClutter6DDatasetLoader,
    GraspClutter6DModelVariant,
)

SCENE_ID_PATTERN = re.compile(r"^\d{6}$")


@pytest.fixture(scope="session")
def loader(tmp_path_factory):
    directory = tmp_path_factory.mktemp("graspclutter6d-dataset")
    return GraspClutter6DDatasetLoader(directory=directory)


def _skip_on_network_error(callable_):
    try:
        return callable_()
    except (HTTPError, HfHubHTTPError) as e:
        pytest.skip(f"GraspClutter6D dataset not available: {e}")


def test_model_variant_enum_has_the_expected_members():
    assert {v.value for v in GraspClutter6DModelVariant} == {
        "models",
        "models_eval",
        "models_m",
        "models_obj",
        "models_obj_eval",
        "models_obj_m",
    }


@pytest.mark.parametrize("object_set,split", [("grasp", "train"), ("ycbv", "test")])
def test_available_scene_ids(loader, object_set, split):
    scene_ids = _skip_on_network_error(
        lambda: loader.available_scene_ids(object_set=object_set, split=split)
    )
    assert len(scene_ids) > 0
    assert all(SCENE_ID_PATTERN.match(scene_id) for scene_id in scene_ids)


def test_object_ids_for_scene(loader):
    scene_ids = _skip_on_network_error(
        lambda: loader.available_scene_ids(object_set="grasp", split="train")
    )
    object_ids = _skip_on_network_error(
        lambda: loader.object_ids_for_scene(scene_ids[0])
    )
    assert len(object_ids) > 0
    assert all(isinstance(object_id, int) for object_id in object_ids)


def test_download_models_eval(loader):
    models_directory = _skip_on_network_error(
        lambda: loader.download_models(GraspClutter6DModelVariant.EVAL)
    )
    assert (models_directory / "models_info.json").is_file()
    mesh_files = list(models_directory.glob("obj_*.ply"))
    assert len(mesh_files) > 0
