from pathlib import Path

import pytest

from semantic_digital_twin.adapters.artvip_dataset.loader import ArtVipDatasetLoader
from semantic_digital_twin.adapters.artvip_dataset.schema import ArtVipCategory
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.world_description.connections import FixedConnection

from ..test_adapters.usd_stages import PXR_AVAILABLE, build_single_joint_stage

pytestmark = pytest.mark.skipif(not PXR_AVAILABLE, reason="usd-core (pxr) not installed")


@pytest.fixture
def main_usd_path(tmp_path) -> Path:
    stage = build_single_joint_stage("FixedJoint")
    path = tmp_path / "model_test_object.usd"
    stage.GetRootLayer().Export(str(path))
    return path


def test_load_delegates_stage_parsing_to_usd_parser(monkeypatch, main_usd_path):
    # ArtVipDatasetLoader.load() must build the World by parsing the downloaded main
    # stage with USDParser, not by re-implementing USD-to-World conversion itself: the
    # stage's own joint graph (built independently by USDParser and exercised
    # separately in test_adapters/test_usd.py) is what a body/connection count here
    # would already be checking.
    loader = ArtVipDatasetLoader()
    monkeypatch.setattr(
        loader, "_download_object_if_not_exists", lambda category, name: main_usd_path
    )

    result = loader.load(ArtVipCategory.SMALL_FURNITURE, "test_object")

    assert len(result.world.bodies) == 2  # root + child
    [connection] = result.world.connections
    assert isinstance(connection, FixedConnection)


def test_load_annotates_the_root_body_with_category_and_name(
    monkeypatch, main_usd_path
):
    loader = ArtVipDatasetLoader()
    monkeypatch.setattr(
        loader, "_download_object_if_not_exists", lambda category, name: main_usd_path
    )

    result = loader.load(ArtVipCategory.SMALL_FURNITURE, "test_object")

    [annotation] = [
        annotation
        for annotation in result.world.semantic_annotations
        if isinstance(annotation, NaturalLanguageWithTypeDescription)
    ]
    assert annotation.root is result.world.root
    assert annotation.description == "test_object"
    assert annotation.type_description == ArtVipCategory.SMALL_FURNITURE.value
