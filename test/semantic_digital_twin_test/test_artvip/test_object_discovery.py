import pytest

from semantic_digital_twin.adapters.artvip_dataset.exceptions import (
    ArtVipMainStageFileAmbiguousError,
)
from semantic_digital_twin.adapters.artvip_dataset.loader import ArtVipDatasetLoader
from semantic_digital_twin.adapters.artvip_dataset.schema import ArtVipCategory

PREFIX = "Articulated_objects/small_furniture/"


def test_object_names_finds_a_flat_object_by_its_main_usd_file():
    files = (
        f"{PREFIX}cabinet_1/model_cabinet_1.usd",
        f"{PREFIX}cabinet_1/resource/material.usd",
        f"{PREFIX}cabinet_1/resource/v_1/link_0.usd",
    )
    assert ArtVipDatasetLoader._object_names(files, PREFIX) == ("cabinet_1",)


def test_object_names_ignores_usd_files_nested_under_resource():
    files = (f"{PREFIX}cabinet_1/resource/material.usd",)
    assert ArtVipDatasetLoader._object_names(files, PREFIX) == ()


def test_object_names_includes_a_nested_subcategory_segment():
    files = (
        "Articulated_objects/major_appliances/refrigerator/fridge/fridge_01/"
        "model_fridge_01.usd",
        "Articulated_objects/major_appliances/refrigerator/refrigerator/"
        "refrigerator_02/model_refrigerator_02.usd",
    )
    names = ArtVipDatasetLoader._object_names(
        files, "Articulated_objects/major_appliances/"
    )
    assert names == (
        "refrigerator/fridge/fridge_01",
        "refrigerator/refrigerator/refrigerator_02",
    )


def test_object_names_does_not_collapse_nested_objects_into_their_subcategory():
    # Before the fix, splitting on the first "/" after the category prefix returned
    # "refrigerator" as if it were one object, hiding both real objects beneath it.
    files = (
        "Articulated_objects/major_appliances/refrigerator/fridge/fridge_01/"
        "model_fridge_01.usd",
    )
    names = ArtVipDatasetLoader._object_names(
        files, "Articulated_objects/major_appliances/"
    )
    assert "refrigerator" not in names


def test_object_names_finds_a_main_file_that_does_not_use_the_model_prefix():
    files = (f"{PREFIX}cabinet_1/cabinet_1.usd",)
    assert ArtVipDatasetLoader._object_names(files, PREFIX) == ("cabinet_1",)


def test_main_stage_file_picks_the_one_top_level_usd_file():
    files = [
        f"{PREFIX}cabinet_1/model_cabinet_1.usd",
        f"{PREFIX}cabinet_1/resource/material.usd",
    ]
    assert (
        ArtVipDatasetLoader._main_stage_file(
            files,
            f"{PREFIX}cabinet_1/",
            category=ArtVipCategory.SMALL_FURNITURE,
            name="cabinet_1",
        )
        == f"{PREFIX}cabinet_1/model_cabinet_1.usd"
    )


def test_main_stage_file_does_not_require_the_model_prefix_naming_convention():
    files = [f"{PREFIX}cabinet_1/cabinet_1.usd"]
    assert (
        ArtVipDatasetLoader._main_stage_file(
            files,
            f"{PREFIX}cabinet_1/",
            category=ArtVipCategory.SMALL_FURNITURE,
            name="cabinet_1",
        )
        == f"{PREFIX}cabinet_1/cabinet_1.usd"
    )


def test_main_stage_file_raises_when_no_top_level_usd_file_exists():
    files = [f"{PREFIX}cabinet_1/resource/material.usd"]
    with pytest.raises(ArtVipMainStageFileAmbiguousError):
        ArtVipDatasetLoader._main_stage_file(
            files,
            f"{PREFIX}cabinet_1/",
            category=ArtVipCategory.SMALL_FURNITURE,
            name="cabinet_1",
        )


def test_main_stage_file_raises_when_more_than_one_top_level_usd_file_exists():
    files = [
        f"{PREFIX}cabinet_1/model_cabinet_1.usd",
        f"{PREFIX}cabinet_1/cabinet_1.usd",
    ]
    with pytest.raises(ArtVipMainStageFileAmbiguousError) as excinfo:
        ArtVipDatasetLoader._main_stage_file(
            files,
            f"{PREFIX}cabinet_1/",
            category=ArtVipCategory.SMALL_FURNITURE,
            name="cabinet_1",
        )
    assert set(excinfo.value.candidates) == set(files)
