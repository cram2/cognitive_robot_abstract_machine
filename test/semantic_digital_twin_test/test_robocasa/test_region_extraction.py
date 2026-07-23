import numpy as np
import pytest

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

from semantic_digital_twin.adapters.robocasa_dataset.loader import (
    RoboCasaDatasetLoader,
)
from semantic_digital_twin.adapters.robocasa_dataset.region_extraction import (
    GripperExclusionZoneReader,
    PlacementSamplerRegionReader,
)
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    GripperExclusionZone,
    PlacementArea,
)
from semantic_digital_twin.world_description.geometry import Box, Sphere


@pytest.fixture(scope="session")
def robocasa_loader() -> RoboCasaDatasetLoader:
    loader = RoboCasaDatasetLoader()
    if not loader.directory.exists():
        pytest.skip(
            "RoboCasa assets not downloaded. Run "
            "'python -m robocasa.scripts.download_kitchen_assets' first."
        )
    return loader


@pytest.fixture
def turn_on_microwave_task(robocasa_loader):
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    environment = robocasa_loader.build_task_environment(
        "TurnOnMicrowave", layout_id=LayoutType.LAYOUT001, style_id=StyleType.STYLE001
    )
    task = robocasa_loader.load_task_from_environment(environment)
    return environment, task


def test_placement_sampler_regions_are_annotated_and_attached_at_their_resolved_pose(
    turn_on_microwave_task,
):
    """
    A placement region read from the live environment is attached as a PlacementArea
    annotation, naming the object it was resolved for, and its region lands in the world
    at the exact position its sampler resolved.
    """
    environment, task = turn_on_microwave_task

    regions = PlacementSamplerRegionReader().read_all(
        environment, name_prefix="TurnOnMicrowave"
    )
    assert len(regions) > 0

    for region_data in regions:
        annotation = region_data.attach_to(task.world)
        assert isinstance(annotation, PlacementArea)
        assert annotation in task.world.semantic_annotations
        assert annotation.placed_object is task.world.get_body_by_name(
            region_data.body_name
        )

        region = annotation.root
        assert region in task.world.regions
        assert len(region.area.shapes) == 1
        assert isinstance(region.area.shapes[0], Box)

        world_position = region.global_pose.to_position().to_np()[:3]
        assert np.allclose(
            world_position,
            region_data.world_pose.to_position().to_np()[:3],
            atol=1e-6,
        )


def test_gripper_exclusion_zone_is_annotated_and_attached_at_the_bodys_resolved_position(
    turn_on_microwave_task,
):
    """
    A gripper exclusion zone read from the live environment is attached as a
    GripperExclusionZone annotation, naming the object the gripper is kept away from,
    with a region that is a sphere centred on that object's resolved world position.
    """
    environment, task = turn_on_microwave_task

    zone_data = GripperExclusionZoneReader.read(
        environment, name_prefix="TurnOnMicrowave", object_name="obj", radius=0.25
    )
    annotation = zone_data.attach_to(task.world)

    assert isinstance(annotation, GripperExclusionZone)
    assert annotation in task.world.semantic_annotations
    assert annotation.excluded_object is task.world.get_body_by_name(
        zone_data.body_name
    )

    region = annotation.root
    assert region in task.world.regions
    assert len(region.area.shapes) == 1
    sphere = region.area.shapes[0]
    assert isinstance(sphere, Sphere)
    assert sphere.radius == 0.25

    world_position = region.global_pose.to_position().to_np()[:3]
    assert np.allclose(world_position, zone_data.center.to_np()[:3], atol=1e-6)

