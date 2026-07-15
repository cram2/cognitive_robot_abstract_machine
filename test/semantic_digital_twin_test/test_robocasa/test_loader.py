import pytest

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

from semantic_digital_twin.adapters.robocasa_dataset.exceptions import (
    RoboCasaTaskNotFoundError,
)
from semantic_digital_twin.adapters.robocasa_dataset.loader import (
    RoboCasaDatasetLoader,
)
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaKitchenApplianceCategory,
    RoboCasaObjectCategory,
)
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Dishwasher


@pytest.fixture(scope="session")
def robocasa_loader() -> RoboCasaDatasetLoader:
    loader = RoboCasaDatasetLoader()
    if not loader.directory.exists():
        pytest.skip(
            "RoboCasa assets not downloaded. Run "
            "'python -m robocasa.scripts.download_kitchen_assets' first."
        )
    return loader


def test_load_kitchen(robocasa_loader):
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    world = robocasa_loader.load_kitchen(
        layout_id=next(iter(LayoutType)), style_id=next(iter(StyleType))
    )
    assert len(world.bodies) > 0
    assert len(world.semantic_annotations) > 0


def test_load_kitchen_appliance(robocasa_loader):
    world = robocasa_loader.load_kitchen_appliance(
        RoboCasaKitchenApplianceCategory.CABINET
    )
    assert len(world.bodies) > 0
    assert len(world.semantic_annotations) >= 1


def test_load_kitchen_appliance_attaches_matching_annotation(robocasa_loader):
    """The loaded appliance is annotated with the semantic type of the requested category."""
    world = robocasa_loader.load_kitchen_appliance(
        RoboCasaKitchenApplianceCategory.DISHWASHER
    )
    assert any(
        isinstance(annotation, Dishwasher)
        for annotation in world.semantic_annotations
    )


def test_load_object(robocasa_loader):
    world = robocasa_loader.load_object(RoboCasaObjectCategory.APPLE)
    assert len(world.bodies) > 0
    annotations = world.semantic_annotations
    assert len(annotations) == 1
    assert not isinstance(annotations[0], NaturalLanguageWithTypeDescription)


def test_load_object_from_group_without_objaverse_assets(robocasa_loader):
    """A category with no objaverse assets still loads from another self-contained group."""
    world = robocasa_loader.load_object(RoboCasaObjectCategory.POT)
    assert len(world.bodies) > 0


def test_load_task_binds_instruction_and_world(robocasa_loader):
    """The loaded task carries its natural-language instruction bound to the scene it plays out in."""
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    task = robocasa_loader.load_task(
        "TurnOnMicrowave",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
    )
    assert "microwave" in task.instruction.lower()
    assert len(task.world.bodies) > 0
    assert len(task.world.semantic_annotations) > 0


def test_load_task_omits_robot(robocasa_loader):
    """The task world contains no robot, since semantic_digital_twin owns the robot."""
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    task = robocasa_loader.load_task(
        "TurnOnMicrowave",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
    )
    assert not any(
        body.name.name.startswith(RoboCasaDatasetLoader.robot_body_name_prefixes)
        for body in task.world.bodies
    )


def test_load_task_manipulated_objects_are_in_world(robocasa_loader):
    """The task's manipulated objects are resolved to bodies present in the task world."""
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    task = robocasa_loader.load_task(
        "TurnOnMicrowave",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
    )
    assert len(task.manipulated_objects) >= 1
    for manipulated_object in task.manipulated_objects:
        assert manipulated_object in task.world.bodies


def test_load_task_unknown_name_raises(robocasa_loader):
    """Requesting a task that is not registered raises a descriptive error."""
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    with pytest.raises(RoboCasaTaskNotFoundError):
        robocasa_loader.load_task(
            "ThisTaskDoesNotExist",
            layout_id=LayoutType.LAYOUT001,
            style_id=StyleType.STYLE001,
        )
