import copy

import pytest

import semantic_digital_twin.orm.ormatic_interface  # type: ignore  # noqa: F401
from semantic_digital_twin.adapters.robocasa_dataset.loader import RoboCasaDatasetLoader
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaObjectCategory,
)

from experiments.confidence_aware_eql.confidence_model import ConfidenceModel

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

MAX_INSTANCES_PER_CLASS = 5
"""How many of the downloaded instances of each class this test suite loads, capping
a class with many more downloaded instances than this to keep the suite fast."""


@pytest.fixture(scope="session")
def robocasa_loader() -> RoboCasaDatasetLoader:
    """A loader for the downloaded robocasa kitchen assets, or a skip if none are present."""
    loader = RoboCasaDatasetLoader()
    if not loader.directory.exists():
        pytest.skip(
            "RoboCasa assets not downloaded. Run 'python -m robocasa.scripts."
            "download_kitchen_assets --type objs_objaverse objs_lw' first."
        )
    return loader


def _load_instances(loader, category, count):
    """Load ``count`` real instances of one robocasa object category as their semantic annotation."""
    return [
        loader.load_object_annotation(category, instance_index=index)
        for index in range(count)
    ]


@pytest.fixture(scope="session")
def real_cups_and_pots(robocasa_loader):
    """Real downloaded cup and pot instances, or a skip if either has none downloaded."""
    cup_count = min(
        robocasa_loader.downloaded_instance_count(RoboCasaObjectCategory.CUP),
        MAX_INSTANCES_PER_CLASS,
    )
    pot_count = min(
        robocasa_loader.downloaded_instance_count(RoboCasaObjectCategory.POT),
        MAX_INSTANCES_PER_CLASS,
    )
    if cup_count == 0 or pot_count == 0:
        pytest.skip(
            "No downloaded cup or pot instances found under the robocasa assets directory."
        )
    cups = _load_instances(robocasa_loader, RoboCasaObjectCategory.CUP, cup_count)
    pots = _load_instances(robocasa_loader, RoboCasaObjectCategory.POT, pot_count)
    return cups, pots


def test_familiar_real_cup_is_accepted(real_cups_and_pots):
    """A real cup the confidence model was fitted on is judged familiar."""
    cups, pots = real_cups_and_pots
    model = ConfidenceModel.fit_from_instances(cups + pots)
    assert model.is_familiar(cups[0])


def test_cup_with_pot_sized_volume_is_rejected(real_cups_and_pots):
    """A real cup whose collision geometry is replaced with a real pot's is no longer familiar.

    Grafting a pot's collision geometry onto an otherwise real cup produces an object
    whose class claims "Cup" but whose volume matches a pot instead - exactly the kind
    of out-of-distribution object the confidence model exists to catch.
    """
    cups, pots = real_cups_and_pots
    model = ConfidenceModel.fit_from_instances(cups + pots)
    anomalous_cup = copy.deepcopy(cups[-1])
    anomalous_cup.root.collision = pots[0].root.collision
    assert not model.is_familiar(anomalous_cup)
