"""
Tests for parsing a sample of the fruit-cutting generator: its record and its USD model
become a fruit with its parts, their material qualities and its ripeness.
"""

from __future__ import annotations

# %% Imports

import json
import os
import shutil
from enum import StrEnum
from pathlib import Path

import pytest
from random_events.interval import Bound, SimpleInterval

from semantic_digital_twin.adapters.fruit_cutting_research.parser import (
    FruitCuttingSampleParser,
)
from semantic_digital_twin.adapters.fruit_cutting_research.schema import (
    FruitCuttingFruit,
    FruitCuttingMaterial,
    FruitCuttingSample,
    FruitCuttingSampleFile,
    FruitCuttingStage,
    FruitCuttingTissue,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    Density,
    Elasticity,
    RipenessStage,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Avocado,
    Fruit,
    FruitPart,
    Orange,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection

from .usd_stages import PXR_AVAILABLE

if PXR_AVAILABLE:
    from pxr import Sdf

pytestmark = pytest.mark.skipif(
    not PXR_AVAILABLE, reason="usd-core (pxr) not installed"
)

# %% Fixtures


class FixtureSample(StrEnum):
    """
    The hand-made samples in ``dataset/fruit_cutting``, each a stage in the layout the
    generator writes and a record taken from a real sample.
    """

    APPLE = "apple"
    """
    Two rigid seeds, at the ``reference`` stage.
    """

    AVOCADO = "avocado"
    """
    A rigid pit outside the fruit's prim, at the ``ripe`` stage.
    """

    ORANGE = "orange"
    """
    A peel made of ``Rind``, and segments whose faces are pulp and membrane.
    """

    @property
    def folder(self) -> Path:
        """
        :return: The folder holding the sample's record and its stage as text.
        """
        return Path(__file__).parent / "dataset" / "fruit_cutting" / self.value

    @property
    def record(self) -> dict:
        """
        :return: The sample's record as JSON data.
        """
        return json.loads((self.folder / FruitCuttingSampleFile.RECORD).read_text())


class FruitCuttingEnvironmentVariable(StrEnum):
    """
    The environment variables these tests read.
    """

    RELEASE_SAMPLES = "FRUIT_CUTTING_RELEASE_SAMPLES"
    """
    The folder holding the generator's samples of the released models, one folder per
    fruit; the tests on them are skipped when it is not set.
    """


def packed_sample(*, sample: FixtureSample, destination: Path) -> Path:
    """
    :param sample: A hand-made sample.
    :param destination: An empty folder to write the sample to.
    :return: The folder holding the sample's record and its stage packed as the USDZ
        file the generator writes.
    """
    shutil.copy(sample.folder / FruitCuttingSampleFile.RECORD, destination)
    writer = Sdf.ZipFileWriter.CreateNew(
        str(destination / FruitCuttingSampleFile.MODEL)
    )
    writer.AddFile(str(sample.folder / "fruit.usda"), "fruit.usda")
    writer.Save()
    return destination


def parsed_world(*, sample: FixtureSample, destination: Path) -> World:
    """
    :param sample: A hand-made sample.
    :param destination: An empty folder to write the sample to.
    :return: The world the parser builds from the sample.
    """
    folder = packed_sample(sample=sample, destination=destination)
    return FruitCuttingSampleParser.from_file(str(folder)).create_world()


def the_fruit(world: World, fruit_type: type[Fruit]) -> Fruit:
    """
    :param world: A world holding one fruit.
    :param fruit_type: The type of that fruit.
    :return: The fruit.
    """
    [fruit] = world.get_semantic_annotations_by_type(fruit_type)
    return fruit


def point(value: float) -> SimpleInterval:
    """
    :param value: A value a sample records.
    :return: The interval holding only ``value``.
    """
    return SimpleInterval.from_data(value, value, Bound.CLOSED, Bound.CLOSED)


def expected_qualities(*, record: dict, tissue: str) -> tuple[Density, Elasticity]:
    """
    :param record: A sample's record as JSON data.
    :param tissue: The name of one of its soft tissues.
    :return: The density and elasticity the record gives that tissue.
    """
    material = record["materials"][tissue]
    return Density(value=point(material["density_kg_m3"])), Elasticity(
        youngs_modulus=point(
            material["E_MPa"] * FruitCuttingMaterial.pascals_per_megapascal
        ),
        poissons_ratio=point(material["nu"]),
    )


@pytest.fixture
def apple(tmp_path: Path) -> Apple:
    """
    The apple parsed from the hand-made apple sample.
    """
    return the_fruit(
        parsed_world(sample=FixtureSample.APPLE, destination=tmp_path), Apple
    )


@pytest.fixture
def avocado(tmp_path: Path) -> Avocado:
    """
    The avocado parsed from the hand-made avocado sample.
    """
    return the_fruit(
        parsed_world(sample=FixtureSample.AVOCADO, destination=tmp_path), Avocado
    )


@pytest.fixture
def orange(tmp_path: Path) -> Orange:
    """
    The orange parsed from the hand-made orange sample.
    """
    return the_fruit(
        parsed_world(sample=FixtureSample.ORANGE, destination=tmp_path), Orange
    )


# %% The record


def test_record_names_the_fruit_stage_and_tissues_of_the_sample() -> None:
    """
    A sample's record gives its fruit and stage, its tissues in the order the tissue ids
    of its meshes count them, and the prims of its rigid bodies.
    """
    record = FixtureSample.APPLE.record

    sample = FruitCuttingSample.from_json(data=record)

    assert sample.fruit == FruitCuttingFruit(record["fruit"])
    assert sample.stage == FruitCuttingStage(record["stage"])
    assert [material.tissue for material in sample.materials] == [
        FruitCuttingTissue(name) for name in record["material_order"]
    ]
    assert [rigid_body.prim_path for rigid_body in sample.rigid_bodies] == [
        rigid_body["path"] for rigid_body in record["rigid_bodies"]
    ]


@pytest.mark.parametrize(
    "stage, ripeness_stage",
    [
        (FruitCuttingStage.GREEN, RipenessStage.UNRIPE),
        (FruitCuttingStage.YELLOW, RipenessStage.RIPE),
        (FruitCuttingStage.FIRM, RipenessStage.UNRIPE),
        (FruitCuttingStage.RIPE, RipenessStage.RIPE),
    ],
)
def test_named_stages_are_ripeness_stages(
    stage: FruitCuttingStage, ripeness_stage: RipenessStage
) -> None:
    """
    A green banana and a firm avocado are unripe; a yellow banana and a ripe avocado are
    ripe.
    """
    assert stage.ripeness.stage == ripeness_stage


def test_reference_stage_leaves_the_ripeness_unknown() -> None:
    """
    The ``reference`` stage infers no ripening, so it gives no ripeness.
    """
    assert FruitCuttingStage.REFERENCE.ripeness is None


# %% Parts


def test_apple_sample_becomes_an_apple_with_its_anatomy(apple: Apple) -> None:
    """
    Each tissue mesh and each rigid seed of the apple becomes the part it is.
    """
    assert apple.peel is not None
    assert apple.flesh is not None
    assert apple.core is not None
    assert apple.stem is not None
    assert [seed.root.name.name for seed in apple.seeds] == [
        "Fruit/Seeds/Seed_001",
        "Fruit/Seeds/Seed_002",
    ]


def test_parts_are_fixed_to_their_fruit(apple: Apple) -> None:
    """
    Every part hangs from the fruit's root by a fixed connection.
    """
    for part in apple.parts:
        assert isinstance(part.root.parent_connection, FixedConnection)
        assert part.root.parent_connection.parent is apple.root


def test_parts_keep_their_pose_in_the_stage(apple: Apple) -> None:
    """
    A seed's body lies where its prim lies in the stage: the fruit's translation plus
    its own.
    """
    [seed, _] = apple.seeds

    position = seed.root.global_transform.to_np()[:3, 3]

    assert position.tolist() == pytest.approx([0.105, 0.205, 0.055])


def test_avocado_pit_outside_the_fruit_prim_is_its_pit(avocado: Avocado) -> None:
    """
    The rigid pit, whose prim lies outside the fruit's prim, is the avocado's pit.
    """
    assert avocado.pit.root.name.name == "RigidPit"


def test_orange_pulp_comes_in_citrus_segments(orange: Orange) -> None:
    """
    The segment meshes of the orange's pulp become its citrus segments.
    """
    assert [segment.root.name.name for segment in orange.citrus_segments] == [
        "Fruit/Pulp/Segment_01",
        "Fruit/Pulp/Segment_02",
    ]


# %% Qualities


def test_tissue_values_become_the_qualities_of_their_part(apple: Apple) -> None:
    """
    The peel's density and elasticity are the values the record gives its tissue.
    """
    density, elasticity = expected_qualities(
        record=FixtureSample.APPLE.record, tissue=FruitCuttingTissue.PEEL
    )

    assert apple.peel.density == density
    assert apple.peel.elasticity == elasticity


def test_rigid_seed_has_a_density_and_no_elasticity(apple: Apple) -> None:
    """
    A rigid seed's record gives it a density but no elastic constants.
    """
    seed_material = FixtureSample.APPLE.record["materials"][FruitCuttingTissue.SEEDS]

    for seed in apple.seeds:
        assert seed.density == Density(value=point(seed_material["density_kg_m3"]))
        assert seed.elasticity is None


def test_fixed_pit_has_no_material_values(avocado: Avocado) -> None:
    """
    The record gives the fixed rigid pit no material values, so they stay unknown.
    """
    assert avocado.pit.density is None
    assert avocado.pit.elasticity is None


def test_peel_takes_the_values_of_the_tissue_its_mesh_is_made_of(
    orange: Orange,
) -> None:
    """
    The orange's peel mesh is made of ``Rind``, so the peel has the rind's values.
    """
    density, elasticity = expected_qualities(
        record=FixtureSample.ORANGE.record, tissue=FruitCuttingTissue.RIND
    )

    assert orange.peel.density == density
    assert orange.peel.elasticity == elasticity


def test_segments_take_the_values_of_their_pulp_not_their_membrane(
    orange: Orange,
) -> None:
    """
    Segment meshes hold pulp and membrane faces; the membrane has no part, so each
    segment has the pulp's values.
    """
    density, elasticity = expected_qualities(
        record=FixtureSample.ORANGE.record, tissue=FruitCuttingTissue.PULP
    )

    for segment in orange.citrus_segments:
        assert segment.density == density
        assert segment.elasticity == elasticity


def test_reference_apple_has_no_ripeness(apple: Apple) -> None:
    """
    An apple at the ``reference`` stage has an unknown ripeness.
    """
    assert apple.ripeness is None


def test_ripe_avocado_is_ripe(avocado: Avocado) -> None:
    """
    An avocado at the ``ripe`` stage is ripe.
    """
    assert avocado.ripeness.stage == RipenessStage.RIPE


# %% Released models


def release_sample_folders() -> list[Path]:
    """
    :return: The sample folders below
        :attr:`FruitCuttingEnvironmentVariable.RELEASE_SAMPLES`, none if it is not set.
    """
    root = os.environ.get(FruitCuttingEnvironmentVariable.RELEASE_SAMPLES)
    if root is None:
        return []
    return sorted(
        sample_record.parent
        for sample_record in Path(root).glob(f"*/*/{FruitCuttingSampleFile.RECORD}")
    )


@pytest.mark.skipif(
    not release_sample_folders(),
    reason=f"{FruitCuttingEnvironmentVariable.RELEASE_SAMPLES} names no samples",
)
@pytest.mark.parametrize(
    "folder",
    release_sample_folders(),
    ids=[folder.parent.name for folder in release_sample_folders()],
)
def test_released_model_becomes_its_fruit_with_qualities(folder: Path) -> None:
    """
    Each released model becomes one fruit of its type with the ripeness of its stage,
    every fruit part of the world is one of its parts, and each part's density is one
    its record gives.
    """
    sample = FruitCuttingSample.from_json(
        data=json.loads((folder / FruitCuttingSampleFile.RECORD).read_text())
    )

    world = FruitCuttingSampleParser.from_file(str(folder)).create_world()

    fruit = the_fruit(world, sample.fruit.fruit_type)
    assert fruit.ripeness == sample.stage.ripeness
    assert {id(part) for part in world.get_semantic_annotations_by_type(FruitPart)} == {
        id(part) for part in fruit.parts
    }
    recorded_densities = [material.to_density() for material in sample.materials]
    for part in fruit.parts:
        assert part.density in recorded_densities
