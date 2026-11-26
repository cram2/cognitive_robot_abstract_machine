"""
The record ``sample.json`` the fruit-cutting generator writes beside each sample's model,
mirrored in dataclasses.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass
from enum import StrEnum

from random_events.interval import Bound, SimpleInterval
from typing_extensions import Any, Self

from krrood.patterns.classproperty import classproperty
from semantic_digital_twin.adapters.fruit_cutting_research.exceptions import (
    UnknownTissueId,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    Density,
    Elasticity,
    Ripeness,
    RipenessStage,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Avocado,
    Banana,
    CitrusFruit,
    CitrusSegment,
    Core,
    Cucumber,
    Flesh,
    Fruit,
    FruitPart,
    Orange,
    Peach,
    Peel,
    Pit,
    Pith,
    Seed,
    SeedChamber,
    Stem,
    Strawberry,
    Watermelon,
)

# %% Files and keys


class FruitCuttingSampleFile(StrEnum):
    """
    The files of one sample in its folder.
    """

    RECORD = "sample.json"
    """
    The record of what the generator drew for the sample.
    """

    MODEL = "fruit.usdz"
    """
    The sample's USD model, one mesh per tissue and one prim per rigid body.
    """


class FruitCuttingRecordKey(StrEnum):
    """
    The keys of a sample's record that the parser reads.
    """

    FRUIT = "fruit"
    """
    The fruit the sample is, a :class:`FruitCuttingFruit`.
    """

    SEED = "seed"
    """
    The seed of the random stream the sample was drawn from.
    """

    INDEX = "index"
    """
    The index of the sample among those drawn with its seed.
    """

    STAGE = "stage"
    """
    The ripening stage the sample was drawn at, a :class:`FruitCuttingStage`.
    """

    MATERIAL_ORDER = "material_order"
    """
    The names of the sample's tissues, in the order the tissue ids of its meshes count
    them from 1.
    """

    MATERIALS = "materials"
    """
    The material values of each tissue, by tissue name.
    """

    RIGID_BODIES = "rigid_bodies"
    """
    The parts simulated as rigid bodies, such as seeds and pits.
    """

    DENSITY = "density_kg_m3"
    """
    In a tissue's material values: its density in kilograms per cubic metre.
    """

    YOUNGS_MODULUS = "E_MPa"
    """
    In a tissue's material values: its Young's modulus in megapascals.
    """

    POISSONS_RATIO = "nu"
    """
    In a tissue's material values: its Poisson's ratio.
    """

    MATERIAL = "material"
    """
    In a rigid body's entry: the name of the tissue it is made of.
    """

    PATH = "path"
    """
    In a rigid body's entry: the path of its prim in the sample's model.
    """


# %% Fruits, stages and tissues


class FruitCuttingFruit(StrEnum):
    """
    The fruits the generator has templates for.
    """

    APPLE = "apple"
    """
    An apple: peel, flesh, a soft core, movable seeds and a stem.
    """

    AVOCADO = "avocado"
    """
    An avocado: peel, flesh and a fixed pit.
    """

    BANANA = "banana"
    """
    A banana: peel, its stalk included, and flesh.
    """

    CUCUMBER = "cucumber"
    """
    A cucumber: peel, flesh and a soft seed chamber.
    """

    MELON = "melon"
    """
    A watermelon: a rind of two layers, flesh and movable seeds.
    """

    ORANGE = "orange"
    """
    An orange: peel and pulp in segments, each wrapped in a membrane.
    """

    PEACH = "peach"
    """
    A peach: peel, flesh, a fixed stone and a stem.
    """

    STRAWBERRY = "strawberry"
    """
    A strawberry: an outer layer, flesh and a central pith.
    """

    @property
    def fruit_type(self) -> type[Fruit]:
        """
        :return: The semantic annotation type of the fruit.
        """
        return {
            FruitCuttingFruit.APPLE: Apple,
            FruitCuttingFruit.AVOCADO: Avocado,
            FruitCuttingFruit.BANANA: Banana,
            FruitCuttingFruit.CUCUMBER: Cucumber,
            FruitCuttingFruit.MELON: Watermelon,
            FruitCuttingFruit.ORANGE: Orange,
            FruitCuttingFruit.PEACH: Peach,
            FruitCuttingFruit.STRAWBERRY: Strawberry,
        }[self]


class FruitCuttingStage(StrEnum):
    """
    The ripening stages the templates name, each with its own material ranges.
    """

    REFERENCE = "reference"
    """
    The single appearance of a fruit whose template infers no ripening.
    """

    GREEN = "green"
    """
    A green banana.
    """

    YELLOW = "yellow"
    """
    A yellow banana.
    """

    FIRM = "firm"
    """
    A firm avocado.
    """

    RIPE = "ripe"
    """
    A ripe avocado.
    """

    @property
    def ripeness(self) -> Ripeness | None:
        """
        :return: The ripeness of a fruit at this stage, ``None`` for
            :attr:`REFERENCE`, which says nothing about ripening.
        """
        ripeness_stages = {
            FruitCuttingStage.GREEN: RipenessStage.UNRIPE,
            FruitCuttingStage.YELLOW: RipenessStage.RIPE,
            FruitCuttingStage.FIRM: RipenessStage.UNRIPE,
            FruitCuttingStage.RIPE: RipenessStage.RIPE,
        }
        if self not in ripeness_stages:
            return None
        return Ripeness.from_stage(stage=ripeness_stages[self])


class FruitCuttingTissue(StrEnum):
    """
    The tissues the templates give materials, by the names their records use.
    """

    PEEL = "Peel"
    """
    The skin of an apple, avocado, banana, cucumber, peach or strawberry.
    """

    RIND = "Rind"
    """
    The peel of an orange or a watermelon.
    """

    GREEN_RIND = "GreenRind"
    """
    The outer layer of a watermelon's rind, which shares the rind's mesh.
    """

    PULP = "Pulp"
    """
    The flesh of every fruit, divided into segments in an orange.
    """

    MEMBRANE = "Membrane"
    """
    The membrane around each segment of an orange, which shares the segment's mesh.
    """

    CORE = "Core"
    """
    The soft core of an apple.
    """

    SEEDS = "Seeds"
    """
    The movable seeds of an apple, each a rigid body.
    """

    SEED = "Seed"
    """
    The movable seeds of a watermelon, each a rigid body.
    """

    STEM = "Stem"
    """
    The stalk of an apple or a peach.
    """

    PIT = "Pit"
    """
    The fixed pit of an avocado, a rigid body outside the fruit's prim.
    """

    STONE = "Stone"
    """
    The fixed stone of a peach, a rigid body.
    """

    SEED_CHAMBER = "SeedChamber"
    """
    The soft, continuous seed chamber of a cucumber.
    """

    PITH = "Pith"
    """
    The central pith of a strawberry.
    """

    def choose_part_type(self, *, fruit_type: type[Fruit]) -> type[FruitPart] | None:
        """
        :param fruit_type: The type of the fruit the tissue belongs to.
        :return: The type of part the tissue makes up in that fruit, ``None`` for a
            tissue that is no part of its own. A citrus fruit's pulp is divided into
            segments.
        """
        if self is FruitCuttingTissue.PULP and issubclass(fruit_type, CitrusFruit):
            return CitrusSegment
        return {
            FruitCuttingTissue.PEEL: Peel,
            FruitCuttingTissue.RIND: Peel,
            FruitCuttingTissue.PULP: Flesh,
            FruitCuttingTissue.CORE: Core,
            FruitCuttingTissue.SEEDS: Seed,
            FruitCuttingTissue.SEED: Seed,
            FruitCuttingTissue.STEM: Stem,
            FruitCuttingTissue.PIT: Pit,
            FruitCuttingTissue.STONE: Pit,
            FruitCuttingTissue.SEED_CHAMBER: SeedChamber,
            FruitCuttingTissue.PITH: Pith,
        }.get(self)


# %% Record


@dataclass(kw_only=True)
class FruitCuttingMaterial:
    """
    The material values a sample was simulated with for one tissue, each drawn from the
    range its template gives.
    """

    tissue: FruitCuttingTissue
    """
    The tissue the values belong to.
    """

    density: float | None = None
    """
    The density in kilograms per cubic metre, ``None`` for a tissue simulated as a fixed
    rigid body.
    """

    youngs_modulus: float | None = None
    """
    Young's modulus in megapascals, ``None`` for a tissue simulated as a rigid body.
    """

    poissons_ratio: float | None = None
    """
    Poisson's ratio, ``None`` for a tissue simulated as a rigid body.
    """

    @classmethod
    def from_json(cls, *, tissue: FruitCuttingTissue, data: dict[str, Any]) -> Self:
        """
        :param tissue: The tissue the values belong to.
        :param data: The tissue's entry in the record's ``materials``.
        :return: The tissue's material values.
        """
        return cls(
            tissue=tissue,
            density=data.get(FruitCuttingRecordKey.DENSITY),
            youngs_modulus=data.get(FruitCuttingRecordKey.YOUNGS_MODULUS),
            poissons_ratio=data.get(FruitCuttingRecordKey.POISSONS_RATIO),
        )

    @classproperty
    def pascals_per_megapascal(cls) -> float:
        """
        :return: The pascals in one megapascal, the unit of :attr:`youngs_modulus`.
        """
        return 1.0e6

    def to_density(self) -> Density | None:
        """
        :return: The density of the tissue, ``None`` if the record gives none.
        """
        if self.density is None:
            return None
        return Density(value=self._point(value=self.density))

    def to_elasticity(self) -> Elasticity | None:
        """
        :return: The elasticity of the tissue, ``None`` if the record gives no elastic
            constants.
        """
        if self.youngs_modulus is None or self.poissons_ratio is None:
            return None
        return Elasticity(
            youngs_modulus=self._point(
                value=self.youngs_modulus * self.pascals_per_megapascal
            ),
            poissons_ratio=self._point(value=self.poissons_ratio),
        )

    @staticmethod
    def _point(*, value: float) -> SimpleInterval:
        """
        :param value: A value the sample was simulated with.
        :return: The interval holding only ``value``: a sample is one draw, not a range.
        """
        return SimpleInterval.from_data(value, value, Bound.CLOSED, Bound.CLOSED)


@dataclass(kw_only=True)
class FruitCuttingRigidBody:
    """
    A part of a sample simulated as a rigid body, such as a seed or a pit.
    """

    tissue: FruitCuttingTissue
    """
    The tissue the rigid body is made of.
    """

    prim_path: str
    """
    The path of the rigid body's prim in the sample's model.
    """

    @classmethod
    def from_json(cls, *, data: dict[str, Any]) -> Self:
        """
        :param data: One entry of the record's ``rigid_bodies``.
        :return: The rigid body.
        """
        return cls(
            tissue=FruitCuttingTissue(data[FruitCuttingRecordKey.MATERIAL]),
            prim_path=data[FruitCuttingRecordKey.PATH],
        )


@dataclass(kw_only=True)
class FruitCuttingSample:
    """
    The record of one sample: which fruit it is, at which stage, and the material values
    of its tissues.
    """

    fruit: FruitCuttingFruit
    """
    The fruit the sample is.
    """

    seed: int
    """
    The seed of the generator's random stream.
    """

    index: int
    """
    The index of the sample among those drawn with :attr:`seed`.
    """

    stage: FruitCuttingStage
    """
    The ripening stage the sample was drawn at.
    """

    materials: list[FruitCuttingMaterial]
    """
    The material values of each tissue, in the order the tissue ids of the model's
    meshes count them from 1.
    """

    rigid_bodies: list[FruitCuttingRigidBody]
    """
    The parts simulated as rigid bodies.
    """

    @classmethod
    def from_json(cls, *, data: dict[str, Any]) -> Self:
        """
        :param data: The record as JSON data.
        :return: The sample the record describes.
        """
        materials = data[FruitCuttingRecordKey.MATERIALS]
        return cls(
            fruit=FruitCuttingFruit(data[FruitCuttingRecordKey.FRUIT]),
            seed=data[FruitCuttingRecordKey.SEED],
            index=data[FruitCuttingRecordKey.INDEX],
            stage=FruitCuttingStage(data[FruitCuttingRecordKey.STAGE]),
            materials=[
                FruitCuttingMaterial.from_json(
                    tissue=FruitCuttingTissue(name), data=materials[name]
                )
                for name in data[FruitCuttingRecordKey.MATERIAL_ORDER]
            ],
            rigid_bodies=[
                FruitCuttingRigidBody.from_json(data=rigid_body)
                for rigid_body in data[FruitCuttingRecordKey.RIGID_BODIES]
            ],
        )

    @property
    def name(self) -> str:
        """
        :return: The name of the sample, such as ``apple_2026_0000``: its fruit, its
            seed and its index.
        """
        return f"{self.fruit}_{self.seed}_{self.index:04d}"

    def material_with_tissue_id(self, *, tissue_id: int) -> FruitCuttingMaterial:
        """
        :param tissue_id: A tissue id a mesh of the model gives its faces.
        :return: The material values of that tissue.
        :raises UnknownTissueId: If the record has no tissue with that id.
        """
        if not 1 <= tissue_id <= len(self.materials):
            raise UnknownTissueId(sample=self, tissue_id=tissue_id)
        return self.materials[tissue_id - 1]

    def tissue_material(self, *, tissue: FruitCuttingTissue) -> FruitCuttingMaterial:
        """
        :param tissue: One of the sample's tissues.
        :return: Its material values.
        """
        [material] = [
            material for material in self.materials if material.tissue is tissue
        ]
        return material
