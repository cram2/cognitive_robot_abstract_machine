"""
The parser that turns one sample of the fruit-cutting generator into a world: its fruit,
each tissue mesh and rigid body as the part it is, and the qualities its record gives.
"""

from __future__ import annotations

# %% Imports

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Self

from semantic_digital_twin.adapters.fruit_cutting_research.exceptions import (
    PrimWithoutSinglePart,
)
from semantic_digital_twin.adapters.fruit_cutting_research.schema import (
    FruitCuttingMaterial,
    FruitCuttingSample,
    FruitCuttingSampleFile,
)
from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    PathResolver,
)
from semantic_digital_twin.adapters.usd.parser import USDStageParser, Usd
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Fruit,
    FruitPart,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% Model layout


class FruitCuttingPrimPath(StrEnum):
    """
    The prims the generator places every sample's model below.
    """

    SCENE = "/World/CulinaryCut"
    """
    Everything the generator places, the fruit and any rigid pit.
    """

    FRUIT = "/World/CulinaryCut/Fruit"
    """
    The fruit, holding its tissue meshes and its rigid seeds.
    """


class FruitCuttingPrimAttribute(StrEnum):
    """
    The attributes of a sample's model the parser reads.
    """

    TISSUE_ID = "primvars:tissueId"
    """
    The tissue each face of a mesh is made of, counted from 1 in the record's order.
    """


@dataclass(kw_only=True)
class FruitCuttingPartPrim:
    """
    A prim of a sample's model that becomes one part of the fruit.
    """

    prim: Usd.Prim
    """
    The prim, a tissue mesh or a rigid body.
    """

    part_type: type[FruitPart]
    """
    The type of part the prim becomes.
    """

    material: FruitCuttingMaterial
    """
    The material values of the tissue the part is made of.
    """


# %% Parser


@dataclass(kw_only=True)
class FruitCuttingSampleParser(USDStageParser):
    """
    Parses one sample of the fruit-cutting generator, a folder holding its record and
    its model, into a world with the fruit and its parts, each fixed to the fruit.

    .. seealso:: https://github.com/zhiyuan-gao/fruit-cutting-research
    """

    sample: FruitCuttingSample
    """
    The sample's record.
    """

    @classmethod
    def from_file(
        cls,
        file_path: str,
        prefix: str | None = None,
        path_resolver: PathResolver | None = None,
    ) -> Self:
        """
        Creates a parser for one sample.

        :param file_path: The folder of the sample, holding the files of
            :class:`FruitCuttingSampleFile`.
        :param prefix: The prefix for every name used in this world, the sample's name
            if not given.
        :param path_resolver: The resolver for the sample's folder and the asset
            references of its model.
        :return: A parser for the sample.
        """
        path_resolver = path_resolver or CompositePathResolver()
        folder = Path(path_resolver.resolve(file_path))
        sample = FruitCuttingSample.from_json(
            data=json.loads((folder / FruitCuttingSampleFile.RECORD).read_text())
        )
        return cls(
            stage=Usd.Stage.Open(str(folder / FruitCuttingSampleFile.MODEL)),
            sample=sample,
            prefix=prefix or sample.name,
            path_resolver=path_resolver,
        )

    def create_world(self) -> World:
        """
        Differs from
        :meth:`~semantic_digital_twin.world_source.WorldSource.create_world` in the
        samples it rejects.

        :raises PrimWithoutSinglePart: If a tissue mesh or rigid body of the model is
            made of tissues that do not make up exactly one part.
        :raises UnknownTissueId: If a mesh uses a tissue id the record has no tissue for.
        """
        fruit_type = self.sample.fruit.fruit_type
        # Every body is built before the world is modified: a World left partway
        # through a failed modification is unusable.
        part_prims = self._part_prims(fruit_type=fruit_type)
        fruit_prim = self.stage.GetPrimAtPath(FruitCuttingPrimPath.FRUIT)
        fruit_body = self._create_free_body(
            prim=fruit_prim,
            name=self._name_below_scene(prim=fruit_prim),
            excluded_prim_paths=frozenset(
                part_prim.prim.GetPath() for part_prim in part_prims
            ),
        )
        part_bodies = [
            self._create_free_body(
                prim=part_prim.prim, name=self._name_below_scene(prim=part_prim.prim)
            )
            for part_prim in part_prims
        ]
        world = self._create_world_with_root_body()
        with world.modify_world():
            self._add_free_body(world=world, prim=fruit_prim, body=fruit_body)
            fruit = fruit_type(root=fruit_body)
            world.add_semantic_annotation(fruit)
            for part_prim, part_body in zip(part_prims, part_bodies):
                self._add_part(
                    world=world, fruit=fruit, part_prim=part_prim, body=part_body
                )
            ripeness = self.sample.stage.ripeness
            if ripeness is not None:
                fruit.set_quality(quality=ripeness)
        return world

    def _part_prims(self, *, fruit_type: type[Fruit]) -> list[FruitCuttingPartPrim]:
        """
        :param fruit_type: The type of the sample's fruit.
        :return: The rigid bodies of the record, then the tissue meshes of the fruit
            that are not rigid bodies, each as the part it becomes.
        :raises PrimWithoutSinglePart: If a prim's tissues do not make up one part.
        :raises UnknownTissueId: If a mesh uses a tissue id the record has no tissue for.
        """
        rigid_body_paths = {
            rigid_body.prim_path for rigid_body in self.sample.rigid_bodies
        }
        rigid_bodies = [
            self._part_prim(
                prim=self.stage.GetPrimAtPath(rigid_body.prim_path),
                materials=[self.sample.tissue_material(tissue=rigid_body.tissue)],
                fruit_type=fruit_type,
            )
            for rigid_body in self.sample.rigid_bodies
        ]
        tissue_meshes = [
            self._part_prim(
                prim=prim,
                materials=[
                    self.sample.material_with_tissue_id(tissue_id=tissue_id)
                    for tissue_id in sorted(
                        set(
                            prim.GetAttribute(FruitCuttingPrimAttribute.TISSUE_ID).Get()
                        )
                    )
                ],
                fruit_type=fruit_type,
            )
            for prim in Usd.PrimRange(
                self.stage.GetPrimAtPath(FruitCuttingPrimPath.FRUIT)
            )
            if prim.HasAttribute(FruitCuttingPrimAttribute.TISSUE_ID)
            and str(prim.GetPath()) not in rigid_body_paths
        ]
        return rigid_bodies + tissue_meshes

    def _part_prim(
        self,
        *,
        prim: Usd.Prim,
        materials: list[FruitCuttingMaterial],
        fruit_type: type[Fruit],
    ) -> FruitCuttingPartPrim:
        """
        :param prim: A tissue mesh or rigid body of the model.
        :param materials: The materials of the tissues the prim is made of.
        :param fruit_type: The type of the sample's fruit.
        :return: The prim as the part its one tissue with a part type makes up; the
            other tissues, such as an orange's membranes, are no parts of their own.
        :raises PrimWithoutSinglePart: If not exactly one of the tissues makes up a part.
        """
        parts = [
            FruitCuttingPartPrim(prim=prim, part_type=part_type, material=material)
            for material in materials
            if (part_type := material.tissue.choose_part_type(fruit_type=fruit_type))
            is not None
        ]
        if len(parts) != 1:
            raise PrimWithoutSinglePart(
                sample=self.sample,
                prim_path=str(prim.GetPath()),
                tissues=[material.tissue for material in materials],
            )
        [part] = parts
        return part

    def _add_part(
        self,
        *,
        world: World,
        fruit: Fruit,
        part_prim: FruitCuttingPartPrim,
        body: Body,
    ) -> None:
        """
        Adds ``body`` to ``world`` at its prim's pose as a part of ``fruit``, fixed to
        it, with the qualities of its material. Must be called inside ``world``'s
        modification context.

        :param world: The world holding ``fruit``.
        :param fruit: The sample's fruit.
        :param part_prim: The prim ``body`` was created from.
        :param body: The part's body.
        """
        world.add_body(body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=body,
                parent_T_connection_expression=self._world_root_T_prim(
                    world=world, prim=part_prim.prim
                ),
            )
        )
        part = part_prim.part_type(root=body)
        world.add_semantic_annotation(part)
        fruit.add(part=part)
        for quality in (
            part_prim.material.to_mass_density(),
            part_prim.material.to_material_elasticity(),
        ):
            if quality is not None:
                part.set_quality(quality=quality)

    def _create_joint_link_body(self, *, prim: Usd.Prim) -> Body:
        return self._create_free_body(prim=prim, name=self._name_below_scene(prim=prim))

    @staticmethod
    def _name_below_scene(*, prim: Usd.Prim) -> str:
        """
        :param prim: A prim below :attr:`FruitCuttingPrimPath.SCENE`.
        :return: The prim's path below it, such as ``Fruit/Seeds/Seed_001``.
        """
        return str(prim.GetPath()).removeprefix(f"{FruitCuttingPrimPath.SCENE}/")
