"""
The concept class of wholes whose parts are made of materials the literature measured,
such as the peel and the flesh of an apple.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from krrood.patterns.classproperty import classproperty
from krrood.patterns.mixin import Mixin
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.part_whole.relationship import (
    HasParts,
)
from semantic_digital_twin.semantic_annotations.qualities.measured_material import (
    MeasuredMaterial,
)

# %% Wholes of measured parts


@dataclass(eq=False, kw_only=True)
class HasMeasuredParts(HasRootBody, HasParts, Mixin):
    """
    A whole whose parts get the qualities the literature reports for their material,
    looked up from the whole, since one kind of part, such as a peel, serves many
    wholes.
    """

    @classproperty
    def measured_part_materials(cls) -> dict[type[HasRootBody], MeasuredMaterial]:
        """
        :return: For each kind of part of this kind of whole whose material was
            measured, that material.
        """
        return {}

    def set_measured_qualities(self) -> None:
        """
        Differs from
        :meth:`~semantic_digital_twin.semantic_annotations.mixins.HasRootBody.set_measured_qualities`
        in also setting, on each part whose exact type :attr:`measured_part_materials`
        maps, the qualities reported for its material.
        """
        super().set_measured_qualities()
        for part in self.parts:
            material = self.measured_part_materials.get(type(part))
            if material is not None:
                part.set_qualities_measured_for(material=material)
