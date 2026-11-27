"""
Mixins for each kind of structural part of furniture and containers, such as doors, legs
or a rim.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from krrood.patterns.classproperty import classproperty
from krrood.patterns.mixin import Mixin
from semantic_digital_twin.grasping.grasp_strategies import (
    GraspStrategy,
    PinchTheTopOfAWall,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasDrawnGraspsByDefault,
    HasRootBody,
    IsGraspable,
)
from semantic_digital_twin.semantic_annotations.part_whole.relationship_marker import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.semantic_annotations.part_whole.relationship import (
    HasParts,
)

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Aperture,
        Cabinet,
        CounterTop,
        Door,
        Drawer,
        Handle,
        Leg,
        Rim,
        ShelfLayer,
        Sink,
    )

# %% Parts of furniture and containers


@dataclass(eq=False, kw_only=True)
class HasApertures(HasRootBody, HasParts, Mixin):
    """
    A mixin class for semantic annotations that have apertures.

    An aperture is an opening through this annotation's material, so mounting one
    removes its volume from the annotation's collision and visual geometry.
    """

    apertures: list[Aperture] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship(
            removes_part_geometry_from_whole=True
        ).as_dict(),
    )
    """
    The apertures of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasDrawers(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have drawers.
    """

    drawers: list[Drawer] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The drawers of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasShelfLayers(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have shelf layers.
    """

    shelf_layers: list[ShelfLayer] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The shelf layers of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasDoors(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have doors.
    """

    doors: list[Door] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The doors of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasHandle(IsGraspable, HasParts, Mixin):
    """
    A mixin class for semantic annotations that have a handle, and are grasped at it
    when they have one.
    """

    handle: Handle | None = field(
        default=None,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The handle of the semantic annotation.
    """

    def grasped_part(self) -> IsGraspable:
        """
        :return: The handle; without a handle, the part the annotation is grasped at
            otherwise.
        """
        if self.handle is None:
            return super().grasped_part()
        return self.handle


@dataclass(eq=False, kw_only=True)
class HasRim(HasDrawnGraspsByDefault, HasParts, Mixin):
    """
    A mixin class for open containers that have a rim, and are grasped at it when they
    have one.
    """

    rim: Rim | None = field(
        default=None,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The rim of the container.
    """

    def grasped_part(self) -> IsGraspable:
        """
        :return: The rim; without a rim, the part the annotation is grasped at
            otherwise.
        """
        if self.rim is None:
            return super().grasped_part()
        return self.rim

    @classproperty
    def grasp_strategy(cls) -> GraspStrategy:
        """
        :return: Pinching the top of the container's own wall, as its rim would be.
        """
        return PinchTheTopOfAWall()


@dataclass(eq=False, kw_only=True)
class HasHandles(HasRootBody, HasParts, Mixin):
    """
    A mixin class for semantic annotations that may have more than one handle.
    """

    handles: list[Handle] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The handles of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasLegs(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have legs.
    """

    legs: list[Leg] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The legs of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasSink(HasParts, Mixin):
    """
    A mixin class for semantic annotations that have a sink.
    """

    sink: Sink | None = field(
        default=None,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The sink of the semantic annotation.
    """


@dataclass(eq=False, kw_only=True)
class HasCabinets(HasParts, Mixin):
    """
    A mixin class for semantic annotations built from cabinets.
    """

    cabinets: list[Cabinet] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The cabinets the semantic annotation is built from.
    """


@dataclass(eq=False, kw_only=True)
class HasCounterTop(HasParts, Mixin):
    """
    A mixin class for semantic annotations that carry a counter top.

    Distinct from :class:`HasSupportingSurface`, which is the bare region something can
    be put on: this is the worktop as a thing in its own right, which is what a scan
    sees and what in turn carries the sink.
    """

    counter_top: CounterTop = field(
        default=None,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The counter top of the semantic annotation.
    """
