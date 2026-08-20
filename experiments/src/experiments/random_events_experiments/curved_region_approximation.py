"""
This module measures how much of a curved region an axis-aligned representation
captures, and what capturing more of it costs.

A predicate thresholding a Euclidean distance describes a ball, and no box is a ball.
But an :class:`~random_events.product_algebra.Event` is a *union* of boxes rather than a
single box, so the fraction of a ball its representation captures is not fixed:
subdividing the region into more boxes captures more of it. The fidelity a survey
reports for such a predicate is therefore a consequence of how many boxes it spends, not
a limit of the representation.

What this measures is the exchange rate between the two: for a ball and for a disc, how
close to the region a covering union of boxes gets, and how many simple sets that takes.

The approximation is always a *covering* one -- every point of the region lies in it --
so a condition represented this way is never satisfied by fewer states than it should
be, and the fidelity reported is a ratio no greater than one.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

from random_events.product_algebra import Event
from semantic_digital_twin.world_description.geometry import (
    BoundingBox,
    Cylinder,
    Sphere,
)

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    TypstRenderer,
)

# %% the regions a distance threshold describes


@dataclass(eq=False)
class CurvedRegion(ABC):
    """
    A region a threshold on a Euclidean distance describes, which no single box
    captures.

    The shape the region has is the one it inherits, so a region is placed and drawn
    like any other shape, its volume is the one that shape states, and the boxes
    covering it are ordinary bounding boxes.
    """

    @abstractmethod
    def covering_boxes(self, subdivisions: int) -> tuple[BoundingBox, ...]:
        """
        :param subdivisions: Number of parts each subdivided axis is cut into.
        :return: Boxes whose union covers the region.
        """

    def half_width_at(self, radius: float, lower: float, upper: float) -> float:
        """
        :param radius: Radius of the circle being cut into strips.
        :param lower: Lower bound of the strip.
        :param upper: Upper bound of the strip.
        :return: Half the width of the narrowest box covering the circle across that
            strip, which is taken at whichever end of the strip lies closest to the
            centre.
        """
        nearest = 0.0 if lower <= 0.0 <= upper else min(abs(lower), abs(upper))
        return math.sqrt(max(radius * radius - nearest * nearest, 0.0))

    def strips(self, radius: float, subdivisions: int) -> list[tuple[float, float]]:
        """
        :param radius: Radius the strips span.
        :param subdivisions: Number of strips.
        :return: The bounds of each strip, spanning the whole diameter.
        """
        step = 2.0 * radius / subdivisions
        return [
            (-radius + index * step, -radius + (index + 1) * step)
            for index in range(subdivisions)
        ]


@dataclass(eq=False)
class EuclideanDisc(CurvedRegion, Cylinder):
    """
    The region a threshold on a distance taken in a plane describes.

    Such a region drawn in space is the disc extruded along the axis the distance leaves
    out, which is a cylinder. That extrusion is a factor of both the region and its
    covering, so it cancels out of the fidelity and what is measured is the disc's.
    """

    def covering_boxes(self, subdivisions: int) -> tuple[BoundingBox, ...]:
        """
        :param subdivisions: Number of strips the disc is cut into.
        :return: Boxes whose union covers the cylinder, one per strip, each spanning its
            whole height.
        """
        half_height = self.height / 2.0
        return tuple(
            BoundingBox(
                -half_width,
                lower,
                -half_height,
                half_width,
                upper,
                half_height,
                self.origin,
            )
            for lower, upper in self.strips(self.radius, subdivisions)
            for half_width in [self.half_width_at(self.radius, lower, upper)]
        )


@dataclass(eq=False)
class EuclideanBall(CurvedRegion, Sphere):
    """
    The region a threshold on a distance taken in space describes.
    """

    def covering_boxes(self, subdivisions: int) -> tuple[BoundingBox, ...]:
        """
        A ball cut into slabs alone is not captured any better than one box captures a
        disc, because each slab still holds a circular cross-section. Two axes therefore
        have to be subdivided, which is why the cost grows as the square.

        :param subdivisions: Number of parts each of the two subdivided axes is cut
            into.
        :return: Boxes whose union covers the ball, one per slab and strip within it.
        """
        return tuple(
            BoundingBox(
                -half_width,
                strip_lower,
                slab_lower,
                half_width,
                strip_upper,
                slab_upper,
                self.origin,
            )
            for slab_lower, slab_upper in self.strips(self.radius, subdivisions)
            for slab_radius in [self.half_width_at(self.radius, slab_lower, slab_upper)]
            for strip_lower, strip_upper in self.strips(slab_radius, subdivisions)
            for half_width in [
                self.half_width_at(slab_radius, strip_lower, strip_upper)
            ]
        )


# %% how much of a region its representation captures


@dataclass
class CoveringApproximation:
    """
    A covering union of boxes representing one curved region, and how much of it that
    union captures.
    """

    region: CurvedRegion
    """
    The region being represented.
    """

    subdivisions: int
    """
    Number of parts each subdivided axis is cut into.
    """

    @cached_property
    def boxes(self) -> tuple[BoundingBox, ...]:
        """
        :return: The boxes covering the region.
        """
        return self.region.covering_boxes(self.subdivisions)

    @cached_property
    def event(self) -> Event:
        """
        :return: The event representing the region, as a union of boxes made disjoint so
            that its terms are the terms a representation actually costs.
        """
        return Event.from_simple_sets(
            *[box.simple_event for box in self.boxes]
        ).make_disjoint()

    def simple_set_count(self) -> int:
        """
        :return: Number of terms the representation costs, which is what the algebra holds
            after the union is made disjoint and not how many boxes it was given.
        """
        return len(self.event.simple_sets)

    def volume(self) -> float:
        """
        :return: The volume the covering union encloses, which is the volume of the event
            the algebra holds rather than the sum over the boxes it was given.
        """
        return self.event.size

    def intersection_over_union(self) -> float:
        """
        :return: The fraction of the covering union that the region itself fills.
        """
        return self.region.volume / self.volume()


# %% experiment results


@dataclass
class RegionApproximationFidelity(ExperimentResult):
    """
    How much of one curved region a covering union captures, and what it costs.
    """

    region: str
    """
    The region being represented.
    """

    subdivisions: int
    """
    Number of parts each subdivided axis is cut into.
    """

    simple_sets: int
    """
    Number of terms the representation needs.
    """

    intersection_over_union: float
    """
    Fraction of the representation that the region itself fills.
    """


@dataclass
class RegionApproximationExperiment:
    """
    Measures, for each region and each number of subdivisions, how much of the region
    its representation captures and how many terms that takes.
    """

    regions: dict[str, CurvedRegion]
    """
    The regions to represent, keyed by the name they are reported under.
    """

    subdivision_counts: tuple[int, ...]
    """
    Numbers of parts to cut each subdivided axis into.
    """

    def run(self) -> list[RegionApproximationFidelity]:
        """
        :return: One entry per region and number of subdivisions.
        """
        return [
            self.measure(name, region, subdivisions)
            for name, region in self.regions.items()
            for subdivisions in self.subdivision_counts
        ]

    @staticmethod
    def measure(
        name: str, region: CurvedRegion, subdivisions: int
    ) -> RegionApproximationFidelity:
        """
        :param name: Name the region is reported under.
        :param region: The region being represented.
        :param subdivisions: Number of parts each subdivided axis is cut into.
        :return: What that representation captures and costs.
        """
        approximation = CoveringApproximation(region=region, subdivisions=subdivisions)
        return RegionApproximationFidelity(
            region=name,
            subdivisions=subdivisions,
            simple_sets=approximation.simple_set_count(),
            intersection_over_union=approximation.intersection_over_union(),
        )


# %% experiment entry point


def main() -> None:
    """
    Prints how much of each region its representation captures at each number of terms.

    ..note:: The subdivisions stop at sixteen because of what it costs to *count* the
        terms, not because of what the representation costs: making a ball's union
        disjoint is itself the expensive step, and it is what the reported number is
        taken from.
    """
    experiment = RegionApproximationExperiment(
        regions={
            "Ball": EuclideanBall(radius=1.0),
            "Disc": EuclideanDisc(width=2.0, height=1.0),
        },
        subdivision_counts=(1, 2, 4, 8, 16),
    )
    print(
        TypstRenderer(ExperimentsTable(experiment.run())).render_figure(
            "How much of the region a distance threshold describes its axis-aligned "
            "representation captures, against the number of simple sets that "
            "representation needs."
        )
    )


if __name__ == "__main__":
    main()
