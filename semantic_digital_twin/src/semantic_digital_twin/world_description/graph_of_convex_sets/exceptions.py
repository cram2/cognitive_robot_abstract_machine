from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.exceptions import UsageError


@dataclass
class UnboundedSearchSpaceError(UsageError):
    """
    Raised when a
    :class:`~semantic_digital_twin.world_description.graph_of_convex_sets.polygons.GraphOfConvexPolygons`
    is built with a search space that is not a single, finite bounding box.

    IRIS grows regions within a bounded convex domain; unlike
    :class:`~semantic_digital_twin.world_description.graph_of_convex_sets.boxes.GraphOfBoundingBoxes`,
    which can decompose an unbounded or multi-box search space via the product algebra,
    Drake's ``Iris`` function requires exactly one finite ``HPolyhedron`` domain.
    """

    def error_message(self) -> str:
        return (
            "GraphOfConvexPolygons requires a search space consisting of exactly one "
            "finite bounding box."
        )

    def suggest_correction(self) -> str:
        return "pass an explicit, finite search_space with a single bounding box."
