"""
Answer :mod:`entity query language <krrood.entity_query_language>` queries about the
Montessori scene by looking at it.

:class:`PerceivedObjects` is an entity query language domain, so a query written over it
is what makes perception run::

    from krrood.entity_query_language.factories import a

    perceived = PerceivedObjects(source=node)
    triangle = (
        a(MontessoriShapeDetection)(category=MontessoriShapeCategory.TRIANGULAR_PRISM)
        .from_(perceived)
        .first()
    )
    reach_for = triangle.pose

Nothing is looked at until the query is materialised, and a domain is iterated once per
materialisation, so one query sees one consistent scene while a later query sees a fresh
one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Iterator

from experiments.montessori.perception.detections import (
    MontessoriDetection,
    MontessoriScene,
)

# %% where a scene comes from


class MontessoriSceneSource(ABC):
    """
    Something that can say what the Montessori scene currently looks like.
    """

    @abstractmethod
    def scene(self) -> MontessoriScene:
        """
        Look at the scene.

        :return: Everything currently recognised.
        """


@dataclass
class FixedScene(MontessoriSceneSource):
    """
    A scene that was already looked at, for querying one captured moment repeatedly.
    """

    captured: MontessoriScene
    """
    The scene to answer every query from.
    """

    def scene(self) -> MontessoriScene:
        return self.captured


# %% the queryable domain


@dataclass
class PerceivedObjects:
    """
    The entity query language domain of everything perception can currently see.

    Iterating this looks at the scene, so a query written over it invokes perception to
    answer itself. Both the loose pieces and the board's holes are yielded from the one
    domain; the query's own type picks out the kind it asked for.
    """

    source: MontessoriSceneSource
    """
    Where a fresh look at the scene comes from.
    """

    def __iter__(self) -> Iterator[MontessoriDetection]:
        """
        Look at the scene and yield everything in it.

        This is a generator function, so building the query costs nothing: perception
        only runs once the query is actually materialised and starts pulling values.
        """
        yield from self.source.scene().detections
