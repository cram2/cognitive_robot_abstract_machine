from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import TYPE_CHECKING

from cramera.knowledge.presets import Preset
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.queryable_knowledge import QueryableKnowledge, QueryScope
from cramera.live.query import LiveQuerySource
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from semantic_digital_twin.world import World


# %% world query vocabulary
class WorldQueryName(StrEnum):
    BODY = "body"
    ANNOTATION = "annotation"
    HANDLE = "handle"
    SURFACE = "surface"
    ROBOT = "robot"
    ARM = "arm"


class WorldQueryLabel(StrEnum):
    TITLE = "Live world"
    BODIES = "show all scene bodies"
    ANNOTATIONS = "show all semantic annotations"
    HANDLES = "show all handles"
    SURFACES = "show all supporting surfaces"
    ROBOTS = "show all robots"
    ARMS = "show all robot arms"


# %% native world queries
@dataclass
class WorldQuerySource(LiveQuerySource):
    world: World
    """
    World supplying the current native entities for each request.
    """

    def title(self) -> str:
        return WorldQueryLabel.TITLE

    def read_scope(self) -> AbstractContextManager[object]:
        return self.world.state.world_lock

    def knowledge(self) -> list[QueryableKnowledge]:
        return [
            QueryableKnowledge(
                scope=QueryScope.CURRENT_STATE,
                domains=[
                    QueryDomain(WorldQueryName.BODY, Body, list(self.world.bodies)),
                    QueryDomain(
                        WorldQueryName.ANNOTATION,
                        SemanticAnnotation,
                        list(self.world.semantic_annotations),
                    ),
                    QueryDomain(
                        WorldQueryName.HANDLE,
                        Handle,
                        list(self.world.get_semantic_annotations_by_type(Handle)),
                    ),
                    QueryDomain(
                        WorldQueryName.SURFACE,
                        HasSupportingSurface,
                        list(
                            self.world.get_semantic_annotations_by_type(
                                HasSupportingSurface
                            )
                        ),
                    ),
                    QueryDomain(
                        WorldQueryName.ROBOT,
                        AbstractRobot,
                        list(
                            self.world.get_semantic_annotations_by_type(AbstractRobot)
                        ),
                    ),
                    QueryDomain(
                        WorldQueryName.ARM,
                        Arm,
                        list(self.world.get_semantic_annotations_by_type(Arm)),
                    ),
                ],
            )
        ]

    def presets(self) -> list[Preset]:
        return [
            Preset(label, f"an(entity({name}))", requires_live=True)
            for label, name in (
                (WorldQueryLabel.BODIES, WorldQueryName.BODY),
                (WorldQueryLabel.ANNOTATIONS, WorldQueryName.ANNOTATION),
                (WorldQueryLabel.HANDLES, WorldQueryName.HANDLE),
                (WorldQueryLabel.SURFACES, WorldQueryName.SURFACE),
                (WorldQueryLabel.ROBOTS, WorldQueryName.ROBOT),
                (WorldQueryLabel.ARMS, WorldQueryName.ARM),
            )
        ]
