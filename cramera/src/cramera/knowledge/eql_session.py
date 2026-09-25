"""
Querying one recorded episode: which domains a scene offers the EQL runner.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Dict, List, Optional

from coraplex.datastructures.enums import Arms
from semantic_digital_twin.spatial_types import Point3

from cramera.knowledge.architecture_entities import Package, PythonClass, SubPackage
from cramera.knowledge.entities import (
    ActionEpisode,
    RecordedArm,
    BenchObject,
    Gripper,
    JointMotion,
    Robot,
)
from cramera.knowledge.detected_events import EVENT_VARIABLE, DetectedEventRecord
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.query_runner import (
    DEFAULT_ROW_LIMIT,
    EqlQueryRunner,
    RenderResult,
)


@dataclass
class EqlSession:
    """
    Runs EQL queries against one recorded episode.

    The knowledge base is held here rather than fetched per query, so a session is
    pinned to the episode it was opened for.
    """

    knowledge_base: EpisodeKnowledgeBase
    """
    The recorded episode every query of this session ranges over.
    """

    @classmethod
    def of_active_scene(cls) -> "EqlSession":
        """
        A session against the scene bundle the server currently serves.
        """
        return cls.of_scene(None)

    @classmethod
    def of_scene(cls, scene: Optional[str]) -> "EqlSession":
        """
        A session against one named scene bundle.

        :param scene: Name of the scene to query, or None for the active one.
        """
        return cls(knowledge_base=EpisodeKnowledgeBase.of_scene(scene))

    def domains(self) -> List[QueryDomain]:
        """
        One ready-made query variable per entity type of the recorded episode.
        """
        return [
            QueryDomain("scene_object", BenchObject, self.knowledge_base.objects),
            QueryDomain("episode", ActionEpisode, self.knowledge_base.episodes),
            QueryDomain("arm", RecordedArm, self.knowledge_base.arms),
            QueryDomain("joint", JointMotion, self.knowledge_base.joints),
            QueryDomain(
                EVENT_VARIABLE, DetectedEventRecord, self.knowledge_base.detected_events
            ),
            QueryDomain("robot", Robot, [self.knowledge_base.robot]),
            QueryDomain("package", Package, self.knowledge_base.packages),
            QueryDomain("subpackage", SubPackage, self.knowledge_base.subpackages),
            QueryDomain("python_class", PythonClass, self.knowledge_base.classes),
        ]

    def runner(self) -> EqlQueryRunner:
        """
        The runner this session's queries are executed by.
        """
        return EqlQueryRunner(
            domains=self.domains(),
            extra_names={
                "Point3": Point3,
                "Gripper": Gripper,
                "Arm": RecordedArm,
                "Arms": Arms,
                "objects": self.knowledge_base.objects,
                "episodes": self.knowledge_base.episodes,
                "arms": self.knowledge_base.arms,
                "grippers": self.knowledge_base.grippers,
                "joints": self.knowledge_base.joints,
                "events": self.knowledge_base.detected_events,
                "robots": [self.knowledge_base.robot],
                "packages": self.knowledge_base.packages,
                "subpackages": self.knowledge_base.subpackages,
                "classes": self.knowledge_base.classes,
            },
        )

    def namespace(self) -> Dict[str, Any]:
        """
        A namespace for evaluating one EQL query (fresh variables each time).
        """
        return self.runner().namespace()

    def run(self, code: str, limit: int = DEFAULT_ROW_LIMIT) -> RenderResult:
        """
        Execute an EQL query string and return its rendered result.

        :param code: The EQL query source.
        :param limit: Maximum number of result rows to return.
        """
        return self.runner().run(code, limit=limit)
