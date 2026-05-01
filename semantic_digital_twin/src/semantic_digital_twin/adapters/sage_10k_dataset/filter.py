from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from semantic_digital_twin.adapters.sage_10k_dataset.semantic_annotations import (
    Sage10kTypeNameCleaner,
)
from semantic_digital_twin.orm.ormatic_interface import (
    Sage10kSceneDAO,
    Sage10kSceneDAO_rooms_association,
    Sage10kRoomDAO,
    Sage10kObjectDAO,
)


@dataclass
class SceneObjectTypeHistogram:
    """
    A histogram of cleaned object types for each scene in the Sage10k dataset.
    """

    histogram: dict[Sage10kSceneDAO, dict[str, int]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    cleaner: Sage10kTypeNameCleaner = field(default_factory=Sage10kTypeNameCleaner)

    @classmethod
    def from_session(cls, session: Session) -> SceneObjectTypeHistogram:
        """
        Query the database and build a histogram of cleaned object types per scene.

        :param session: The SQLAlchemy session to use for querying.
        :return: A SceneObjectTypeHistogram instance.
        """
        instance = cls()

        query = (
            select(
                Sage10kSceneDAO.database_id,
                Sage10kObjectDAO.type,
                func.count(Sage10kObjectDAO.database_id).label("count"),
            )
            .join(
                Sage10kSceneDAO_rooms_association,
                Sage10kSceneDAO_rooms_association.source_sage10kscenedao_id
                == Sage10kSceneDAO.database_id,
            )
            .join(
                Sage10kRoomDAO,
                Sage10kRoomDAO.database_id
                == Sage10kSceneDAO_rooms_association.target_sage10kroomdao_id,
            )
            .join(
                Sage10kObjectDAO,
                Sage10kObjectDAO.room_id == Sage10kRoomDAO.id,
            )
            .group_by(Sage10kSceneDAO.database_id, Sage10kObjectDAO.type)
        )

        rows = session.execute(query).all()

        scene_ids = {row.database_id for row in rows}
        scenes = {
            scene.database_id: scene
            for scene in session.execute(
                select(Sage10kSceneDAO).where(
                    Sage10kSceneDAO.database_id.in_(scene_ids)
                )
            ).scalars()
        }

        for scene_id, object_type, count in rows:
            cleaned_type: Optional[str] = instance.cleaner.clean(object_type)
            if cleaned_type is None:
                continue
            scene = scenes[scene_id]
            instance.histogram[scene][cleaned_type] = (
                instance.histogram[scene].get(cleaned_type, 0) + count
            )

        return instance
