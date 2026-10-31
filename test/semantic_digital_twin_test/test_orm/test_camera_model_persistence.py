from __future__ import annotations

# %% Imports
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.spatial_types import Point2

# %% Principal point persistence


def test_pinhole_retains_principal_point_after_database_roundtrip(
    in_memory_session_maker: sessionmaker[Session],
) -> None:
    """
    Persist fractional image coordinates without attaching a spatial frame.
    """
    model = PinholeCameraModel(
        image_resolution=CameraResolution(width=640, height=480),
        focal_length_x=500.0,
        focal_length_y=510.0,
        principal_point=Point2(x=319.5, y=239.25),
    )
    stored_model = to_dao(model)
    with in_memory_session_maker() as session:
        session.add(stored_model)
        session.commit()
        session.expunge_all()
        restored = session.scalar(select(type(stored_model))).from_dao()

    assert float(restored.principal_point.x) == float(model.principal_point.x)
    assert float(restored.principal_point.y) == float(model.principal_point.y)
    assert restored.principal_point.reference_frame is None
