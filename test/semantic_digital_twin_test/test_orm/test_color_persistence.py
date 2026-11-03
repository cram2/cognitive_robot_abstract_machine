from __future__ import annotations

# %% Imports

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.world_description.geometry import Color

# %% Color persistence


@pytest.mark.parametrize("constructed_from_hue_saturation_value", [False, True])
def test_color_retains_hue_saturation_value_after_database_roundtrip(
    in_memory_session_maker: sessionmaker[Session],
    constructed_from_hue_saturation_value: bool,
) -> None:
    """
    Database persistence retains stored hue, saturation, value, and opacity.
    """
    color = (
        Color(hue=0.75, saturation=0.0, value=0.5, opacity=0.25)
        if constructed_from_hue_saturation_value
        else Color.from_rgba(red=0.25, green=0.5, blue=0.75, opacity=0.25)
    )
    stored_color = to_dao(color)
    with in_memory_session_maker() as session:
        session.add(stored_color)
        session.commit()
        session.expunge_all()
        restored = session.scalar(select(type(stored_color))).from_dao()

    assert restored.hue == color.hue
    assert restored.saturation == color.saturation
    assert restored.value == color.value
    assert restored.opacity == color.opacity
