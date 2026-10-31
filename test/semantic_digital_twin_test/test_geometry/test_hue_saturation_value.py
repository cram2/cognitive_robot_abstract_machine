from __future__ import annotations

# %% Imports

import pytest

from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.world_description.geometry import Color

# %% Coordinate preservation


@pytest.mark.parametrize("saturation, value", [(0.0, 0.5), (0.75, 0.0)])
def test_serialization_preserves_achromatic_coordinates(
    saturation: float, value: float
) -> None:
    """
    Gray and black bounds retain their specified hue and saturation.
    """
    coordinates = Color(hue=0.75, saturation=saturation, value=value)

    restored = from_json(to_json(coordinates))

    assert restored.hue == coordinates.hue
    assert restored.saturation == coordinates.saturation
    assert restored.value == coordinates.value
