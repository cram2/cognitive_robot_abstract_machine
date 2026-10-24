from __future__ import annotations

# %% Imports

import pytest

from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.world_description.geometry import HueSaturationValue

# %% Coordinate preservation


@pytest.mark.parametrize("saturation, value", [(0.0, 0.5), (0.75, 0.0)])
def test_serialization_preserves_achromatic_coordinates(
    saturation: float, value: float
) -> None:
    """
    Gray and black bounds retain their specified hue and saturation.
    """
    coordinates = HueSaturationValue(hue=0.75, saturation=saturation, value=value)

    assert from_json(to_json(coordinates)) == coordinates
