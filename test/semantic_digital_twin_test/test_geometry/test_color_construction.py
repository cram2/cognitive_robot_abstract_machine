from __future__ import annotations

# %% Imports
import pytest

from semantic_digital_twin.world_description.geometry import Color


# %% Native color construction
@pytest.mark.parametrize("saturation, value", [(0.0, 0.5), (0.75, 0.0), (0.75, 0.5)])
def test_constructor_preserves_native_coordinates(
    saturation: float, value: float
) -> None:
    """
    Constructing a color retains native coordinates even for gray and black.
    """
    color = Color(hue=0.75, saturation=saturation, value=value, opacity=0.25)

    assert (color.hue, color.saturation, color.value, color.opacity) == (
        0.75,
        saturation,
        value,
        0.25,
    )


def test_rgba_factory_accepts_explicit_channels() -> None:
    """
    Named RGB channels are converted into native coordinates with independent opacity.
    """
    color = Color.from_rgba(red=0.25, green=0.5, blue=0.75, opacity=0.5)

    assert (color.hue, color.saturation, color.value) == pytest.approx(
        (7 / 12, 2 / 3, 0.75)
    )
    assert color.to_rgba() == (0.25, 0.5, 0.75, 0.5)


def test_default_color_is_opaque_white() -> None:
    """
    The generated dataclass constructor defaults to opaque white.
    """
    assert Color().to_rgba() == (1.0, 1.0, 1.0, 1.0)
