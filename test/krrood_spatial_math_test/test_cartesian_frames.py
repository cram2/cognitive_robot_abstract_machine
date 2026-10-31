from __future__ import annotations

import pytest

from krrood.patterns.exceptions import InterfaceInstantiationError
from krrood.patterns.identifier_and_name import HasID, HasName
from krrood_spatial_math.cartesian_frame_3d import (
    CartesianFrame3D,
    CartesianFrame3DReference,
)
from krrood_spatial_math.exceptions import MissingCartesianFrame3DResolver

from .dataset.frames import KnownFrames

# %% the frame interface


def test_a_three_dimensional_cartesian_frame_cannot_be_instantiated():
    with pytest.raises(InterfaceInstantiationError):
        CartesianFrame3D()


@pytest.mark.parametrize("trait", [HasID, HasName])
def test_a_three_dimensional_cartesian_frame_is_identified_and_named(trait):
    assert issubclass(CartesianFrame3D, trait)


def test_a_three_dimensional_cartesian_frame_declares_expressing_values_in_it():
    assert CartesianFrame3D.__abstractmethods__ == {CartesianFrame3D.express.__name__}


# %% referring to a frame in a serialized value


def test_a_reference_resolves_to_the_readers_frame(table_frame):
    reference = CartesianFrame3DReference("reference_frame")
    data = {}
    reference.write(data, table_frame)
    known_frames = KnownFrames()
    known_frames.add(table_frame.id, table_frame)

    assert reference.resolve(data, **known_frames.create_kwargs()) is table_frame


def test_a_reference_cannot_be_resolved_without_a_resolver(table_frame):
    reference = CartesianFrame3DReference("reference_frame")
    data = {}
    reference.write(data, table_frame)

    with pytest.raises(MissingCartesianFrame3DResolver):
        reference.resolve(data)
