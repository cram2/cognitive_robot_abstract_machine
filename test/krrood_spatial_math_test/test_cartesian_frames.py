"""
Cartesian frames, and how serialized values refer to them.
"""

from __future__ import annotations

import pytest

from krrood.patterns.exceptions import InterfaceInstantiationError
from krrood_spatial_math.cartesian_frame_2_dimensional import CartesianFrame2Dimensional
from krrood_spatial_math.cartesian_frame_3_dimensional import (
    CartesianFrame3Dimensional,
    CartesianFrame3DimensionalReference,
)
from krrood_spatial_math.coordinate_system import CartesianFrame, GenericSpace
from krrood_spatial_math.embedded_plane import EmbeddedPlane
from krrood_spatial_math.exceptions import MissingCartesianFrame3DimensionalResolver
from krrood_spatial_math.space import EuclideanSpace

from .dataset.frames import KnownFrames, ThingFrame

# %% the frame interface


def test_a_three_dimensional_cartesian_frame_cannot_be_instantiated():
    with pytest.raises(InterfaceInstantiationError):
        CartesianFrame3Dimensional()


def test_a_three_dimensional_cartesian_frame_is_a_cartesian_frame():
    assert issubclass(CartesianFrame3Dimensional, CartesianFrame)


def test_a_three_dimensional_cartesian_frame_charts_a_euclidean_space():
    """
    The kind of space a body's frame charts is read back from the frame class.
    """
    assert (
        CartesianFrame3Dimensional.get_type_of_generic_parameter(parameter=GenericSpace)
        is EuclideanSpace
    )


def test_a_three_dimensional_cartesian_frame_declares_what_it_computes_within_its_space():
    """
    Checking the spaces is the frame interface's own; changing a value, comparing x-y
    planes within one space and stating its z-axis are the frame's.
    """
    assert CartesianFrame3Dimensional.__abstractmethods__ == {
        CartesianFrame3Dimensional.transform_into_own_frame.__name__,
        CartesianFrame3Dimensional.compare_embedded_plane_in_own_space.__name__,
        "z_axis",
    }


def test_a_frame_can_extend_an_object_named_before_it_is_identified():
    assert issubclass(ThingFrame, CartesianFrame3Dimensional)


# %% the frame on a plane


def test_a_two_dimensional_cartesian_frame_is_a_cartesian_frame():
    assert issubclass(CartesianFrame2Dimensional, CartesianFrame)


def test_a_two_dimensional_cartesian_frame_charts_a_plane_in_three_dimensional_space():
    assert (
        CartesianFrame2Dimensional.get_type_of_generic_parameter(parameter=GenericSpace)
        is EmbeddedPlane
    )


def test_a_two_dimensional_cartesian_frame_is_no_three_dimensional_one():
    assert not issubclass(CartesianFrame2Dimensional, CartesianFrame3Dimensional)


# %% referring to a frame in a serialized value


def test_a_reference_resolves_to_the_readers_frame(table_frame):
    reference = CartesianFrame3DimensionalReference(subject="reference_frame")
    data = {}
    reference.write(data=data, identified_object=table_frame)
    known_frames = KnownFrames()
    known_frames.add(key=table_frame.id, tracked_object=table_frame)

    assert reference.resolve(data=data, **known_frames.create_kwargs()) is table_frame


def test_a_reference_cannot_be_resolved_without_a_resolver(table_frame):
    reference = CartesianFrame3DimensionalReference(subject="reference_frame")
    data = {}
    reference.write(data=data, identified_object=table_frame)

    with pytest.raises(MissingCartesianFrame3DimensionalResolver):
        reference.resolve(data=data)
