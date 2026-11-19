from __future__ import annotations

import pytest

from krrood.patterns.identifier_and_name import NameT
from krrood.patterns.exceptions import InterfaceInstantiationError
from krrood_spatial_math.cartesian_frame_3_dimensional import CartesianFrame3Dimensional
from krrood_spatial_math.coordinate_system import (
    AffineCoordinateChange,
    AffineFrame,
    GenericAffineFrame,
    CartesianFrame,
    GenericCartesianFrame,
    CoordinateChange,
    CoordinateRepresentation,
    CoordinateSystem,
    RigidCoordinateChange,
    GenericSpace,
)
from krrood_spatial_math.space import (
    EuclideanSpace,
    EuclideanSpace2Dimensional,
    EuclideanSpace3Dimensional,
    Space,
)

from .dataset.spaces import (
    CartesianFrame3DimensionalChange,
    ScalablePlane,
    ScalablePlaneFrame,
    ScalingChange,
)

# %% spaces


@pytest.mark.parametrize(
    "space_kind",
    [Space, EuclideanSpace, EuclideanSpace2Dimensional, EuclideanSpace3Dimensional],
)
def test_a_space_cannot_be_instantiated(space_kind):
    with pytest.raises(InterfaceInstantiationError):
        space_kind()


def test_euclidean_space_is_a_space():
    assert issubclass(EuclideanSpace, Space)


@pytest.mark.parametrize(
    "space_kind", [EuclideanSpace2Dimensional, EuclideanSpace3Dimensional]
)
def test_a_space_of_fixed_dimension_is_a_euclidean_space(space_kind):
    assert issubclass(space_kind, EuclideanSpace)


def test_a_two_dimensional_euclidean_space_has_two_dimensions():
    assert EuclideanSpace2Dimensional.dimension == 2


def test_a_three_dimensional_euclidean_space_has_three_dimensions():
    assert EuclideanSpace3Dimensional.dimension == 3


# %% coordinate systems on spaces


@pytest.mark.parametrize(
    "coordinate_system_kind", [CoordinateSystem, AffineFrame, CartesianFrame]
)
def test_a_coordinate_system_interface_cannot_be_instantiated(coordinate_system_kind):
    with pytest.raises(InterfaceInstantiationError):
        coordinate_system_kind()


def test_an_affine_frame_is_a_coordinate_system():
    assert issubclass(AffineFrame, CoordinateSystem)


def test_a_cartesian_frame_is_an_affine_frame():
    assert issubclass(CartesianFrame, AffineFrame)


def test_an_affine_frame_need_not_be_cartesian():
    assert issubclass(ScalablePlaneFrame, AffineFrame)
    assert not issubclass(ScalablePlaneFrame, CartesianFrame)


def test_an_affine_frame_reports_the_space_it_charts():
    assert (
        ScalablePlaneFrame.get_type_of_generic_parameter(parameter=GenericSpace)
        is ScalablePlane
    )


def test_an_affine_frame_reports_the_kind_of_name_it_is_called_by():
    assert ScalablePlaneFrame.get_type_of_generic_parameter(parameter=NameT) is str


# %% values given in a coordinate system


def test_a_coordinate_representation_cannot_be_instantiated():
    with pytest.raises(InterfaceInstantiationError):
        CoordinateRepresentation()


# %% changing coordinates within one space


@pytest.mark.parametrize(
    "change_kind",
    [CoordinateChange, AffineCoordinateChange, RigidCoordinateChange],
)
def test_a_coordinate_change_interface_cannot_be_instantiated(change_kind):
    with pytest.raises(InterfaceInstantiationError):
        change_kind()


def test_a_coordinate_change_declares_inverting_and_composing():
    assert CoordinateChange.__abstractmethods__ == {
        CoordinateChange.inverse.__name__,
        CoordinateChange.__matmul__.__name__,
    }


def test_a_rigid_change_is_an_affine_change():
    assert issubclass(RigidCoordinateChange, AffineCoordinateChange)


def test_an_affine_change_is_a_coordinate_change():
    assert issubclass(AffineCoordinateChange, CoordinateChange)


def test_an_affine_change_need_not_be_rigid():
    assert issubclass(ScalingChange, AffineCoordinateChange)
    assert not issubclass(ScalingChange, RigidCoordinateChange)


def test_an_affine_change_changes_between_affine_frames():
    assert GenericAffineFrame.__bound__ is AffineFrame


def test_a_rigid_change_changes_between_cartesian_frames():
    assert GenericCartesianFrame.__bound__ is CartesianFrame


def test_a_rigid_change_reports_the_kind_of_frame_it_changes():
    assert (
        CartesianFrame3DimensionalChange.get_type_of_generic_parameter(
            parameter=GenericCartesianFrame
        )
        is CartesianFrame3Dimensional
    )


def test_an_affine_change_reports_the_kind_of_frame_it_changes():
    assert (
        ScalingChange.get_type_of_generic_parameter(parameter=GenericAffineFrame)
        is ScalablePlaneFrame
    )
