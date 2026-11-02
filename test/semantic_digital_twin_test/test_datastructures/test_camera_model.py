from __future__ import annotations

# %% Imports
from unittest.mock import Mock

from krrood.adapters.json_serializer import from_json, to_json

import numpy as np
import pytest

from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.datastructures import camera_model as camera_model_module
from semantic_digital_twin.datastructures.camera_model import (
    CameraDistortion,
    CameraDistortionModel,
    CameraRange,
    EquidistantCameraDistortion,
    NoCameraDistortion,
    PlumbBobCameraDistortion,
    FieldOfViewCameraModel,
    PinholeCameraModel,
    RationalPolynomialCameraDistortion,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.exceptions import (
    InvalidCameraDistortionError,
    InvalidCameraFieldOfViewError,
    InvalidCameraRangeError,
    InvalidPinholeCameraModelError,
)


def test_pinhole_camera_model_from_field_of_view_preserves_projection() -> None:
    resolution = CameraResolution(width=640, height=480)
    field_of_view = FieldOfView(
        horizontal_angle=np.radians(90.0),
        vertical_angle=np.radians(60.0),
    )

    model = PinholeCameraModel.from_field_of_view(
        resolution=resolution,
        field_of_view=field_of_view,
    )

    assert model.resolution == resolution
    assert np.isclose(
        model.field_of_view.horizontal_angle,
        field_of_view.horizontal_angle,
    )
    assert np.isclose(
        model.field_of_view.vertical_angle,
        field_of_view.vertical_angle,
    )
    assert isinstance(model.distortion, NoCameraDistortion)


def test_field_of_view_camera_model_rejects_zero_angle() -> None:
    with pytest.raises(InvalidCameraFieldOfViewError):
        FieldOfViewCameraModel(
            angular_extent=FieldOfView(
                horizontal_angle=0.0, vertical_angle=np.radians(60.0)
            )
        )


def test_field_of_view_camera_model_has_default_resolution() -> None:
    model = FieldOfViewCameraModel(angular_extent=FieldOfView())

    assert model.resolution == CameraResolution()


def test_field_of_view_camera_model_preserves_explicit_resolution() -> None:
    resolution = CameraResolution(width=320, height=240)

    model = FieldOfViewCameraModel(
        angular_extent=FieldOfView(),
        image_resolution=resolution,
    )

    assert model.resolution is resolution


def test_pinhole_camera_model_rejects_nonpositive_focal_length() -> None:
    with pytest.raises(InvalidPinholeCameraModelError):
        PinholeCameraModel(
            image_resolution=CameraResolution(width=640, height=480),
            focal_length_x=0.0,
            focal_length_y=400.0,
            principal_point=Point2(x=320.0, y=240.0),
        )


def test_pinhole_camera_model_exposes_intrinsic_matrix() -> None:
    model = PinholeCameraModel(
        image_resolution=CameraResolution(width=640, height=480),
        focal_length_x=500.0,
        focal_length_y=510.0,
        principal_point=Point2(x=321.0, y=239.0),
    )

    assert np.array_equal(
        model.intrinsic_matrix,
        np.array(
            [
                [model.focal_length_x, 0.0, float(model.principal_point.x)],
                [0.0, model.focal_length_y, float(model.principal_point.y)],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def test_pinhole_camera_model_scales_projection_and_preserves_distortion() -> None:
    distortion = PlumbBobCameraDistortion(radial_coefficient_1=0.1)
    model = PinholeCameraModel(
        image_resolution=CameraResolution(width=640, height=480),
        focal_length_x=500.0,
        focal_length_y=510.0,
        principal_point=Point2(x=321.0, y=239.0),
        distortion=distortion,
    )

    scaled_model = model.scaled(scale_x=0.5, scale_y=0.25)

    assert scaled_model.resolution == CameraResolution(width=320, height=120)
    assert scaled_model.focal_length_x == model.focal_length_x * 0.5
    assert scaled_model.focal_length_y == model.focal_length_y * 0.25
    assert float(scaled_model.principal_point.x) == float(model.principal_point.x) * 0.5
    assert (
        float(scaled_model.principal_point.y) == float(model.principal_point.y) * 0.25
    )
    assert scaled_model.distortion is distortion


def test_camera_range_rejects_empty_interval() -> None:
    with pytest.raises(InvalidCameraRangeError):
        CameraRange(minimum_distance=1.0, maximum_distance=1.0)


@pytest.mark.parametrize(
    "model, coefficients",
    [
        (CameraDistortionModel.NONE, (0.0,)),
        (CameraDistortionModel.PLUMB_BOB, (0.0,) * 4),
        (CameraDistortionModel.RATIONAL_POLYNOMIAL, (0.0,) * 7),
        (CameraDistortionModel.EQUIDISTANT, (0.0,) * 3),
    ],
)
def test_camera_distortion_rejects_wrong_coefficient_count(
    model: CameraDistortionModel, coefficients: tuple[float, ...]
) -> None:
    with pytest.raises(InvalidCameraDistortionError):
        CameraDistortion.from_ordered_coefficients(
            model=model,
            coefficients=coefficients,
        )


def test_plumb_bob_distortion_names_ordered_coefficients() -> None:
    coefficients = (0.1, -0.2, 0.01, 0.02, 0.03)

    distortion = CameraDistortion.from_ordered_coefficients(
        model=CameraDistortionModel.PLUMB_BOB,
        coefficients=coefficients,
    )

    assert isinstance(distortion, PlumbBobCameraDistortion)
    assert distortion.radial_coefficient_1 == coefficients[0]
    assert distortion.radial_coefficient_2 == coefficients[1]
    assert distortion.tangential_coefficient_1 == coefficients[2]
    assert distortion.tangential_coefficient_2 == coefficients[3]
    assert distortion.radial_coefficient_3 == coefficients[4]
    assert distortion.to_ordered_coefficients() == coefficients


# %% Distortion construction dispatch
@pytest.mark.parametrize(
    "implementations", [[], [NoCameraDistortion, NoCameraDistortion]]
)
def test_distortion_model_requires_unique_implementation(
    monkeypatch: pytest.MonkeyPatch,
    implementations: list[type[CameraDistortion]],
) -> None:
    """
    Missing or ambiguous model implementations raise a distortion error.
    """
    monkeypatch.setattr(
        camera_model_module,
        "concrete_subclasses",
        Mock(return_value=implementations),
    )

    with pytest.raises(InvalidCameraDistortionError):
        CameraDistortion.get_type_for_model(CameraDistortionModel.NONE)


@pytest.mark.parametrize(
    ("distortion_type", "coefficients"),
    [
        (NoCameraDistortion, ()),
        (PlumbBobCameraDistortion, (0.1, -0.2, 0.01, 0.02, 0.03)),
        (
            RationalPolynomialCameraDistortion,
            (0.1, -0.2, 0.01, 0.02, 0.03, -0.04, 0.05, -0.06),
        ),
        (EquidistantCameraDistortion, (0.1, -0.2, 0.3, -0.4)),
    ],
)
def test_distortion_factory_roundtrips_subclass_coefficients(
    distortion_type: type[CameraDistortion], coefficients: tuple[float, ...]
) -> None:
    """
    Each concrete distortion implementation preserves its ROS coefficient sequence.
    """
    distortion = CameraDistortion.from_ordered_coefficients(
        model=distortion_type.model, coefficients=coefficients
    )

    assert type(distortion) is distortion_type
    assert distortion.to_ordered_coefficients() == coefficients


def test_distortion_factory_delegates_to_subclass_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The matching distortion implementation owns ordered coefficient construction.
    """
    coefficients = (0.1, -0.2, 0.01, 0.02, 0.03)
    expected = PlumbBobCameraDistortion(radial_coefficient_1=coefficients[0])
    parser = Mock(return_value=expected)
    monkeypatch.setattr(
        PlumbBobCameraDistortion, "_from_ordered_coefficients", parser, raising=False
    )

    distortion = CameraDistortion.from_ordered_coefficients(
        model=PlumbBobCameraDistortion().model, coefficients=coefficients
    )

    parser.assert_called_once_with(coefficients)
    assert distortion is expected


def test_equidistant_distortion_names_ordered_coefficients() -> None:
    coefficients = (0.1, -0.2, 0.3, -0.4)

    distortion = CameraDistortion.from_ordered_coefficients(
        model=CameraDistortionModel.EQUIDISTANT,
        coefficients=coefficients,
    )

    assert isinstance(distortion, EquidistantCameraDistortion)
    assert distortion.radial_coefficient_1 == coefficients[0]
    assert distortion.radial_coefficient_2 == coefficients[1]
    assert distortion.radial_coefficient_3 == coefficients[2]
    assert distortion.radial_coefficient_4 == coefficients[3]
    assert distortion.to_ordered_coefficients() == coefficients


@pytest.mark.parametrize("coefficient", [np.nan, np.inf, -np.inf])
def test_camera_distortion_rejects_nonfinite_coefficient(coefficient: float) -> None:
    with pytest.raises(InvalidCameraDistortionError):
        PlumbBobCameraDistortion(radial_coefficient_1=coefficient)


# %% Declared angular extent


def test_field_of_view_model_exposes_declared_value() -> None:
    """
    The common field-of-view property exposes the explicitly configured value.
    """
    angular_extent = FieldOfView()
    model = FieldOfViewCameraModel(angular_extent=angular_extent)
    assert model.field_of_view is angular_extent


# %% Finite calibration and distance bounds
@pytest.mark.parametrize("focal_length", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("horizontal", [True, False])
def test_pinhole_rejects_nonfinite_focal_lengths(
    focal_length: float, horizontal: bool
) -> None:
    """
    Neither projection axis accepts a nonfinite focal length.
    """
    with pytest.raises(InvalidPinholeCameraModelError):
        PinholeCameraModel(
            image_resolution=CameraResolution(width=8, height=6),
            focal_length_x=focal_length if horizontal else 10.0,
            focal_length_y=12.0 if horizontal else focal_length,
            principal_point=Point2(x=2.0, y=1.0),
        )


@pytest.mark.parametrize("minimum_distance", [np.nan, np.inf, -np.inf])
def test_camera_range_rejects_nonfinite_minimum(minimum_distance: float) -> None:
    """
    The nearest usable distance must be finite.
    """
    with pytest.raises(InvalidCameraRangeError):
        CameraRange(minimum_distance=minimum_distance)


@pytest.mark.parametrize("maximum_distance", [np.nan, -np.inf])
def test_camera_range_rejects_invalid_maximum(maximum_distance: float) -> None:
    """
    The far bound cannot be undefined or negative infinity.
    """
    with pytest.raises(InvalidCameraRangeError):
        CameraRange(maximum_distance=maximum_distance)


def test_camera_range_allows_unbounded_maximum() -> None:
    """
    The default range intentionally has no finite far bound.
    """
    assert CameraRange().maximum_distance == np.inf


# %% Fractional image principal point


def test_pinhole_retains_fractional_principal_point() -> None:
    """
    Keep a frame-free image point and expose numeric calibration entries.
    """
    principal_point = Point2(x=319.5, y=239.25)
    model = PinholeCameraModel(
        image_resolution=CameraResolution(width=640, height=480),
        focal_length_x=500.0,
        focal_length_y=510.0,
        principal_point=principal_point,
    )

    assert model.principal_point is principal_point
    assert model.principal_point.reference_frame is None
    assert model.intrinsic_matrix.dtype == np.dtype(float)
    np.testing.assert_array_equal(
        model.intrinsic_matrix[:2, 2],
        [float(principal_point.x), float(principal_point.y)],
    )


# %% Calibration serialization


def test_pinhole_deserializes_recorded_principal_point() -> None:
    """
    Preserve a structured image point through JSON serialization.
    """
    model = PinholeCameraModel(
        image_resolution=CameraResolution(width=640, height=480),
        focal_length_x=500.0,
        focal_length_y=510.0,
        principal_point=Point2(x=319.5, y=239.25),
    )
    payload = to_json(model)
    restored = from_json(payload)

    assert restored == model
    assert isinstance(restored.principal_point, Point2)
    assert restored.principal_point.reference_frame is None
    assert to_json(restored) == to_json(model)
