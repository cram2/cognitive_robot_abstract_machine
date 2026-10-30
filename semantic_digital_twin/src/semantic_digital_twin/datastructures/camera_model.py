from __future__ import annotations

# %% Imports
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from typing_extensions import Self, Sequence

from krrood.ormatic.utils import classproperty
from krrood.patterns.specificity_ranking import concrete_subclasses
from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.exceptions import (
    InvalidCameraDistortionError,
    InvalidCameraFieldOfViewError,
    InvalidCameraRangeError,
    InvalidPinholeCameraModelError,
)

# %% Projection models


@dataclass
class CameraModel(ABC):
    """
    Describe how a camera maps viewing rays to an image.
    """

    @property
    @abstractmethod
    def field_of_view(self) -> FieldOfView:
        """
        Return the angular extent represented by the model.
        """

    @property
    @abstractmethod
    def resolution(self) -> CameraResolution:
        """
        Return the image resolution represented by the model.
        """

    @staticmethod
    def _validate_field_of_view(field_of_view: FieldOfView) -> None:
        """
        Validate that both viewing angles lie between zero and pi.

        :param field_of_view: Angular extent to validate.
        :raises InvalidCameraFieldOfViewError: If either angle is outside the valid
            range.
        """
        if not 0.0 < field_of_view.horizontal_angle < math.pi:
            raise InvalidCameraFieldOfViewError(
                horizontal_angle=field_of_view.horizontal_angle,
                vertical_angle=field_of_view.vertical_angle,
            )
        if not 0.0 < field_of_view.vertical_angle < math.pi:
            raise InvalidCameraFieldOfViewError(
                horizontal_angle=field_of_view.horizontal_angle,
                vertical_angle=field_of_view.vertical_angle,
            )


@dataclass
class FieldOfViewCameraModel(CameraModel):
    """
    Describe an ideal camera from its angular extent and image resolution.
    """

    angular_extent: FieldOfView = field(kw_only=True)
    """
    Angular extent of the camera image.
    """

    image_resolution: CameraResolution = field(
        default_factory=CameraResolution, kw_only=True
    )
    """
    Pixel dimensions of images produced by the camera.
    """

    def __post_init__(self) -> None:
        """
        Validate that both viewing angles describe a physical camera.
        """
        self._validate_field_of_view(self.angular_extent)

    @property
    def field_of_view(self) -> FieldOfView:
        return self.angular_extent

    @property
    def resolution(self) -> CameraResolution:
        return self.image_resolution


@dataclass(eq=False)
class PinholeCameraModel(CameraModel):
    """
    Describe a calibrated pinhole projection for one image stream.
    """

    image_resolution: CameraResolution = field(kw_only=True)
    """
    Pixel dimensions of images produced with this calibration.
    """

    focal_length_x: float = field(kw_only=True)
    """
    Horizontal focal length in pixels.
    """

    focal_length_y: float = field(kw_only=True)
    """
    Vertical focal length in pixels.
    """

    principal_point: Point2 = field(kw_only=True)
    """
    Image position in pixels from the top-left corner, including fractional pixels.
    """

    distortion: CameraDistortion = field(default_factory=lambda: NoCameraDistortion())
    """
    Lens-distortion parameters associated with the calibration.
    """

    def __post_init__(self) -> None:
        """
        Validate the focal lengths and principal point.
        """
        if any(
            not math.isfinite(focal_length) or focal_length <= 0.0
            for focal_length in (self.focal_length_x, self.focal_length_y)
        ):
            raise InvalidPinholeCameraModelError(
                reason="focal lengths must be finite and positive"
            )
        if not 0.0 <= float(self.principal_point.x) <= self.image_resolution.width:
            raise InvalidPinholeCameraModelError(
                reason="the horizontal principal point must lie within the image"
            )
        if not 0.0 <= float(self.principal_point.y) <= self.image_resolution.height:
            raise InvalidPinholeCameraModelError(
                reason="the vertical principal point must lie within the image"
            )

    def __eq__(self, other: object) -> bool:
        """
        Compare calibration values, including numeric principal-point coordinates.
        """
        if not isinstance(other, PinholeCameraModel):
            return NotImplemented
        return (
            self.image_resolution == other.image_resolution
            and self.focal_length_x == other.focal_length_x
            and self.focal_length_y == other.focal_length_y
            and float(self.principal_point.x) == float(other.principal_point.x)
            and float(self.principal_point.y) == float(other.principal_point.y)
            and self.principal_point.reference_frame
            is other.principal_point.reference_frame
            and self.distortion == other.distortion
        )

    @classmethod
    def from_field_of_view(
        cls,
        resolution: CameraResolution,
        field_of_view: FieldOfView,
        distortion: CameraDistortion | None = None,
    ) -> PinholeCameraModel:
        """
        Create a centered pinhole model for an angular image extent.

        :param resolution: Pixel dimensions of the modeled image.
        :param field_of_view: Horizontal and vertical angular extent.
        :param distortion: Lens distortion associated with the image.
        :return: Centered pinhole calibration matching the requested extent.
        """
        cls._validate_field_of_view(field_of_view)
        focal_length_x = resolution.width / (
            2.0 * math.tan(field_of_view.horizontal_angle / 2.0)
        )
        focal_length_y = resolution.height / (
            2.0 * math.tan(field_of_view.vertical_angle / 2.0)
        )
        return cls(
            image_resolution=resolution,
            focal_length_x=focal_length_x,
            focal_length_y=focal_length_y,
            principal_point=Point2(x=resolution.width / 2.0, y=resolution.height / 2.0),
            distortion=(distortion if distortion is not None else NoCameraDistortion()),
        )

    @property
    def resolution(self) -> CameraResolution:
        return self.image_resolution

    @property
    def field_of_view(self) -> FieldOfView:
        """
        Calculate the angular extent represented by the calibration.
        """
        horizontal_angle = math.atan(
            float(self.principal_point.x) / self.focal_length_x
        ) + math.atan(
            (self.image_resolution.width - float(self.principal_point.x))
            / self.focal_length_x
        )
        vertical_angle = math.atan(
            float(self.principal_point.y) / self.focal_length_y
        ) + math.atan(
            (self.image_resolution.height - float(self.principal_point.y))
            / self.focal_length_y
        )
        return FieldOfView(
            horizontal_angle=horizontal_angle,
            vertical_angle=vertical_angle,
        )

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """
        Return the pinhole calibration matrix.
        """
        return np.array(
            [
                [self.focal_length_x, 0.0, float(self.principal_point.x)],
                [0.0, self.focal_length_y, float(self.principal_point.y)],
                [0.0, 0.0, 1.0],
            ]
        )

    def scaled(self, scale_x: float, scale_y: float) -> PinholeCameraModel:
        """
        Return the calibration for an image scaled along both axes.

        :param scale_x: Horizontal image scale.
        :param scale_y: Vertical image scale.
        :return: Calibration matching the scaled image.
        """
        return PinholeCameraModel(
            image_resolution=CameraResolution(
                width=int(self.image_resolution.width * scale_x),
                height=int(self.image_resolution.height * scale_y),
            ),
            focal_length_x=self.focal_length_x * scale_x,
            focal_length_y=self.focal_length_y * scale_y,
            principal_point=Point2(
                x=float(self.principal_point.x) * scale_x,
                y=float(self.principal_point.y) * scale_y,
            ),
            distortion=self.distortion,
        )


# %% Distortion and range


class CameraDistortionModel(StrEnum):
    """
    Identify the mathematical model used for lens distortion.
    """

    NONE = "none"
    """
    The image has no modeled lens distortion.
    """

    PLUMB_BOB = "plumb_bob"
    """
    Follow ROS order ``(k1, k2, t1, t2, k3)`` for plumb-bob distortion.
    """

    RATIONAL_POLYNOMIAL = "rational_polynomial"
    """
    Follow ROS order ``(k1, k2, t1, t2, k3, k4, k5, k6)``.
    """

    EQUIDISTANT = "equidistant"
    """
    Follow ROS order ``(k1, k2, k3, k4)`` for equidistant distortion.
    """

    @property
    def coefficient_count(self) -> int:
        """
        Return the number of coefficients defined by this model.
        """
        return CameraDistortion.get_type_for_model(self).coefficient_count


@dataclass(frozen=True, kw_only=True)
class CameraDistortion(ABC):
    """
    Describe lens distortion using ROS ``CameraInfo`` model conventions.
    """

    def __post_init__(self) -> None:
        """
        Validate that every coefficient is finite.
        """
        if not all(
            math.isfinite(coefficient) for coefficient in self.to_ordered_coefficients()
        ):
            raise InvalidCameraDistortionError(
                model=self.model.value,
                reason="coefficients must be finite",
            )

    @property
    @abstractmethod
    def model(self) -> CameraDistortionModel:
        """
        Return the mathematical model represented by this value.
        """

    @abstractmethod
    def to_ordered_coefficients(self) -> tuple[float, ...]:
        """
        Return coefficients in the ROS order defined by :attr:`model`.
        """

    @property
    @abstractmethod
    def coefficient_count(self) -> int:
        """
        Return the number of coefficients required by this distortion model.
        """

    @classmethod
    @abstractmethod
    def _from_ordered_coefficients(
        cls, ordered_coefficients: tuple[float, ...]
    ) -> Self:
        """
        Construct distortion from coefficients with a validated count and ROS order.
        """

    @staticmethod
    def get_type_for_model(model: CameraDistortionModel) -> type[CameraDistortion]:
        """
        Find the concrete distortion implementation declaring the model identifier.

        :param model: Mathematical model to resolve.
        :return: The unique concrete implementation of the model.
        :raises InvalidCameraDistortionError: If no unique implementation exists.
        """
        candidates = [
            distortion_type
            for distortion_type in concrete_subclasses(CameraDistortion)
            if distortion_type.model == model
        ]
        if len(candidates) != 1:
            raise InvalidCameraDistortionError(
                model=model.value,
                reason=f"requires one concrete implementation, found {len(candidates)}",
            )
        return candidates[0]

    @staticmethod
    def from_ordered_coefficients(
        model: CameraDistortionModel,
        coefficients: Sequence[float],
    ) -> CameraDistortion:
        """
        Parse a coefficient sequence ordered according to ROS ``CameraInfo``.

        :param model: Mathematical model defining coefficient order.
        :param coefficients: Coefficients ordered according to ``model``.
        :return: Structured distortion value.
        :raises InvalidCameraDistortionError: If the coefficient count is invalid or the
            model has no unique concrete implementation.
        """
        distortion_type = CameraDistortion.get_type_for_model(model)
        ordered_coefficients = tuple(coefficients)
        if len(ordered_coefficients) != distortion_type.coefficient_count:
            raise InvalidCameraDistortionError(
                model=model.value,
                reason=(
                    f"requires {distortion_type.coefficient_count} coefficients, got "
                    f"{len(ordered_coefficients)}"
                ),
            )
        return distortion_type._from_ordered_coefficients(ordered_coefficients)


@dataclass(frozen=True, kw_only=True)
class NoCameraDistortion(CameraDistortion):
    """
    Represent an image without modeled lens distortion.
    """

    @classproperty
    def model(cls) -> CameraDistortionModel:
        return CameraDistortionModel.NONE

    @classproperty
    def coefficient_count(cls) -> int:
        return 0

    @classmethod
    def _from_ordered_coefficients(
        cls, ordered_coefficients: tuple[float, ...]
    ) -> Self:
        return cls()

    def to_ordered_coefficients(self) -> tuple[float, ...]:
        return ()


@dataclass(frozen=True, kw_only=True)
class RadialTangentialCameraDistortion(CameraDistortion, ABC):
    """
    Provide coefficients shared by ROS radial-tangential models.

    ROS names the tangential terms ``t1`` and ``t2``. They are the coefficients commonly
    written as ``p1`` and ``p2`` in camera-model equations.
    """

    radial_coefficient_1: float = 0.0
    """
    First radial distortion coefficient, conventionally ``k1``.
    """

    radial_coefficient_2: float = 0.0
    """
    Second radial distortion coefficient, conventionally ``k2``.
    """

    tangential_coefficient_1: float = 0.0
    """
    First tangential coefficient, conventionally ``p1`` and named ``t1`` by ROS.
    """

    tangential_coefficient_2: float = 0.0
    """
    Second tangential coefficient, conventionally ``p2`` and named ``t2`` by ROS.
    """

    radial_coefficient_3: float = 0.0
    """
    Third radial distortion coefficient, conventionally ``k3``.
    """

    @property
    def radial_tangential_coefficients(self) -> tuple[float, ...]:
        """
        Return the coefficients shared by radial-tangential models.
        """
        return (
            self.radial_coefficient_1,
            self.radial_coefficient_2,
            self.tangential_coefficient_1,
            self.tangential_coefficient_2,
            self.radial_coefficient_3,
        )


@dataclass(frozen=True, kw_only=True)
class PlumbBobCameraDistortion(RadialTangentialCameraDistortion):
    """
    Represent radial-tangential distortion using the plumb-bob model.
    """

    @classproperty
    def model(cls) -> CameraDistortionModel:
        return CameraDistortionModel.PLUMB_BOB

    @classproperty
    def coefficient_count(cls) -> int:
        return 5

    @classmethod
    def _from_ordered_coefficients(
        cls, ordered_coefficients: tuple[float, ...]
    ) -> Self:
        return cls(
            radial_coefficient_1=ordered_coefficients[0],
            radial_coefficient_2=ordered_coefficients[1],
            tangential_coefficient_1=ordered_coefficients[2],
            tangential_coefficient_2=ordered_coefficients[3],
            radial_coefficient_3=ordered_coefficients[4],
        )

    def to_ordered_coefficients(self) -> tuple[float, ...]:
        return self.radial_tangential_coefficients


@dataclass(frozen=True, kw_only=True)
class RationalPolynomialCameraDistortion(RadialTangentialCameraDistortion):
    """
    Represent radial-tangential distortion with rational radial terms.
    """

    radial_coefficient_4: float = 0.0
    """
    Fourth radial distortion coefficient, conventionally ``k4``.
    """

    radial_coefficient_5: float = 0.0
    """
    Fifth radial distortion coefficient, conventionally ``k5``.
    """

    radial_coefficient_6: float = 0.0
    """
    Sixth radial distortion coefficient, conventionally ``k6``.
    """

    @classproperty
    def model(cls) -> CameraDistortionModel:
        return CameraDistortionModel.RATIONAL_POLYNOMIAL

    @classproperty
    def coefficient_count(cls) -> int:
        return 8

    @classmethod
    def _from_ordered_coefficients(
        cls, ordered_coefficients: tuple[float, ...]
    ) -> Self:
        return cls(
            radial_coefficient_1=ordered_coefficients[0],
            radial_coefficient_2=ordered_coefficients[1],
            tangential_coefficient_1=ordered_coefficients[2],
            tangential_coefficient_2=ordered_coefficients[3],
            radial_coefficient_3=ordered_coefficients[4],
            radial_coefficient_4=ordered_coefficients[5],
            radial_coefficient_5=ordered_coefficients[6],
            radial_coefficient_6=ordered_coefficients[7],
        )

    def to_ordered_coefficients(self) -> tuple[float, ...]:
        return (
            *self.radial_tangential_coefficients,
            self.radial_coefficient_4,
            self.radial_coefficient_5,
            self.radial_coefficient_6,
        )


@dataclass(frozen=True, kw_only=True)
class EquidistantCameraDistortion(CameraDistortion):
    """
    Represent fisheye distortion using the equidistant model.
    """

    radial_coefficient_1: float = 0.0
    """
    First equidistant distortion coefficient, conventionally ``k1``.
    """

    radial_coefficient_2: float = 0.0
    """
    Second equidistant distortion coefficient, conventionally ``k2``.
    """

    radial_coefficient_3: float = 0.0
    """
    Third equidistant distortion coefficient, conventionally ``k3``.
    """

    radial_coefficient_4: float = 0.0
    """
    Fourth equidistant distortion coefficient, conventionally ``k4``.
    """

    @classproperty
    def model(cls) -> CameraDistortionModel:
        return CameraDistortionModel.EQUIDISTANT

    @classproperty
    def coefficient_count(cls) -> int:
        return 4

    @classmethod
    def _from_ordered_coefficients(
        cls, ordered_coefficients: tuple[float, ...]
    ) -> Self:
        return cls(
            radial_coefficient_1=ordered_coefficients[0],
            radial_coefficient_2=ordered_coefficients[1],
            radial_coefficient_3=ordered_coefficients[2],
            radial_coefficient_4=ordered_coefficients[3],
        )

    def to_ordered_coefficients(self) -> tuple[float, ...]:
        return (
            self.radial_coefficient_1,
            self.radial_coefficient_2,
            self.radial_coefficient_3,
            self.radial_coefficient_4,
        )


@dataclass
class CameraRange:
    """
    Describe the usable distance interval of a camera.
    """

    minimum_distance: float = 0.0
    """
    Nearest usable distance in meters.
    """

    maximum_distance: float = math.inf
    """
    Farthest usable distance in meters.
    """

    def __post_init__(self) -> None:
        """
        Validate the distance interval.
        """
        if not math.isfinite(self.minimum_distance) or self.minimum_distance < 0.0:
            raise InvalidCameraRangeError(
                minimum_distance=self.minimum_distance,
                maximum_distance=self.maximum_distance,
            )
        if (
            math.isnan(self.maximum_distance)
            or self.maximum_distance <= self.minimum_distance
        ):
            raise InvalidCameraRangeError(
                minimum_distance=self.minimum_distance,
                maximum_distance=self.maximum_distance,
            )


class CameraModality(StrEnum):
    """
    Identify a kind of image data produced by a camera.
    """

    COLOR = "color"
    """
    Color intensity images.
    """

    DEPTH = "depth"
    """
    Per-pixel depth images.
    """

    INFRARED = "infrared"
    """
    Infrared intensity images.
    """

    SEGMENTATION = "segmentation"
    """
    Per-pixel semantic or instance labels.
    """
