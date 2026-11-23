# krrood_spatial_math

The spatial mathematics that robot and image packages share. It models a space, such as
the physical space a robot moves in or the plane an image lies on, separately from the
coordinate systems on it, so that a value always says which coordinate system its
coordinates are given in, and coordinates are only combined after changing them into one
coordinate system. Its base depends on `krrood` alone.

- `krrood_spatial_math.space`: `Space`, a set of points with a geometry, and
  `EuclideanSpace`, one with distances and angles, such as the physical space a robot's
  bodies are in or the plane of a frame. A value's dimension is the number of its
  coordinates.
- `krrood_spatial_math.coordinate_system`: `CoordinateSystem`, what coordinates on one
  space are given in; `AffineFrame`, one with an origin and straight axes, and
  `CartesianFrame`, an affine frame whose axes are perpendicular and of unit length,
  whose kind measures the length of a vector, the inner product of two and the distance
  between two points;
  `CoordinateRepresentation`, a point, direction or orientation given by its
  coordinates in a coordinate system, which `apply_transformation` re-expresses in the
  target coordinate system of a change;
  `CoordinateChange`, between two coordinate systems of the same space, which can be
  inverted and composed; `AffineCoordinateChange`, between affine frames, which keeps
  straight lines and maps homogeneous coordinates; and `RigidCoordinateChange`, between
  Cartesian frames, which also keeps distances and angles and splits into its
  `rotation_matrix` and `translation`.
- `krrood_spatial_math.cartesian_frame_3_dimensional`: `CartesianFrame3Dimensional`, the right-handed
  Cartesian frame of physical space that positions and directions are expressed in,
  and how serialized values refer to one.
- `krrood_spatial_math.embedded_plane`: `EmbeddedPlane`, a plane lying in a known
  three-dimensional space: the x-y plane of a `CartesianFrame3Dimensional`, whose
  `normal` is that frame's `z_axis`. A plane with another normal is the x-y plane of a
  frame whose z-axis is that normal.
- `krrood_spatial_math.cartesian_frame_2_dimensional`: `CartesianFrame2Dimensional`, the
  right-handed Cartesian frame on the `EmbeddedPlane` of a `CartesianFrame3Dimensional`,
  with that frame's origin, x-axis and y-axis, that values on the plane are given in.
- `krrood_spatial_math.vector`: `Vector[frame, coordinate]`, a translation of an affine
  space given by its coordinates in an affine frame, one per axis that frame's kind
  names in its `axes`, held in the field or property of that name, each of the
  coordinate type, such as `float` or `sm.Scalar`. Its `norm` and `dot` are measured by
  the kind of frame, and an affine one that is not Cartesian raises
  `FrameMeasuresNoLengthsError`. Its `cross` is the kind of frame's too: only a
  `CartesianFrame3Dimensional` has a cross product, and any other kind raises
  `FrameHasNoCrossProductError`. A change of frame maps its axes but never shifts it, as
  homogeneous coordinates with weight 0.
- `krrood_spatial_math.point`: `Point[vector]`, a point of an affine space, given in the
  kind of frame of its vector type and measured by it in the same way. The difference
  of two points is a vector, and a point moved by a vector is a point. A change of frame
  also shifts it, as homogeneous coordinates with weight 1, and its change type, like
  its kind of frame, is read from its vector type.
- `krrood_spatial_math.pose`: `Pose` and `Orientation`, a frame given in another, with
  and without its position. Each is the value of a rigid change, and a change of frame
  moves it by composing with that change, an orientation with the rotation only.
- `krrood_spatial_math.motion_vector_pair`: `MotionVectorPair[vector]`, a rigid motion
  as how a body turns and how the point at the frame's origin moves, which a change of
  frame moves by its adjoint action.
- `krrood_spatial_math.line`: `Line[point]`, the points `point + direction * t` for every
  number `t`, running along its point type's vector type, and `LineSegment[line]`, the
  part of a line between two of its points.
- `krrood_spatial_math.space_map`: `SpaceMap`, a map from the points of one space to
  those of another, such as a camera, and `OrthogonalProjection`, one onto a plane along
  its unit normal, which derives its projection lines and `unproject` from that normal
  and from where a point of the plane lies in the source space.
- `krrood_spatial_math.spatial_types`, with the extra `krrood_spatial_math[casadi]`:
  `Point3Dimensional`, `Vector3Dimensional`, `Line3Dimensional`,
  `RotationMatrix`, `Quaternion`, `AxisAngle`, `HomogeneousTransformationMatrix3Dimensional`,
  `Pose3Dimensional`, `Displacement`, `Twist` and `ScrewAxis`, as symbolic expressions
  that can also hold plain numbers. Values derive from `SymbolicSpatialValue` and changes
  from `SymbolicSpatialChange`. A value type binds only the change type that re-expresses
  it, `SymbolicSpatialValue[HomogeneousTransformationMatrix3Dimensional]`, which fixes its
  kind of frame and coordinate type, and `transform` takes a transformation as a change of
  that type. How a change moves a value comes from its kind of value: the vector and
  point are a `Vector` and a `Point`, the pose a `Pose`, `Quaternion` and `AxisAngle` an
  `Orientation`, and `Displacement`, `Twist` and `ScrewAxis` a `MotionVectorPair`. A
  `RotationMatrix` is the transformation without translation, SO(3) within SE(3).
- `krrood_spatial_math.spatial_types_2_dimensional`, with the same extra: their
  counterparts on a plane, `Point2Dimensional`, `Vector2Dimensional` and
  `Pose2Dimensional`, which lie on the x-y plane of their reference frame, an
  `EmbeddedPlane`, and are given in its `CartesianFrame2Dimensional`. They are
  re-expressed only in frames on that plane; reaching another plane, even a parallel
  one, is an `EmbeddedPlaneProjection`.
  `HomogeneousTransformationMatrix2Dimensional` is the rigid change between two frames
  on one plane, and `EmbeddedPlaneProjection` the orthogonal projection onto a frame's
  plane. Their change type is `HomogeneousTransformationMatrix2Dimensional`, and they
  are moved as the same kinds of value; a transformation refuses a value of the other
  dimension.

| Space | Frame | Coordinate change | Kept by the change |
|---|---|---|---|
| `EuclideanSpace` | `CartesianFrame3Dimensional`, metres | `RigidCoordinateChange` | distances, angles |
| `EmbeddedPlane` | `CartesianFrame2Dimensional`, metres | `RigidCoordinateChange` | distances, angles |

Other domain models add their own spaces on these interfaces; `krrood_image`, for
example, adds image planes, whose pixel frames are affine and related by affine changes.

The documentation, including how a twist or a screw axis becomes a displacement, is at
https://cram2.github.io/cognitive_robot_abstract_machine/krrood_spatial_math.

## Adding a space

This example adds the pixels of camera images. A camera's image plane is a plane, but
its pixels need not be square, so its pixel frames are affine, not Cartesian: they
measure no lengths or angles. Cropping and resizing an image changes the pixel frame,
which is an affine change. A new space needs five things:

1. the space, here `ImagePlane`;
2. a kind of frame on it that names its axes, here `PixelFrame` with the axes u and v;
3. a frame of that kind, here `ImagePixels`, the pixels of one image;
4. a vector type and a point type, which only declare one field per axis;
5. the coordinate changes between its frames, here `CropAndResize`.

`Point` and `Vector` do the arithmetic, refuse to combine values of two frames, and move
values through any affine change, so steps 1 to 4 hold no logic:

```python
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID, uuid4

from typing_extensions import Self

from krrood.patterns.classproperty import classproperty
from krrood.patterns.interface import Interface
from krrood_spatial_math.coordinate_system import AffineCoordinateChange, AffineFrame
from krrood_spatial_math.point import Point
from krrood_spatial_math.space import EuclideanSpace
from krrood_spatial_math.vector import Vector


class ImagePlane(EuclideanSpace):
    """The plane a camera's images lie on."""


class PixelAxis(StrEnum):
    """An axis of a pixel frame: u runs right along a row, v down along a column."""

    U = "u"
    V = "v"


class PixelFrame(AffineFrame[ImagePlane, str], Interface):
    """Pixel coordinates on an image plane, whose axes may differ in length."""

    @classproperty
    def axes(cls) -> tuple[PixelAxis, ...]:
        return (PixelAxis.U, PixelAxis.V)


@dataclass(eq=False, kw_only=True)
class ImagePixels(PixelFrame):
    """The pixels of one image, counted from its top-left corner."""

    name: str
    space: ImagePlane | None = None
    id: UUID = field(default_factory=uuid4)


@dataclass(kw_only=True)
class PixelVector(Vector[PixelFrame, float]):
    """An offset between two pixels, such as the width of a bounding box."""

    u: float
    v: float


@dataclass(kw_only=True)
class PixelPoint(Point[PixelVector]):
    """A position in an image, such as a detected keypoint."""

    u: float
    v: float
```

Step 5 is the only code with logic: the change says how it maps homogeneous
coordinates, and how it inverts and composes. A crop moves the origin to the crop's
top-left corner, and a resize scales each axis:

```python
@dataclass(kw_only=True)
class CropAndResize(AffineCoordinateChange[PixelFrame, float]):
    """From an image's pixels to those of a crop of it, scaled per axis."""

    source_coordinate_system: PixelFrame | None
    target_coordinate_system: PixelFrame | None
    crop_u: float = 0.0
    crop_v: float = 0.0
    scale_u: float = 1.0
    scale_v: float = 1.0

    def apply_to_homogeneous_coordinates(
        self, *, homogeneous_coordinates: tuple[float, ...]
    ) -> tuple[float, ...]:
        u, v, weight = homogeneous_coordinates
        return (
            self.scale_u * (u - self.crop_u * weight),
            self.scale_v * (v - self.crop_v * weight),
            weight,
        )

    def inverse(self) -> Self:
        return type(self)(
            source_coordinate_system=self.target_coordinate_system,
            target_coordinate_system=self.source_coordinate_system,
            crop_u=-self.crop_u * self.scale_u,
            crop_v=-self.crop_v * self.scale_v,
            scale_u=1 / self.scale_u,
            scale_v=1 / self.scale_v,
        )

    def __matmul__(self, other: Self) -> Self:
        return type(self)(
            source_coordinate_system=other.source_coordinate_system,
            target_coordinate_system=self.target_coordinate_system,
            crop_u=other.crop_u + self.crop_u / other.scale_u,
            crop_v=other.crop_v + self.crop_v / other.scale_v,
            scale_u=self.scale_u * other.scale_u,
            scale_v=self.scale_v * other.scale_v,
        )
```

Everything else comes from the interfaces. A thumbnail cropped from (100, 50) and halved
moves a keypoint by the crop and then scales it. A bounding box's width is only scaled:
a vector has no position, so the crop never shifts it.

```python
camera_image = ImagePixels(name="camera image")
thumbnail = ImagePixels(name="thumbnail")
thumbnail_from_camera_image = CropAndResize(
    source_coordinate_system=camera_image,
    target_coordinate_system=thumbnail,
    crop_u=100,
    crop_v=50,
    scale_u=0.5,
    scale_v=0.5,
)

keypoint = PixelPoint(u=300, v=250, coordinate_system=camera_image)
box_width = PixelVector(u=40, v=0, coordinate_system=camera_image)

assert keypoint.apply_transformation(
    transformation=thumbnail_from_camera_image
) == PixelPoint(u=100, v=100, coordinate_system=thumbnail)
assert box_width.apply_transformation(
    transformation=thumbnail_from_camera_image
) == PixelVector(u=20, v=0, coordinate_system=thumbnail)

# the change back returns the keypoint
assert keypoint.apply_transformation(
    transformation=thumbnail_from_camera_image
).apply_transformation(
    transformation=thumbnail_from_camera_image.inverse()
) == keypoint

# arithmetic stays within one frame
assert keypoint + box_width == PixelPoint(u=340, v=250, coordinate_system=camera_image)
thumbnail_keypoint = PixelPoint(u=1, v=1, coordinate_system=thumbnail)
```

What the interfaces refuse, each where it is first possible:

- `keypoint - thumbnail_keypoint`, points of two frames, raises `ReferenceFrameMismatchError`.
- `box_width.norm()` raises `FrameMeasuresNoLengthsError`: pixels may not be square, so a
  pixel frame measures no lengths. Measuring needs a camera model that maps pixels to
  the Cartesian frame of the camera.
- `box_width.cross(other=box_width)` raises `FrameHasNoCrossProductError`.
- A `PixelPoint` declared with fields `x` and `y` instead of `u` and `v` raises
  `MissingAxisCoordinatesError` where the class is defined.

Points of two spaces, such as a body and an image of it, relate only through a map
between the spaces, such as a camera, never through a coordinate change.

### Symbolic coordinates

Being symbolic is the coordinate type, not another kind of value. A point and vector
type that leave their coordinate type open take numbers or symbols, and the same change
moves both:

```python
from typing_extensions import Generic

from krrood.symbolic_math import symbolic_math as sm
from krrood_spatial_math.coordinate_system import GenericCoordinate


@dataclass(kw_only=True)
class OpenCoordinatePixelVector(
    Vector[PixelFrame, GenericCoordinate], Generic[GenericCoordinate]
):
    u: GenericCoordinate
    v: GenericCoordinate


@dataclass(kw_only=True)
class OpenCoordinatePixelPoint(
    Point[OpenCoordinatePixelVector[GenericCoordinate]], Generic[GenericCoordinate]
):
    u: GenericCoordinate
    v: GenericCoordinate


column = sm.FloatVariable(name="column")
row = sm.FloatVariable(name="row")
symbolic_keypoint = OpenCoordinatePixelPoint(u=column, v=row, coordinate_system=camera_image)

moved_keypoint = symbolic_keypoint.apply_transformation(
    transformation=thumbnail_from_camera_image
)
assert moved_keypoint.u.equivalent(other=0.5 * column - 50)
assert moved_keypoint.v.equivalent(other=0.5 * row - 25)
```

The symbolic spatial types, such as `Point3Dimensional`, are the Cartesian points and
vectors of a 3D frame whose coordinates are `sm.Scalar`, stored as one casadi matrix.
