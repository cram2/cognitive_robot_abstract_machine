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
- `krrood_spatial_math.vector`: `Vector[frame]`, a translation of an affine space given
  by its coordinates in an affine frame, the fields a subclass marks with
  `AxisCoordinate`. A second parameter names the coordinate type when it is not
  `float`. Its `norm` and `dot` are measured by the kind of frame, and an affine one
  that is not Cartesian raises `FrameMeasuresNoLengthsError`. A change of frame maps
  its axes but never shifts it, as homogeneous coordinates with weight 0.
  `CrossProductVector` is a vector of three-dimensional space with a `cross` product.
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

## Adding a space

A new space needs a space, a kind of frame on it with a frame of that kind, and a
vector type given in that kind of frame and a point type moved by it. `Point` and
`Vector` declare the frame field, do the arithmetic and refuse values of two frames, so
the point and vector only mark their coordinates:

```python
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from krrood.patterns.interface import Interface
from krrood_spatial_math.coordinate_system import AffineFrame, AxisCoordinate
from krrood_spatial_math.point import Point
from krrood_spatial_math.space import EuclideanSpace
from krrood_spatial_math.vector import Vector


class ImagePlane(EuclideanSpace):
    """The plane a camera's images lie on."""


class PixelFrame(AffineFrame[ImagePlane, str], Interface):
    """Pixel coordinates on an image plane, whose axes may differ in length."""


@dataclass(eq=False, kw_only=True)
class CameraPixels(PixelFrame):
    """The pixels of one camera's images."""

    name: str
    space: ImagePlane | None = None
    id: UUID = field(default_factory=uuid4)


@dataclass(kw_only=True)
class PixelVector(Vector[PixelFrame]):
    u: float = field(metadata=AxisCoordinate().as_dict())
    v: float = field(metadata=AxisCoordinate().as_dict())


@dataclass(kw_only=True)
class PixelPoint(Point[PixelVector]):
    u: float = field(metadata=AxisCoordinate().as_dict())
    v: float = field(metadata=AxisCoordinate().as_dict())
```

With `frame = CameraPixels(name="camera")`,
`PixelPoint(u=4, v=2, coordinate_system=frame) - PixelPoint(u=1, v=1, coordinate_system=frame)`
is `PixelVector(u=3, v=1, coordinate_system=frame)`. Combining values of two frames
raises `ReferenceFrameMismatchError`, and the point's kind of frame is always its
vector's, since it is read from the vector type. A change
between two pixel frames implements `AffineCoordinateChange[PixelFrame]` and its
`apply_to_homogeneous_coordinates`; `apply_transformation` then moves pixel points and
vectors with no code of their own.

Points of two spaces, such as a body and an image of it, relate only through a map
between the spaces, such as a camera, never through a coordinate change.
