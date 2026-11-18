# krrood_spatial_math

The spatial mathematics that robot and image packages share. It models a space, such as
the physical space a robot moves in or the plane an image lies on, separately from the
coordinate systems on it, so that a value always says which coordinate system its
coordinates are given in, and coordinates are only combined after changing them into one
coordinate system. Its base depends on `krrood` alone.

- `krrood_spatial_math.space`: `Space`, a set of points with a geometry;
  `EuclideanSpace`, one with distances and angles, whose `dimension` each subclass
  fixes; `EuclideanSpace2Dimensional`, such as the floor a robot drives on; and
  `EuclideanSpace3Dimensional`, such as the physical space a robot's bodies are in.
- `krrood_spatial_math.coordinate_system`: `CoordinateSystem`, what coordinates on one
  space are given in; `AffineFrame`, one with an origin and straight axes, and
  `CartesianFrame`, an affine frame whose axes are perpendicular and of unit length,
  whose kind measures the length of a vector, the inner product of two and the distance
  between two points;
  `CoordinateRepresentation`, a point, direction or orientation given by its
  coordinates in a coordinate system;
  `CoordinateChange`, between two coordinate systems of the same space, which can be
  inverted and composed; `AffineCoordinateChange`, between affine frames, which keeps
  straight lines; and `RigidCoordinateChange`, between Cartesian frames, which also
  keeps distances and angles.
- `krrood_spatial_math.cartesian_frame_3_dimensional`: `CartesianFrame3Dimensional`, the right-handed
  Cartesian frame of physical space that positions and directions are expressed in,
  and how serialized values refer to one.
- `krrood_spatial_math.cartesian_frame_2_dimensional`: `CartesianFrame2Dimensional`, the right-handed
  Cartesian frame that values on a plane, such as the x-y plane of a
  `CartesianFrame3Dimensional`, are given in.
- `krrood_spatial_math.point`: `Point`, a point of an affine space given by its
  coordinates in an affine frame, and `MetricPoint`, one given in a Cartesian frame,
  whose distance to another point its frame kind measures. Each implementation binds
  the kind of frame, what its coordinates are stored as and the kind of vector that
  moves it; the frame's space fixes how many coordinates it has.
- `krrood_spatial_math.vector`: `Vector`, a translation of an affine space, and
  `MetricVector`, one given in a Cartesian frame, whose length and inner product its
  frame kind measures. The difference of two points is a vector, and a point moved by
  a vector is a point.
- `krrood_spatial_math.line`: `Line`, the points `point + direction * t` for every
  number `t`.
- `krrood_spatial_math.space_map`: `SpaceMap`, a map from the points of one space to
  those of another, such as a camera, and `OrthogonalProjection`, one onto a plane along
  its unit normal, which derives its projection lines and `unproject` from that normal
  and from where a point of the plane lies in the source space.
- `krrood_spatial_math.spatial_types`, with the extra `krrood_spatial_math[casadi]`:
  `CartesianPoint2Dimensional`, `CartesianPoint3Dimensional`, `Vector2Dimensional`, `Vector3Dimensional`, `Line3Dimensional`,
  `RotationMatrix`, `Quaternion`, `AxisAngle`, `HomogeneousTransformationMatrix3Dimensional`, `Pose3Dimensional`
  and `Pose2Dimensional`, as symbolic expressions that can also hold plain numbers. The 2D values
  lie on the x-y plane of their reference frame, a `FramePlane`, and are given in its
  `PlanarFrame`; `HomogeneousTransformationMatrix2Dimensional` is the rigid change between two
  frames on one plane, and `FramePlaneProjection` the orthogonal projection onto a
  frame's plane. Values derive from `SymbolicSpatialValue3Dimensional` (or `SymbolicSpatialValue2Dimensional`)
  and changes from `SymbolicSpatialChange`; a value names the group that acts on it:
  `SpecialEuclideanGroupAction` (SE(3)), `SpecialOrthogonalGroupAction` (SO(3)) or
  `PlanarSpecialEuclideanGroupAction` (SE(2)).

| Space | Frame | Coordinate change | Kept by the change |
|---|---|---|---|
| `EuclideanSpace3Dimensional` | `CartesianFrame3Dimensional`, metres | `RigidCoordinateChange` | distances, angles |
| `EuclideanSpace2Dimensional` | `CartesianFrame2Dimensional`, metres | `RigidCoordinateChange` | distances, angles |

Other domain models add their own spaces on these interfaces; `krrood_image`, for
example, adds image planes, whose pixel frames are affine and related by affine changes.

Points of two spaces, such as a body and an image of it, relate only through a map
between the spaces, such as a camera, never through a coordinate change.
