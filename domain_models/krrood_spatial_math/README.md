# krrood_spatial_math

The spatial mathematics robot and image packages share.

- `krrood_spatial_math.cartesian_frame_3d`: `CartesianFrame3D`, a right-handed Cartesian
  frame in three dimensions, in metres, that values are given in, and how a serialized
  value refers to one.
- `krrood_spatial_math.spatial_types`, with the extra `krrood_spatial_math[casadi]`:
  `Point2`, `Point3`, `Vector3`, `RotationMatrix`, `Quaternion`, `AxisAngle`,
  `HomogeneousTransformationMatrix`, `Pose` and `Pose2D`, as symbolic expressions that
  can also hold plain numbers.
