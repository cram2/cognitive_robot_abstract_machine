# krrood_spatial_math

The spatial mathematics robot and image packages share.

- `krrood_spatial_math.cartesian_frame_3d`: `CartesianFrame3D`, a right-handed Cartesian
  frame in three dimensions, in metres, that values are given in, and how a serialized
  value refers to one.
- `krrood_spatial_math.spatial_types`, with the extra `krrood_spatial_math[casadi]`:
  `Point2`, `Point3`, `Vector3`, `RotationMatrix`, `Quaternion`, `AxisAngle`,
  `HomogeneousTransformationMatrix`, `Pose`, `Pose2D`, and the motions `Displacement`,
  `Twist` and `ScrewAxis`, as symbolic expressions that can also hold plain numbers.

The documentation, including how a twist or a screw axis becomes a displacement, is at
https://cram2.github.io/cognitive_robot_abstract_machine/krrood_spatial_math.
