# krrood_spatial_math

The spatial mathematics that robot and image packages share: points, vectors,
rotations, poses and motions in Cartesian frames, as symbolic expressions that can
also hold plain numbers.

- {py:mod}`krrood_spatial_math.cartesian_frame_3_dimensional` holds
  {py:class}`~krrood_spatial_math.cartesian_frame_3_dimensional.CartesianFrame3Dimensional`,
  the frame values are given in.
- {py:mod}`krrood_spatial_math.spatial_types`, with the extra
  `krrood_spatial_math[casadi]`, holds the types themselves: positions and directions
  (`Point3Dimensional`, `Vector3Dimensional`), orientations (`RotationMatrix`,
  `Quaternion`, `AxisAngle`), poses (`HomogeneousTransformationMatrix3Dimensional`,
  `Pose3Dimensional`) and motions (`Displacement`, `Twist`, `ScrewAxis`).
- {py:mod}`krrood_spatial_math.spatial_types_2_dimensional`, with the same extra, holds
  their counterparts on the x-y plane of a frame, such as `Point2Dimensional` and
  `Pose2Dimensional`.

How to name frames and combine transformations is described in semantic_digital_twin's
[style guide](https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/style_guide.html).

```{tableofcontents}
```
