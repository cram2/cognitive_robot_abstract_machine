# krrood_spatial_math

The spatial mathematics that robot and image packages share: points, vectors,
rotations, poses and motions in Cartesian frames, as symbolic expressions that can
also hold plain numbers.

- {py:mod}`krrood_spatial_math.cartesian_frame_3d` holds
  {py:class}`~krrood_spatial_math.cartesian_frame_3d.CartesianFrame3D`, the frame
  values are given in.
- {py:mod}`krrood_spatial_math.spatial_types`, with the extra
  `krrood_spatial_math[casadi]`, holds the types themselves: positions and directions
  (`Point3`, `Vector3`), orientations (`RotationMatrix`, `Quaternion`, `AxisAngle`),
  poses (`HomogeneousTransformationMatrix`, `Pose`, `Pose2D`) and motions
  (`Displacement`, `Twist`, `ScrewAxis`).

How to name frames and combine transformations is described in semantic_digital_twin's
[style guide](https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/style_guide.html).

```{tableofcontents}
```
