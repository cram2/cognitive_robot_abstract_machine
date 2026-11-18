# Domain models

Packages that each model one domain. A domain model depends on `krrood`, on other domain
models and on libraries outside this repository.

- `krrood_spatial_math`: the spatial mathematics robot and image packages share; with the
  extra `krrood_spatial_math[casadi]`, points, vectors, rotations and transformations in
  Cartesian frames.
- `krrood_synchronization`: streams of messages that keep copies of a model in different
  processes in step, such as worlds and statecharts.
