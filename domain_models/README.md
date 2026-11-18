# Domain models

Packages that each model one domain. A domain model depends on `krrood`, on other domain
models and on libraries outside this repository.

- `krrood_spatial_math`: the spatial mathematics robot and image packages share; with the
  extra `krrood_spatial_math[casadi]`, points, vectors, rotations and transformations in
  Cartesian frames.
- `krrood_compute_hardware`: the processor cores, memory and GPUs a computation runs on,
  and how a machine is divided among processes that run side by side; with the extras
  `[jax]` and `[torch]`, the memory of the devices of those libraries.
