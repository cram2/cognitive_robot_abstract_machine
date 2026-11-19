# krrood_compute_resources

The processor cores, memory and GPUs a computation runs on, and how a machine is divided
among processes that run side by side.

- `krrood_compute_resources.devices`: `Device`, the kinds of device a computation runs on.
- `krrood_compute_resources.memory`: `DeviceMemory`, the free memory of a device;
  `HostMemory`, the main memory of a Linux host within its control group; and
  `BudgetedMemory`, the share of a device's memory one process may use.
- `krrood_compute_resources.machine`: `Machine`, the cores and memory of the machine a
  process runs on; `MachineDivision`, which divides it into `MachineShare`s; and the
  environment and core affinity that keep a process within its share.
- `krrood_compute_resources.jax_memory`, with the extra `krrood_compute_resources[jax]`:
  `JaxDeviceMemory`, the memory of an accelerator jax stores arrays on.
- `krrood_compute_resources.torch_memory`, with the extra `krrood_compute_resources[torch]`:
  `TorchDeviceMemory`, the memory of a CUDA device torch stores tensors on.
