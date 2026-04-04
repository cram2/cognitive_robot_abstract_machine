# Multiverse Simulators

The **Multiverse Simulators** is a collection of simulators for the [Multiverse Framework](https://github.com/Multiverse-Framework/Multiverse).

---

## 📋 Prerequisites

- **Python** ≥ 3.10 (Linux), 3.12 (Windows)

Each simulator connector will have different dependencies, e.g.:
  - Python packages for MuJoCo listed in [requirements.txt](https://github.com/Multiverse-Framework/Multiverse-Simulators-Connector/blob/main/src/mujoco_connector/requirements.txt)
  - Python packages for Isaac Sim listed in [requirements.txt](https://github.com/Multiverse-Framework/Multiverse-Simulators-Connector/blob/main/src/isaac_sim_connector/requirements.txt)

Install the required packages:

```bash
pip install -r src/mujoco_connector/requirements.txt
```

---

## ⚙️ Setup

First, clone the repository:

```bash
git clone https://github.com/Multiverse-Framework/Multiverse-Simulators-Connector --depth 1
```
---

Then install it as a local Python package using a symbolic link (editable mode):
Currently, only MuJoCo is installed via pip; support for Isaac Sim and others will be added later.

```bash
pip install -e .
```

This allows you to make changes to the source code and immediately reflect them without reinstalling.

You can then test it in a Python shell:

```python
from mujoco_connector import MultiverseMujocoConnector
from multiverse_simulator import MultiverseSimulatorState, MultiverseSimulatorConstraints

simulator = MultiverseMujocoConnector(file_path=os.path.join(resources_path, "mjcf/unitree/h1_scene.xml"))
constraints = MultiverseSimulatorConstraints(max_simulation_time=10.0)
simulator.start(constraints=constraints)
while simulator.state != MultiverseSimulatorState.STOPPED:
    time.sleep(1)
```
