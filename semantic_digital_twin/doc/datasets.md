# Datasets

Semantic Digital Twin can load datasets from internet resources.
The results of the loaded datasets are completely function digital twins 
(World instances including Semantic Annotations, Kinematics, etc.).


## Sage
Scenes from [Sage](https://nvlabs.github.io/sage/) can be loaded with:

```python
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader

loader = Sage10kDatasetLoader()
scene = loader.create_scene(scene_url=Sage10kDatasetLoader.available_scenes()[0])
world = scene.create_world()
```

## Sapien / PartNet

Articulated assets from the [PartNet-Mobility](https://sapien.ucsd.edu/browse) dataset can be loaded with:

```python
from semantic_digital_twin.adapters.partnet_mobility_dataset.loader import PartNetMobilityDatasetLoader

loader = PartNetMobilityDatasetLoader()
world = loader.load(model_id=179) # model_id can be found at https://sapien.ucsd.edu/browse
```

Note that this requires the `sapien` library to be installed and the `SAPIEN_ACCESS_TOKEN` environment variable to be set.

## RoboCasa

Objects, fixtures, full kitchen scenes, and manipulation tasks from [RoboCasa](https://github.com/robocasa/robocasa) can be loaded with:

```python
from semantic_digital_twin.adapters.robocasa_dataset.loader import RoboCasaDatasetLoader
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaKitchenApplianceCategory,
    RoboCasaObjectCategory,
)

loader = RoboCasaDatasetLoader()

kitchen_world = loader.load_kitchen(layout_id=..., style_id=...)  # a full kitchen scene
appliance_world = loader.load_kitchen_appliance(RoboCasaKitchenApplianceCategory.CABINET)  # a single appliance
object_world = loader.load_object(RoboCasaObjectCategory.APPLE)  # a single object
```

A RoboCasa task (for example `"TurnOnMicrowave"`) can be loaded together with the scene it is defined
over. `load_task` returns a `RoboCasaTask` binding the `World` to the task's natural-language
instruction, the bodies to be manipulated, and the pose the robot should start at. RoboCasa's own
robot is stripped from the world, since `semantic_digital_twin` owns the robot.

```python
task = loader.load_task("TurnOnMicrowave", layout_id=..., style_id=...)
task.instruction          # e.g. "Press the start button on the microwave."
task.manipulated_objects  # the bodies the task requires the robot to interact with
task.robot_base_pose      # where to spawn the semantic_digital_twin-owned robot
```

Note that this requires the `robocasa` and `robosuite` libraries to be installed (`robosuite` must be
installed from git, `pip install git+https://github.com/ARISE-Initiative/robosuite.git`), and the
fixture/object assets to be downloaded via `python -m robocasa.scripts.download_kitchen_assets`
(pointed at by `RoboCasaDatasetLoader.directory`, `~/robocasa-assets` by default).
