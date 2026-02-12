---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

(loading-worlds)=
# Loading worlds from files

This tutorial shows how to load a world description from a file into a `World` object using the `URDFParser`.

First, we need to compose the path to your world file.

```{code-cell} ipython3
import logging
import os

from pkg_resources import resource_filename

logging.disable(logging.CRITICAL)
apartment = os.path.join(resource_filename("semantic_digital_twin", "../../"), "resources", "urdf", "apartment.urdf")

```

Next we need to initialize a parser that reads this file. There are many parsers available. 
To read this specific urdf file, the `https://github.com/code-iai/iai_maps/tree/ros-jazzy/` repository needs to be installed
inside your ROS2 workspace.

```{code-cell} ipython3
from semantic_digital_twin.adapters.urdf import URDFParser  
  
parser = URDFParser.from_file(apartment)  
world = parser.parse()  
```

This constructs a world you can visualize, interact and annotate. Be aware that worlds loaded from files have no semantic annotations and serve as purely kinematic models.
Supported file formates are:
- urdf
- mjcf
- stl
