---
jupytext:
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

(graph-of-convex-sets-quiz)=
# Graph of Convex Sets Quiz

This page provides a self-check quiz for the tutorial: Graph of Convex Sets.  
Source: Jupyter quiz. $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "What does the Graph of Convex Sets (GCS) represent in this context?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Adjacency of convex free-space cells (boxes)", "correct": True},
        {"answer": "A rendering graph for meshes", "correct": False},
        {"answer": "A database of URDF files", "correct": False},
        {"answer": "A ROS2 node graph", "correct": False}
      ],
    },
    {
      "question": "Which class constructs free space from a world?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "GraphOfConvexSets.free_space_from_world(...)\n", "correct": True},
        {"answer": "URDFParser.from_file(...)\n", "correct": False},
        {"answer": "WorldState.make_free_space(...)\n", "correct": False},
        {"answer": "RayTracer.update_scene(...)\n", "correct": False}
      ],
    },
    {
      "question": "What limits the area in which the free-space is computed in the example?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "A BoundingBoxCollection called search_space", "correct": True},
        {"answer": "A camera frustum", "correct": False},
        {"answer": "The world bounds auto-detected from visuals", "correct": False},
        {"answer": "The URDF joint limits", "correct": False}
      ],
    },
    {
      "question": "Which library is used to display 3D plots of occupied/free space?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Plotly", "correct": True},
        {"answer": "Matplotlib 2D", "correct": False},
        {"answer": "RViz2", "correct": False},
        {"answer": "Gazebo", "correct": False}
      ],
    },
    {
      "question": "How is a path through free space computed between start and goal?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "gcs.path_from_to(start, goal)", "correct": True},
        {"answer": "URDFParser.plan_path(start, goal)", "correct": False},
        {"answer": "RayTracer.trace_path(start, goal)", "correct": False},
        {"answer": "Use EQL to get the path", "correct": False}
      ],
    }
]

import json
json_str = json.dumps(questions)
json.loads(json_str) 

display_quiz(questions)
```
