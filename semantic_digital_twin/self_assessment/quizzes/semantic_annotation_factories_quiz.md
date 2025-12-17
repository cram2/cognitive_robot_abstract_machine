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

(semantic-annotation-factories-quiz)=
# Semantic Annotation Factories Quiz

This page provides a self-check quiz for the tutorial: [](semantic-annotation-factories).  
Source: Jupyter quiz. $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "What is the main purpose of factories in the semantic world?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Convenience builders that create consistent worlds and views", "correct": True},
        {"answer": "Physics simulation of rigid bodies", "correct": False},
        {"answer": "Rendering visualization in RViz2", "correct": False},
        {"answer": "Parsing URDF files", "correct": False}
      ],
    },
    {
      "question": "Which factory combination creates a drawer with a centered handle?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "DrawerFactory + ContainerFactory + HandleFactory", "correct": True},
        {"answer": "WorldFactory + MeshFactory", "correct": False},
        {"answer": "URDFFactory only", "correct": False},
        {"answer": "RayTracerFactory", "correct": False}
      ],
    },
    {
      "question": "How is the handle centered semantically in the example?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Using SemanticPositionDescription with FULLY_CENTER directions", "correct": True},
        {"answer": "By setting visual color to center", "correct": False},
        {"answer": "By passing center=True to HandleFactory", "correct": False},
        {"answer": "By rotating 180 degrees", "correct": False}
      ],
    },
    {
      "question": "How can an extra handle be merged into the world at a different pose?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "world.merge_world_at_pose(other_world, TransformationMatrix.from_xyz_rpy(...))", "correct": True},
        {"answer": "world.add_handle(other_world)", "correct": False},
        {"answer": "handle.attach_to(world)", "correct": False},
        {"answer": "URDFParser.merge(world)", "correct": False}
      ],
    },
    {
      "question": "Which query returns all Handle views using EQL?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "an(entity(variable(Handle, world.views)))", "correct": True},
        {"answer": "world.get_views_by_type(Handle)", "correct": False},
        {"answer": "select * from views where type='Handle'", "correct": False},
        {"answer": "handles = world.views['Handle']", "correct": False}
      ],
    }
]

import json
json_str = json.dumps(questions)
json.loads(json_str) 

display_quiz(questions)
```
