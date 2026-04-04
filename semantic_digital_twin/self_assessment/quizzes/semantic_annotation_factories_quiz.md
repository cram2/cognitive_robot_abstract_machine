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
      "question": "What is the main purpose of semantic annotation factories in the semantic digital twin?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Classmethods that create a body with simple geometry and the corresponding semantic annotation inside a world ", "correct": True},
        {"answer": "Physics simulation of rigid bodies", "correct": False},
        {"answer": "Rendering visualization in RViz2", "correct": False},
        {"answer": "Parsing URDF files", "correct": False}
      ],
    },
    {
      "question": "Which factory combination creates a drawer with a centered handle?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Drawer.create_with_new_body_in_world, Handle.create_with_new_body_in_world, Drawer.add_handle", "correct": True},
        {"answer": "World.create_new_drawer + World.create_new_handle + World.add_handle_to_drawer", "correct": False},
        {"answer": "URDFFactory only", "correct": False},
        {"answer": "RayTracerFactory", "correct": False}
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
