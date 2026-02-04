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

(loading-worlds-quiz)=
# Loading Worlds Quiz

This page provides a self-check quiz for the tutorial: [](loading-worlds).  
Source: Jupyter quiz. $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "Which class is used to load a world from a URDF file?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "URDFParser", "correct": True},
        {"answer": "WorldLoader", "correct": False},
        {"answer": "RayTracer", "correct": False},
        {"answer": "SceneParser", "correct": False}
      ],
    },
    {
      "question": "Which helper locates packaged resources (e.g., URDFs) in the semantic digital twin?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "resource_filename('semantic_digital_twin')", "correct": True},
        {"answer": "find_repo_root('semantic_digital_twin')", "correct": False},
        {"answer": "locate_resources('semantic_digital_twin')", "correct": False},
        {"answer": "get_data_dir('semantic_digital_twin')", "correct": False}
      ],
    },
    {
      "question": "Loaded worlds from files typically contain which information?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Kinematic structure only without semantic annotations", "correct": True},
        {"answer": "Full semantic annotations", "correct": False},
        {"answer": "Robot motion plans", "correct": False},
        {"answer": "Texturing information only", "correct": False}
      ],
    },
    {
      "question": "Which file formats are mentioned as supported in the tutorial?",
      "type": "multiple_select",
      "answers": [
        {"answer": "URDF", "correct": True},
        {"answer": "MJCF", "correct": True},
        {"answer": "STL", "correct": True},
        {"answer": "FBX", "correct": False}
      ],
    },
    {
      "question": "What is the correct sequence to parse a URDF into a World?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "URDFParser.from_file(path).parse()", "correct": True},
        {"answer": "World.parse(path)", "correct": False},
        {"answer": "URDFParser(path).world()", "correct": False},
        {"answer": "parse_urdf_to_world(path)", "correct": False}
      ],
    }
]

import json
json_str = json.dumps(questions)
json.loads(json_str) 

display_quiz(questions)
```
