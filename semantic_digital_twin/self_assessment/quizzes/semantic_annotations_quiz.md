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

(semantic-annotations-quiz)=
# Semantic Annotations Quiz

This page provides a self-check quiz for the tutorial: [](semantic_annotations).  
Source: Jupyter quiz. $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "What is a SemanticAnnotation in the semantic world?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "A semantic annotation attached to world entities", "correct": True},
        {"answer": "A mesh loader for STL files", "correct": False},
        {"answer": "A plotting backend", "correct": False},
        {"answer": "A physics engine", "correct": False}
      ],
    },
    {
      "question": "How does a SemanticAnnotation set a default name?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "In __post_init__, if name is None it generates a name from the class name and an incrementing number", "correct": True},
        {"answer": "It auto-generates a UUID", "correct": False},
        {"answer": "It inherits the world's name", "correct": False},
        {"answer": "It uses getattr to fetch a label", "correct": False}
      ],
    },
    {
      "question": "Which library is used to query for SemanticAnnotations like apples?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Entity Query Language (EQL)", "correct": True},
        {"answer": "SQLAlchemy", "correct": False},
        {"answer": "NetworkX", "correct": False},
        {"answer": "NumPy", "correct": False}
      ],
    },
    {
      "question": "What does the FruitBox SemanticAnnotation group together in the example?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "A Container and a list of Apple SemanticAnnotations", "correct": True},
        {"answer": "Two Body instances into a single mesh", "correct": False},
        {"answer": "A URDF file and a parser", "correct": False},
        {"answer": "A World and a RayTracer", "correct": False}
      ],
    },
    {
      "question": "Which factory is used to create a hollow container in the tutorial?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "HasCaseAsRootBody provides a classmethod as a factory", "correct": True},
        {"answer": "DrawerFactory", "correct": False},
        {"answer": "HandleFactory", "correct": False},
        {"answer": "MeshFactory", "correct": False}
      ],
    }
]

import json
json_str = json.dumps(questions)
json.loads(json_str) 

display_quiz(questions)
```
