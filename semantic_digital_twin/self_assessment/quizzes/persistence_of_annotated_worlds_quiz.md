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

(persistence-of-annotated-worlds-quiz)=
# Persistence of Annotated Worlds Quiz

This page provides a self-check quiz for the tutorial: [](persistence-of-annotated-worlds).  
Source: Jupyter quiz. $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "What is the role of the ORM in the semantic digital twin?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Serialize and reconstruct worlds (including semantic annotations, robot limits, actuators, ...) to/from SQL", "correct": True},
        {"answer": "Render worlds in notebooks", "correct": False},
        {"answer": "Plan robot trajectories", "correct": False},
        {"answer": "Load URDF files", "correct": False}
      ],
    },
    {
      "question": "Which function converts a world to a data access object (DAO)?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "to_dao(world)", "correct": True},
        {"answer": "world.to_sql()", "correct": False},
        {"answer": "serialize(world)", "correct": False},
        {"answer": "dump_world(world)", "correct": False}
      ],
    },
    {
      "question": "Which library is used to interact with the SQL database?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "SQLAlchemy", "correct": True},
        {"answer": "pandas", "correct": False},
        {"answer": "sqlite3 module only", "correct": False},
        {"answer": "NetworkX", "correct": False}
      ],
    },
    {
      "question": "What happens to semantic annotations when persisting and reconstructing?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "They are stored and available after reconstruction", "correct": True},
        {"answer": "They are lost and must be reapplied", "correct": False},
        {"answer": "Only body geometry persists", "correct": False},
        {"answer": "Only connection DoFs persist", "correct": False}
      ],
    },
    {
      "question": "Where do you maintain which classes are mapped by the ORM?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "scripts/generate_orm.py", "correct": True},
        {"answer": "doc/_toc.yml", "correct": False},
        {"answer": "requirements.txt", "correct": False},
        {"answer": "World.__init__", "correct": False}
      ],
    }
]

import json
json_str = json.dumps(questions)
json.loads(json_str) 

display_quiz(questions)
```
