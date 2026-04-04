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

(creating-custom-bodies-quiz)=
# Creating Custom Bodies Quiz

This page provides a self-check quiz for the tutorial: [](creating-custom-bodies).  
Source: [Jupyter quiz](https://pypi.org/project/jupyterquiz/#description). $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "What is the purpose of the `PrefixedName` data structure?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "To give world entities human readable names.", "correct": True, "feedback": "Correct! PrefixedNames are easy to read but not necessarily unique." },
        { "answer": "To act as unique variable type.", "correct": False, "feedback": "Incorrect! Try again." },
        { "answer": "To identify the world a world entity belongs to.", "correct": False, "feedback": "Incorrect! Try again." },
        { "answer": "To manage ray-tracing parameters.", "correct": False, "feedback": "Incorrect! Try again."   }
      ],
    },
    {
      "question": "Which two shape collections can a body have?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "Visual and collision", "correct": True, "feedback": "Correct! If a body doesn't have a visual, then the collision shape is used instead."},
        { "answer": "Static and dynamic", "correct": False, "feedback": "Incorrect! Try again."},
        { "answer": "Physical and logical", "correct": False, "feedback": "Incorrect! Try again." },
        { "answer": "Primary and secondary", "correct": False, "feedback": "Incorrect! Try again." }
      ],
    },
    {
      "question": "Which shapes are supported?",
      "type": "many_choice",
      "answer_cols": 4,
      "answers": [
        { "answer": "Box", "correct": True },
        { "answer": "Sphere", "correct": True },
        { "answer": "Cylinder", "correct": True },
        { "answer": "Mesh", "correct": True },
        { "answer": "Cone", "correct": False }
      ],
    },
    {
      "question": "How should you add a body to the world?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "with world.modify_world(): \n world.add_body(body)", "correct": True, "feedback": "Correct! The world must be modified within a `with` block. To know why, check out own cross-process-synchronization tutorial (TBD)" },
        { "answer": "world.add_body(body)", "correct": False, "feedback": "Incorrect! Try again." },
        { "answer": "world.create_body(body)", "correct": False, "feedback": "Incorrect! Try again." },
        { "answer": "world.append(body)", "correct": False, "feedback": "Incorrect! Try again." },
      ],
    },
    {
      "question": "What issue will occur when creating multiple unconnected bodies?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "World validation fails.", "correct": True, "feedback": "Correct! Since two unconnected bodies would mean an ambiguous root in the world-structure, validation fails." },
        { "answer": "Rendering always crashes.", "correct": False, "feedback": "Incorrect! Try again." },
        { "answer": "Meshes are auto-merged.", "correct": False, "feedback": "Incorrect! Try again." },
        { "answer": "Textures are dropped.", "correct": False, "feedback": "Incorrect! Try again." }
      ],
    },
    {
      "question": "What does `get_semantic_world_directory_root()` help with?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "Locating the Semantic World project root.", "correct": True, "feedback": "Correct! This makes it easy to reference paths from anywhere in the project." },
        { "answer": "Gives you permissions to push onto the Semantic World cram2/main branch without doing Pull Requests.", "correct": False, "feedback": "YOU MONSTER! SHAME. ON. YOU. >:C" },
        { "answer": "Gives you read and write access to a world.", "correct": False, "feedback": "Incorrect! Try again." },
        { "answer": "Get the root of your world.", "correct": False, "feedback": "Incorrect! Try again." }
      ],
    },
    {
      "question": "What does a `Box` take for its `scale` parameter?",
      "type": "multiple_choice",
      "answers": [
        { "answer": "Scale(x=x, y=y, z=z)", "correct": True, "feedback": "Correct!" },
        { "answer": "Tuple[x, y, z]", "correct": False, "feedback": "Incorrect! Please refer to https://testing.googleblog.com/2017/11/obsessed-with-primitives.html" },
        { "answer": "List[x, y, z].", "correct": False, "feedback": "Incorrect! Please refer to https://testing.googleblog.com/2017/11/obsessed-with-primitives.html" },
        { "answer": "None of the answers listed, its individual x_scale, y_scale and z_scale parameters", "correct": False, "feedback": "Incorrect! Please refer to https://testing.googleblog.com/2017/11/obsessed-with-primitives.html" }
      ],
    }
  ]

import json
json_str = json.dumps(questions)
json.loads(json_str) 

display_quiz(questions)
```