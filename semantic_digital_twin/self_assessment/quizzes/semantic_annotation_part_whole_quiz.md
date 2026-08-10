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

(semantic-annotation-part-whole-quiz)=
# Part-Whole Relationships Quiz

This page provides a self-check quiz for the tutorial: [](semantic_annotation_part_whole).  
Source: Jupyter quiz. $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "What makes a dataclass field a part-whole relationship field?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "The IsPartWholeRelationship marker in the field's metadata", "correct": True},
        {"answer": "Its position in the class hierarchy", "correct": False},
        {"answer": "A name ending in _part", "correct": False},
        {"answer": "Being annotated with a semantic annotation type", "correct": False}
      ],
    },
    {
      "question": "Your custom annotation needs a handle part. How should it get its handle field?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "By inheriting the HasHandle mixin, which brings the field and its routing", "correct": True},
        {"answer": "By declaring a handle field itself with IsPartWholeRelationship metadata", "correct": False},
        {"answer": "By calling add(handle); the field is created on demand", "correct": False},
        {"answer": "By storing the handle in a plain attribute after spawning it", "correct": False}
      ],
    },
    {
      "question": "When is declaring your own field with IsPartWholeRelationship metadata appropriate?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Only for a part kind that no built-in mixin covers", "correct": True},
        {"answer": "Whenever you want to rename an inherited part field", "correct": False},
        {"answer": "Always; mixins are only a convenience for the library itself", "correct": False},
        {"answer": "Never; part-whole fields are reserved for built-in annotations", "correct": False}
      ],
    },
    {
      "question": "How does whole.add(part) decide where to store the part?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "It matches type(part) against the element types of the whole's part-whole relationship fields", "correct": True},
        {"answer": "It stores the part in the first empty field it finds", "correct": False},
        {"answer": "It requires the field name as a second argument", "correct": False},
        {"answer": "It matches the part's name against the field names", "correct": False}
      ],
    },
    {
      "question": "Besides storing the part in a field, what else does add do?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "It lets the part mount itself into the kinematic structure, so the part moves with the whole", "correct": True},
        {"answer": "Nothing; the kinematic structure must be wired manually", "correct": False},
        {"answer": "It merges the part's geometry into the whole's collision shapes", "correct": False},
        {"answer": "It removes the part from the world's annotation registry", "correct": False}
      ],
    },
    {
      "question": "What happens to a list-valued part-whole field like dresser.drawers when a drawer is added?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "The drawer is appended to the list", "correct": True},
        {"answer": "The list is replaced by the new drawer", "correct": False},
        {"answer": "Adding fails because list fields are read-only", "correct": False},
        {"answer": "The drawer is stored under its name as a dictionary key", "correct": False}
      ],
    },
    {
      "question": "How does a part-whole relationship differ from an object standing on a supporting surface?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Part-whole is parthood (a handle is part of a drawer); occupancy is handled separately, e.g. by IsStorageSpace.add_object", "correct": True},
        {"answer": "They are the same relation with different method names", "correct": False},
        {"answer": "Part-whole applies only to regions, occupancy only to bodies", "correct": False},
        {"answer": "Occupancy implies parthood, but not the other way around", "correct": False}
      ],
    },
    {
      "question": "What happens when a mechanical joint part (e.g. a Slider) is mounted on a Drawer?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "It is stored on the drawer's mechanical_joint field and carries the drawer's motion", "correct": True},
        {"answer": "It is stored in the drawer's handle field", "correct": False},
        {"answer": "It replaces the drawer's root body", "correct": False},
        {"answer": "It is rejected because joints are not parts", "correct": False}
      ],
    }
]

import json
json_str = json.dumps(questions)
json.loads(json_str) 

display_quiz(questions)
```
