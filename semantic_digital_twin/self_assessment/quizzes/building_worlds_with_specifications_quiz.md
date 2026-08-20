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

(building-worlds-with-specifications-quiz)=
# Building Worlds with Specifications Quiz

This page provides a self-check quiz for the tutorial: [](building-worlds-with-specifications).  
Source: Jupyter quiz. $ $

% NOTE: The lone `$ $` above ensures some math is rendered before the quiz,
% which fixes a known math-rendering quirk inside the quiz widget.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz

questions = [
    {
      "question": "What distinguishes a specification from the entity it describes?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "It is a reusable, world-independent recipe that is materialized later", "correct": True},
        {"answer": "It is a lightweight proxy that mirrors the entity's live state", "correct": False},
        {"answer": "It is the serialized form of an entity that already exists in a world", "correct": False},
        {"answer": "It is a read-only view on the entity's geometry", "correct": False}
      ],
    },
    {
      "question": "What does spawn(world) do for a body specification?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Materializes the body, attaches it to a parent, and spawns its children in one modification block", "correct": True},
        {"answer": "Only creates the body; connections must be added manually afterwards", "correct": False},
        {"answer": "Registers the specification in the world for lazy construction", "correct": False},
        {"answer": "Returns a copy of the specification bound to the world", "correct": False}
      ],
    },
    {
      "question": "Which connection attaches a spawned entity when its specification's connection_specification is left unset?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "A FixedConnection", "correct": True},
        {"answer": "A Connection6DoF", "correct": False},
        {"answer": "A PrismaticConnection", "correct": False},
        {"answer": "No connection; the entity floats unattached", "correct": False}
      ],
    },
    {
      "question": "Why does connect require the child while spawn does not take one?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "A connection joins two pre-existing entities, so there is nothing to materialize", "correct": True},
        {"answer": "connect is a legacy method kept for backwards compatibility", "correct": False},
        {"answer": "The child is optional; the world root is used when omitted", "correct": False},
        {"answer": "spawn infers the child from the specification's name", "correct": False}
      ],
    },
    {
      "question": "How do get_annotation_specification and get_default_root_kinematic_structure_entity_specification work together?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "get_default_root_kinematic_structure_entity_specification builds the root geometry; get_annotation_specification wraps it into the annotation specification", "correct": True},
        {"answer": "get_annotation_specification builds the geometry; get_default_root_kinematic_structure_entity_specification names it", "correct": False},
        {"answer": "They are alternatives: each returns a complete annotation specification", "correct": False},
        {"answer": "get_default_root_kinematic_structure_entity_specification spawns the root; get_annotation_specification registers the annotation", "correct": False}
      ],
    },
    {
      "question": "When are the keys of part_specifications validated?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "At specification construction time, before any world is mutated", "correct": True},
        {"answer": "At spawn time, when the parts are mounted", "correct": False},
        {"answer": "Only when the world is saved or serialized", "correct": False},
        {"answer": "Never; unknown keys are silently ignored", "correct": False}
      ],
    },
    {
      "question": "What does WorldSpecification.to_domain_object return on repeated calls?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "A fresh, independent world each time; the stored environment is deep-copied", "correct": True},
        {"answer": "The same world instance, cached after the first call", "correct": False},
        {"answer": "A new world that shares bodies with the previous one", "correct": False},
        {"answer": "It fails on the second call because the environment was consumed", "correct": False}
      ],
    },
    {
      "question": "What does a RobotSpecification bundle?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "A robot's semantic annotation class together with its localization and start poses", "correct": True},
        {"answer": "A parsed robot world that is merged as-is into the environment", "correct": False},
        {"answer": "The robot's joint limits and controller configuration", "correct": False},
        {"answer": "The list of objects a robot is allowed to manipulate", "correct": False}
      ],
    },
    {
      "question": "Where does a robot end up in the world a WorldSpecification materializes?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Under world.root via an odom body, attached by the drive its mobile base declares", "correct": True},
        {"answer": "Directly at world.root, replacing the previous root body", "correct": False},
        {"answer": "As a detached branch that must be connected manually afterwards", "correct": False},
        {"answer": "Under the first object in the objects list", "correct": False}
      ],
    },
    {
      "question": "WorldSpecification.robots is a list rather than a single robot. What does that buy you?",
      "type": "multiple_choice",
      "answers": [
        {"answer": "Several robots can be described in one scene, each with its own localization", "correct": True},
        {"answer": "The same robot can be retried if its description fails to parse", "correct": False},
        {"answer": "The robots are spawned in parallel to speed up materialization", "correct": False},
        {"answer": "Nothing; the list may only ever hold one entry", "correct": False}
      ],
    }
]

import json
json_str = json.dumps(questions)
json.loads(json_str) 

display_quiz(questions)
```
