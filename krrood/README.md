# KRROOD
KRROOD stands for Knowlege Representation & Reasoning Through Object Oriented Design. This project aims to provide a
unified way to represent and reason about knowledge in Python using the principles of object oriented design.

This project also is consistent with many of the ideas mentioned in the book
[Knowledge Representation and Reasoning](https://www.oreilly.com/library/view/knowledge-representation-and/9781558609327/).
And of course the books about object oriented design.

## Installation

```bash
pip install krrood
```
If you want to use the visualization feature, you will also need to install [rustworkx_utils](https://github.com/AbdelrhmanBassiouny/rustworkx_utils).
```bash
pip install rustworkx_utils
```

## Entity Query Language (EQL)

EQL is a relational query language that is pythonic, and intuitive.

The interface side of EQL is inspired by [euROBIN](https://www.eurobin-project.eu/) entity query language white paper.

## Ormatic

## PyCharm plugin

**Python Role Lens**, a PyCharm/IntelliJ plugin that teaches the IDE about krrood's `Role`
delegation pattern (completion, type inference, and Go-to-Declaration on delegated attributes),
lives at [`plugins/pycharm/pyroles-pycharm/`](plugins/pycharm/pyroles-pycharm/). It is a
self-contained Gradle project, built and published to the JetBrains Marketplace independently of
the Python workspace.

## Documentation

Read the documentation [here](https://cram2.github.io/cognitive_robot_abstract_machine/krrood/intro.html).
