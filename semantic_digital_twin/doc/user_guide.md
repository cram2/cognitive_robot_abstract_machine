(user-guide)=
# User Guide

## Prerequisites
You’ll need to know a bit of Python. For a refresher, see the [Python tutorial](https://docs.python.org/3/tutorial/).

To work with the examples, you'll additionally need ROS2 installed.

The examples are meant to be run from a clone of this repository.
If you encounter paths not working in your environment, 
make sure to use paths with respect to your custom setup.

If you want to convert the examples to Jupyter notebooks, you can use
```bash
pip install jupytext
cd semantic_digital_twin/examples
jupytext --to notebook *.md
```

## Learning Objectives

The user guide is divided into multiple chapters teaching you the following topics:

- Fundamental topics
  - [](using-transformations)
  - [](creating-custom-bodies)
  - [](loading-worlds)
  - [](visualizing-worlds)
  - [](world-structure-manipulation)
  - [](semantic_annotations)
- Advanced Topics
  - [](world-state-manipulation)
  - [](regions)
  - [](semantic_annotation_factories)
  - [](persistence-of-annotated-worlds)
  - Synchronizing worlds across multiple processes
  - Pipelines
  - Inverse Kinematics
  - Collision Checking
  - Casadi


