Cognitive Robot Abstract Machine (CRAM)
=======================================

Monorepo for the CRAM cognitive architecture.

A hybrid cognitive architecture enabling robots to accomplish everyday
manipulation tasks.

This documentation serves as a central hub for all sub-packages within
the CRAM ecosystem.

About CRAM
----------

The Cognitive Robot Abstract Machine (CRAM) ecosystem is a comprehensive
cognitive architecture for autonomous robots, organized as a monorepo of
interconnected components. Together, they form a pipeline from abstract
task descriptions to physically executable actions, bridging the gap
between high-level intentions and low-level robot control.

:ref:`ref-to-installation` | :ref:`Contribute<contribute>` |
`Github <https://github.com/cram2/cognitive_robot_abstract_machine>`__

Architecture Overview
~~~~~~~~~~~~~~~~~~~~~

CRAM consists of the following sub-packages:

-  `PyCRAM <https://cram2.github.io/cognitive_robot_abstract_machine/pycram>`__:
   is the central control unit of the CRAM architecture. It
   interprets and executes high-level action plans using the
   CRAM plan language (CPL).

-  The `Semantic Digital Twin <https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin>`__
   is a world representation that integrates sensor data,
   robot models, and external knowledge in form of semantic
   annotations to provide a comprehensive understanding of
   the robot's environment and tasks.

-  `Giskardpy <https://github.com/SemRoCo/giskardpy>`__ is a
   Python library for motion control for robots. It uses constraint-
   and optimization-based task-space control to control the whole
   body of a robot.

-  `KRROOD <(https://cram2.github.io/cognitive_robot_abstract_machine/krrood)>`__
   is a Python framework that integrates symbolic knowledge
   representation, powerful querying, and rule-based reasoning through
   intuitive, object-oriented abstractions.

-  `Probabilistic
   Model <(https://cram2.github.io/cognitive_robot_abstract_machine/probabilistic_model)>`__
   is a Python library that offers a clean and
   unified API for probabilistic models, similar to scikit-learn for
   classical machine learning.

-  `Random
   Events <https://cram2.github.io/cognitive_robot_abstract_machine/random_events>`__
   is a Python library to provide a simple and flexible way to generate events that are suitable for probabilistic reasoning.

.. mermaid:: img/architecture_diagram.mmd
    :caption: Architecture Diagram

.. _ref-to-installation:

Installation
------------

To install the CRAM architecture, follow these steps:

Set up the Python virtual environment:

.. code:: bash

  sudo apt install -y virtualenv virtualenvwrapper && \
  grep -qxF 'export WORKON_HOME=$HOME/.virtualenvs' ~/.bashrc || echo 'export WORKON_HOME=$HOME/.virtualenvs' >> ~/.bashrc && \
  grep -qxF 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' ~/.bashrc || echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' >> ~/.bashrc && \
  grep -qxF 'source /usr/share/virtualenvwrapper/virtualenvwrapper.sh' ~/.bashrc || echo 'source /usr/share/virtualenvwrapper/virtualenvwrapper.sh' >> ~/.bashrc && \
  source ~/.bashrc && \
  mkvirtualenv cram-env

Activate / deactivate

..code:: bash

  workon cram-env
  deactivate

Pull the submodules:

.. code:: bash

  cd cognitive_robot_abstract_machine
  git submodule update --init --recursive

Install using UV
~~~~~~~~~~~~~~~~

To install the whole repo we use uv (https://github.com/astral-sh/uv),
first to install uv:

.. code:: bash

  # On macOS and Linux.
  curl -LsSf https://astral.sh/uv/install.sh | sh

then install packages:

.. code:: bash

  uv sync --active

Alternative: Poetry
~~~~~~~~~~~~~~~~~~~

Alternatively you can use poetry to install all packages in the
repository.

Install poetry if you haven't already:

.. code:: bash

  pip install poetry

Install the CRAM package along with its dependencies:

.. code:: bash

  poetry install

.. _ref-to-contributing:

Contribute
------------

Before contributing please check our guidelines on how to :ref:`contribute to CRAM<contribute>`.

About the AICOR Institute for Artificial Intelligence
-----------------------------------------------------

The AICOR Institute for Artificial Intelligence researches how robots
can understand and perform everyday tasks using fundamental cognitive
abilities – essentially teaching robots to think and act in practical,
real-world situations.

The institute is headed by Prof. Michael Beetz, and is based at the
`University of Bremen <https://www.uni-bremen.de/en/>`__, where is is
affiliated with the `Center for Computing and Communication Technologies
(TZI) <https://www.uni-bremen.de/tzi/>`__ and the high-profile area
`Minds, Media and Machines
(MMM) <https://minds-media-machines.de/en/>`__.

Beyond Bremen, AICOR is also part of several research networks:

-  `Robotics Institute Germany
   (RIG) <https://robotics-institute-germany.de/>`__ – a national
   robotics research initiative
-  `euROBIN <https://www.eurobin-project.eu/>`__ – a European network
   focused on advancing robot learning and intelligence

`Website <https://ai.uni-bremen.de/>`__ \|
`Github <https://github.com/code-iai>`__

.. _research--publications:

Research & Publications
-----------------------

| [1] A. Bassiouny et al., “Implementing Knowledge Representation and
  Reasoning with Object Oriented Design,” Jan. 21, 2026, arXiv: arXiv:2601.14840.
  doi: 10.48550/arXiv.2601.14840.
| [2] M. Beetz, G. Kazhoyan, and D. Vernon, “The CRAM Cognitive
  Architecture for Robot Manipulation in Everyday Activities,” p. 20,
  2021.
| [3] M. Beetz, G. Kazhoyan, and D. Vernon, “Robot manipulation in
  everyday activities with the CRAM 2.0 cognitive architecture and
  generalized action plans,” Cognitive Systems Research, vol. 92, p.
  101375, Sep. 2025, doi: 10.1016/j.cogsys.2025.101375.
| [4] J. Dech, A. Bassiouny, T. Schierenbeck, V. Hassouna, L. Krohm, and
  D. Prüsser, PyCRAM: A Python framework for cognition-enbabled robtics.
  (2025). [Online]. Available: https://github.com/cram2/pycram
| [5] T. Schierenbeck, probabilistic_model: A Python package for
  probabilistic models. (Jul. 01, ). [Online]. Available:
  https://github.com/tomsch420/probabilistic_model
| [6] T. Schierenbeck, Random-Events. (Apr. 01, 2002). [Online].
  Available: https://github.com/tomsch420/random-events
| [7] S. Stelter, “A Robot-Agnostic Kinematic Control Framework: Task
  Composition via Motion Statecharts and Linear Model Predictive
  Control,” Universität Bremen, 2025. doi: 10.26092/ELIB/3743.

Acknowledgements
----------------

This work has been partially supported by the German Research Foundation
DFG, as part of Collaborative Research Center (Sonderforschungsbereich)
1320 Project-ID 329551904 "EASE - Everyday Activity Science and
Engineering", University of Bremen (http://www.ease-crc.org/).
