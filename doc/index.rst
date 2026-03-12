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

:ref:`ref-to-installation` | :ref:`ref-to-contributing` |
`Github <https://github.com/cram2/cognitive_robot_abstract_machine>`__

Architecture Overview
~~~~~~~~~~~~~~~~~~~~~

CRAM consists of the following sub-packages:

-  `PyCRAM <https://cram2.github.io/cognitive_robot_abstract_machine/pycram>`__:
   is the central control unit of the CRAM architecture. It
   interprets and executes high-level action plans using the CRAM plan
   language (CPL).

-  The `Semantic Digital Twin <https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin>`__
   is a world representation that integrates
   sensor data, robot models, and external knowledge to provide a
   comprehensive understanding of the robot's environment and tasks.

-  `Giskardpy <https://github.com/SemRoCo/giskardpy>`__ is a
   Python library for motion planning and control for robots. It uses
   constraint- and optimization-based task-space control to control the
   whole body of a robot.

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
   is a Python library for modeling and simulating random
   events.

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

Contributing
------------

Before committing any changes, please navigate into the project root and
install pre-commit hooks:

.. code:: bash

  sudo apt install pre-commit
  pre-commit install

If you have any questions or feedback, consider submitting a `GitHub
Issue <https://github.com/cram2/cognitive_robot_abstract_machine/issues>`__.

Code of Conduct
~~~~~~~~~~~~~~~

   Any code added to the repository must have at least an 85% test
   coverage.

🚀 How to Create a Pull Request (PR) in Our System

This guide outlines the best practices for creating a Pull Request in
our system to ensure high-quality, maintainable, and robust code.

1. 🤖 AI Code Review First (Pre-PR Check)

   Before submitting your PR, let AI review your code to catch common
   issues and suggest improvements.

   -  GitHub Copilot Reviewer: You can integrate GitHub Copilot as a
      reviewer directly into your PR process.

   -  PyCharm Integration: Alternatively, use AI features directly
      within PyCharm for an immediate local review.

2. 🛡️ Embrace Test-Driven Thinking

   Our process is Test-Driven Development (TDD):

   -  Bug Fixes: If you find a bug, your first step is to write a test
      that fails (reproduces the bug). Push this failing test, then
      implement the fix, and ensure the test now passes.

   -  The Beyoncé Rule: “If you like it, you should put a test on it.”
      Every new feature or piece of logic needs corresponding tests.

3. 🎯 Focus on Method Quality and Complexity

   -  Method Length/Complexity: Keep your methods concise and focused.
      If the cyclomatic complexity (which you can check using plugins
      like Code Complexity for JetBrains) reaches the hundreds, your
      method is highly likely to need refactoring/improvement.

   -  Helper Methods: Extract duplicate code into helper methods to
      adhere to the Don’t Repeat Yourself (DRY) principle.

   -  Modularity/Plugin Thinking: Think modularly. Code should be
      designed with a plugin-like approach, making components easily
      interchangeable or extendable.

   -  Side Effects & Entanglement: Methods should be decoupled (not
      entangled) and should not have hidden side effects. They should
      ideally do one thing and do it well, following the Single
      Responsibility Principle (SRP) from SOLID. Example:

      .. code:: python

          counter = 0 # outer scope state

          def increment_counter():
              global counter
              counter += 1 # side effect: modifies outer scope state
              print(f"Counter is now {counter}") # side effect: I/O

4. 📐 Adhere to Code Style and Principles

   -  Code Formatting: We use Black, which is fully PEP 8 compliant. All
      code must be automatically formatted with Black before submission.

   -  SOLID Principles: Read and understand the SOLID principles for
      writing robust, maintainable, and scalable software. This article
      is a great resource:
      https://realpython.com/solid-principles-python/

5. ✍️ Naming, Typing, and Imports

   -  Descriptive Naming: Choose descriptive names for variables,
      functions, and classes. Names should clearly communicate intent.

   -  Correct Typing: Use correct type hints consistently throughout
      your code.

   Import Strategy:

   -  Use absolute imports always within the package as this is easier
      to maintain and clearer to read and understand.

   -  Use relative imports always in tests when importing modules
      defined in the same test folder/package.

   -  When importing types, use typing extensions instead of typing or
      the standard library types;

   -  Avoid importing types directly from modules if you don’t need to
      construct an instance of the type inside the module. Annotations
      can be imported with the TYPE_CHECKING guard.

6. 📝 Documentation and Comments

   -  Non-Trivial Code: Document everything that is not trivial to
      understand.

   -  Avoid Over-Swaffling: Be concise. Do not use unnecessary, verbose
      explanations.

   -  Inline Comments: Use inline comments sparingly, primarily for
      explaining complex logic or a long block of code.

7. 🔁 Final Review and Responsibility

   -  Human Review: Always perform a final human review of your own code
      before submitting the PR. Read it line-by-line as if you were the
      reviewer.

   -  “If you break it, you fix it” Rule: You are the primary owner and
      person responsible for the code you introduce. If a bug is found
      in your changes, you must prioritize its fix.

   -  The “One Change, One Commit” Rule: Each commit should be a
      logical, atomic unit of work.

8. Post-Submission and Review 🔎

   After you open your Pull Request (PR), it enters the review stage.
   This is a critical step for ensuring code quality and collaboration.

   Responding to Feedback:

   -  Mindset is Key: Remember that code reviews are about the code, not
      you. Feedback is given to improve the project’s quality and help
      you learn. Take all comments professionally and constructively.

   -  Addressing the Feedback: When a reviewer requests changes, you
      don’t need to close the PR and start over! Simply make the
      required modifications in your local working directory.

   -  Commit and Push: Once the changes are made, create a new commit
      and push it to the same feature branch you used for the PR. The PR
      will automatically update with your new commits.

      .. code:: bash

        # 1. Make the changes locally...
        git add .
        git commit -m "Address review feedback on component X"
        git push origin <your-feature-branch-name>

PR Checklist Summary

::

   [ ] AI (Copilot/PyCharm) has reviewed the code.

   [ ] Black has formatted the code (PEP 8 compliant).

   [ ] New features/logic have tests (Beyoncé Rule).

   [ ] Bug fixes include a test that reproduced the bug.

   [ ] Methods are concise and low in complexity.

   [ ] Descriptive names and correct type hints are used.

   [ ] Relative importing is used correctly.

   [ ] Non-trivial code is documented concisely.

   [ ] Code is modular, decoupled, and adheres to SOLID principles.

   [ ] Final personal human review complete.

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
