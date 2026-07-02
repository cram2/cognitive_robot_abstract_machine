.. _developer-guide:

Developer Guide
===============

This guide explains how to effectively contribute to the semantic digital twin.
General code style related guidelines that can also be used for your AI assistant are found in the
`junie guidelines <https://github.com/cram2/cognitive_robot_abstract_machine/blob/main/AGENTS.md>`_.

Environment Setup
-----------------

Most developers of this project work from an Ubuntu 24.04 machine with ROS2 Jazzy installed.
Almost all development is done with Python 3.12 and the help of PyCharm.
We require `black <https://pypi.org/project/black/>`_ as a formatter for python.
While you are free to use whatever setup you want, I strongly recommend this as it ensures the availability of all features. 
Furthermore, the CI tests with this setup, and hence you may get unexpected results from the GitHub workflows.
If this still doesn't fit your style of work, you can also use the docker image for a fully functional setup.

For the management of the python packages, I strongly recommend `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_.

You can locally install the package with

.. code-block:: bash

   mkvirtualenv semantic_digital_twin --system-site-packages
   cd /path/to/semantic_digital_twin
   pip install -e .

Core Parts
----------

- :py:mod:`semantic_digital_twin.adapters`: Bridges between the World and external ecosystems and file formats. They import/export world data and keep it in sync with tools like ROS2, simulators, and mesh/robot formats.

- :py:mod:`semantic_digital_twin.world`: The central scene-graph/kinematic container you interact with. It owns bodies, connections, and degrees of freedom; validates and updates the kinematic structure; provides forward/inverse kinematics, collision-related utilities, callbacks for state/model changes, and orchestration of higher-level semantics like semantic annotations.

- :py:mod:`semantic_digital_twin.orm`: The database layer. It maps world entities and relationships to SQL databases via SQLAlchemy (auto-generated with `ormatic`), enabling serialization, persistence, and retrieval of complete worlds. It defines the SQL types and a thin interface to store/load worlds reproducibly. If you are unhappy with the storage and retrieval of data from databases you most likely have to change something here

- :py:mod:`semantic_digital_twin.spatial_types`: Numeric/symbolic geometric primitives and transformations used across the package (e.g., `TransformationMatrix`, points, vectors, expressions, derivatives). They support composing poses, doing kinematics, and differentiating expressions for solvers.

- :py:mod:`semantic_digital_twin.semantic_annotations`: Semantic abstractions built on top of raw bodies and connections. A `SemanticAnnotation` represents a higher-level concept (e.g., a drawer composed of a handle and a prismatic container) and can be inferred by rules. semantic annotations provide task-relevant groupings and behaviors without changing the underlying kinematic graph.
- :py:mod:`semantic_digital_twin.world_description`: The domain model for kinematic structure and geometry: `Body`, `Region`, `Connection` (fixed, passive/active joints, 6-DoF), `DegreeOfFreedom`, geometry/mesh types, state and modification objects. These classes define the structure and editable state that the `World` manages and reasoners and adapters consume.

Testing
-------

The tests are all contained in the test folder and further grouped by topics and written with `pytest <https://docs.pytest.org/en/7.1.x/index.html>`_.
The semantic digital twin contains a module :py:mod:`semantic_digital_twin.testing` for useful fixtures that shorten your boilerplate code.
We aim for a test coverage > 95% and generally test everything we write. 
Naturally, there are some exceptions, mainly including printing and visualization functionality.

The tests are executed with pytest.
You can execute them with

.. code-block:: bash

   cd /path/to/semantic_digital_twin
   pytest test

Documentation
-------------
The documentation is built with `jupyter book <https://jupyterbook.org/en/stable/intro.html>`_. 

You can build it locally using

.. code-block:: bash

   cd /path/to/semantic_digital_twin/doc
   jb build .


The docstrings are formatted using `ReStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_. 
We aim for documentation coverage of >95%.
When developing complete features, we want an example that explains the high-level usage showcasing the most important use-cases.
The examples should be a `jupyter notebook <https://jupyter.org/>`_. 
Do not commit the jupyter notebook directly to git as this will bloat up the commit and lead to unreadable diffs.
Convert the jupyter notebook to a `myst-notebook <https://jupyterbook.org/en/stable/file-types/myst-notebooks.html>`_, e.g.

.. code-block:: bash

   jupytext convert --to md your_example.ipynb

You can test the entirety of notebooks using treon with

.. code-block:: bash

   cd /path/to/semantic_digital_twin/scripts
   bash test_notebooks.sh

Contribution Guidelines
-----------------------
Contributions are exclusively done via GitHub pull requests.
PRs only get merged if:

- At least one reviewer, who is not the author, approves it
- There are no open discussions
- The CI is green

PyCharm Setup
-------------
If you are a fellow pycharm enjoyer, there are a couple of hints I want to hand down to you:

- If you want proper ROS2 support, you need to start PyCharm from a terminal that has ROS2 sourced.
- You can enable black as a default code formatter for your IDE
- You can enable ReStructuredText as a default docstring formatter for your IDE
- As a researcher/student/teacher you are eligible for GitHub Pro which has excellent integration with PyCharm, especially with Copilot.

Help, people are unhappy with my PR
-----------------------------------
We only accept clean code that does something useful and feels like it belongs inside the semantic digital twin.
If the reviews address the quality/cleanness of your code, here are resources to improve your python object-oriented programming skills:

- `SOLID Principles <https://realpython.com/solid-principles-python/>`_
- `OOP Design Patterns <https://www.youtube.com/playlist?list=PLlsmxlJgn1HJpa28yHzkBmUY-Ty71ZUGc>`_

If the conversations are not leading anywhere, 
consider writing a User story for your PR such that the reviewers are getting the full story of your contribution. 
`User Story Mapping <https://www.audible.de/pd/User-Story-Mapping-Hoerbuch/B08TZWYL85?overrideBaseCountry=true&bp_o=true&ef_id=Cj0KCQjwxL7GBhDXARIsAGOcmIMnBFcYFg9NbKtB6MCDhs_Z-Jp76hz8robGdm3LQq19mzjkQByUsJcaAtJ0EALw_wcB%3AG%3As&gclsrc=aw.ds&source_code=GAWPP30DTRIAL45305022590T4&ipRedirectOverride=true&gad_source=1&gad_campaignid=22540587480&gbraid=0AAAAADzxWuhMO1IbkLpihZf2FHbHB2mgj&gclid=Cj0KCQjwxL7GBhDXARIsAGOcmIMnBFcYFg9NbKtB6MCDhs_Z-Jp76hz8robGdm3LQq19mzjkQByUsJcaAtJ0EALw_wcB>`_ is a book you can use as a reference for writing user stories.