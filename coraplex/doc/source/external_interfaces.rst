==================
External Interfaces
==================

External interfaces connect CoraPlex to services that live outside of the belief state, such as perception pipelines or
third-party hardware APIs. They live in :mod:`coraplex.external_interfaces`.

RoboKudo
--------
:mod:`coraplex.external_interfaces.robokudo` is the interface to the `RoboKudo <https://robokudo.ai.uni-bremen.de/>`_
perception framework. It sends queries over a ROS action interface and returns the detected objects, for example via
``query_object`` for a specific object description, ``query_all_objects`` to detect everything in view, or
``query_human`` and related helpers for human perception. The interface is initialised lazily through the
``init_robokudo_interface`` decorator, which establishes the action client on first use.

Blum kitchen API
----------------
:mod:`coraplex.external_interfaces.blum_api` is the interface to a Blum smart-kitchen service. It authenticates against
the service and issues commands to motorised kitchen furniture, for example ``open_cabinet`` and ``close_cabinet``, and
reads the current ``kitchen_state``. As with RoboKudo, the connection is set up through the
``init_kitchen_interface`` decorator.
