Synchronizing Worlds
====================

This document explains how worlds are synchronized across multiple instances, threads, or processes. 
It answers the question:

    How can I synchronize worlds across multiple processes when I do not have access to the other processes memory?

For all synchronizations, ROS2 topics are used to communicate changes in a peer-2-peer like network.
In the semantic digital twin package, the following classes and modules are needed to understand this document:

Modules:
- :mod:`semantic_digital_twin.adapters.ros.messages`
- :mod:`semantic_digital_twin.adapters.ros.world_synchronizer`
- :mod:`semantic_digital_twin.world_modification`

Classes:

- :class:`semantic_digital_twin.world.World`

How it works
------------
The world state is synchronized whenever the state_change_callbacks of a :class:`~semantic_digital_twin.world.World` are called
by publishing the changed free variables. The details are found in 
:class:`semantic_digital_twin.adapters.ros.world_synchronizer.StateSynchronizer`.

The changes to the world model are a bit more complicated.
Conceptually, every instance of the :class:`~semantic_digital_twin.world.World` keeps track of atomic modifications done to it in :class:`~semantic_digital_twin.world.World`._atomic_modifications.
Atomic modifications are changes to the world, that, if replayed, produce the same world structure.
Atomic modifications cannot be split further and hence must not call other atomic modifications.
When the model_change_callbacks are triggered, the latest changes to the world are published and repeated by the other
subscribers. The details are found in :class:`semantic_digital_twin.adapters.ros.world_synchronizer.ModelSynchronizer`.

If you ever have the case that you make changes to a world that are not repeatable via this mechanism or just want every
process to load a new world, you can use the :class:`semantic_digital_twin.adapters.ros.world_synchronizer.ModelReloadSynchronizer` to force all worlds subscribed to that to do so.
The details are found in :class:`semantic_digital_twin.adapters.ros.world_synchronizer.ModelReloadSynchronizer`.

Expanding Modifications
----------------------
If you want to expand the capability to communicate changes to the world's model via ROS2 topics, you have to check out the
:mod:`semantic_digital_twin.world_modification` module. In there you find different ways of communicating different changes to the
world via data structures. This is not trivial, since ROS topics cannot communicate data structures that have many-to-one
relationships easily. For instance, when a :class:`semantic_digital_twin.world_entity.Body` is removed from the world, this must not be communicated by sending
the entire body data around. Instead, every process needs some way to identify this body in their memory and remove it.
Hence, the :class:`semantic_digital_twin.world_modification.RemoveBodyModification` just takes the name of the body and publishes a
call to remove the body with this name.

Why JSON?
---------
Due to the limited capabilities of ROS2 communication, it is not trivial to reflect the definitions and mechanisms of 
the classes of semantic digital twin in ROS2 messages. If you choose a dedicated message for each class, you get issues with
polymorphism, many-to-one references and back-references. Furthermore, maintaining the ROS2 messages when the
datastructures change is complicated. JSON provides an easy fix to some of these problems.

Finally, fully functional shipping ofthis package via PyPi is only possible if you don't need to build custom
ros messages.
