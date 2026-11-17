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
- :mod:`semantic_digital_twin.world_description.world_modification`
- :mod:`krrood.message_stream.messages`
- :mod:`krrood.message_stream.stream_participants`

Classes:

- :class:`semantic_digital_twin.world.World`

How it works
------------
A :class:`~semantic_digital_twin.adapters.ros.world_synchronizer.WorldSynchronizer` publishes model and state changes
on one topic, so that a state change can never overtake the model change that introduced what it refers to.

The world state is synchronized whenever the state change callbacks of a :class:`~semantic_digital_twin.world.World`
are called, by publishing the free variables that changed since the last message.

The changes to the world model are a bit more complicated.
Every method decorated with :func:`~semantic_digital_twin.world.atomic_world_modification` records a
:class:`~semantic_digital_twin.world_description.world_modification.WorldModification`.
Atomic modifications are changes to the world that, if replayed, produce the same world structure.
They cannot be split further and hence must not call other atomic modifications.
The outermost :meth:`~semantic_digital_twin.world.World.modify_world` block collects them in a
:class:`~semantic_digital_twin.world_description.world_modification.WorldModelModificationBlock`; when it closes, the
model change callbacks are triggered, the block is published and the other subscribers replay it.

If you ever have the case that you make changes to a world that are not repeatable via this mechanism or just want every
process to load a new world, you can use the :class:`~semantic_digital_twin.adapters.ros.world_synchronizer.ModelReloadSynchronizer`
to force all worlds subscribed to it to do so.

Message streams
---------------
What does not depend on worlds lives in :mod:`krrood.message_stream`, so that other models can be synchronized the same way.
Every message carries its :class:`~krrood.message_stream.messages.MessageOrigin` and its position in the stream of that
publisher. A synchronizer is a :class:`~krrood.message_stream.stream_participants.StreamPublisher`, which numbers the
messages it sends, and a :class:`~krrood.message_stream.stream_participants.StreamReceiver`, which remembers per publisher
how far it received and applied that stream. With that, two processes can tell each other what to catch up with
(:meth:`~krrood.message_stream.stream_participants.StreamReceiver.has_applied`), a gap in a stream reveals lost messages,
and a message that could not be applied stops its publisher from counting as caught up with until
:meth:`~krrood.message_stream.stream_participants.StreamReceiver.forget_divergence_from` is called.

Expanding Modifications
-----------------------
If you want to expand the capability to communicate changes to the world's model via ROS2 topics, you have to check out the
:mod:`semantic_digital_twin.world_description.world_modification` module. In there you find different ways of communicating different changes to the
world via data structures. This is not trivial, since ROS topics cannot communicate data structures that have many-to-one
relationships easily. For instance, when a :class:`~semantic_digital_twin.world_description.world_entity.Body` is removed from the world, this must not be communicated by sending
the entire body data around. Instead, every process needs some way to identify this body in their memory and remove it.
Hence, the :class:`~semantic_digital_twin.world_description.world_modification.RemoveKinematicStructureEntityModification` just takes the id of the body and publishes a
call to remove the body with this id.

Why JSON?
---------
Due to the limited capabilities of ROS2 communication, it is not trivial to reflect the definitions and mechanisms of 
the classes of semantic digital twin in ROS2 messages. If you choose a dedicated message for each class, you get issues with
polymorphism, many-to-one references and back-references. Furthermore, maintaining the ROS2 messages when the
datastructures change is complicated. JSON provides an easy fix to some of these problems.

Finally, fully functional shipping of this package via PyPi is only possible if you don't need to build custom
ros messages.
