# krrood_synchronization

Streams of messages that keep copies of a model in different processes in step. It does
not depend on ROS or on any model in particular.

- `krrood_synchronization.messages`: `Sender`, the node of a process that sends messages;
  `MessageOrigin`, a sender together with the model whose changes it publishes;
  `StreamPosition`, where a message sits in the stream of its publisher; and `Message`,
  the base of every message of a stream.
- `krrood_synchronization.stream_participants`: `StreamPublisher`, which numbers the
  messages it publishes, and `StreamReceiver`, which remembers per publisher how far it
  received and applied that stream, which messages were lost and which message it could
  not apply.
- `krrood_synchronization.exceptions`: `DivergedFromPublisherError`, raised for a message
  that could not be applied.
