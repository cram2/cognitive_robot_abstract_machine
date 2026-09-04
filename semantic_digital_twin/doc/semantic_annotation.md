# Semantic Annotations

A semantic annotation ({py:class}`semantic_digital_twin.world_entity.semantic_annotation`) is a different representation for a part or a collection of parts in the world that has a semantic meaning and
functional purpose in specific contexts.

For example, a Drawer can be seen as a semantic annotation on a handle and a container that is connected via a fixed connection
and where the container has some prismatic connection.

Semantic annotations can be inferred automatically by specifying rules that make up a semantic annotation.

## How to use the semantic annotations

Semantic annotations and any other attribute of the world that can be inferred or should be inferred through reasoning can be used
through the world reasoner, you can check how to use the world reasoner [here](world_reasoner.md).

Some helper methods exist in the world reasoner just for the semantic annotations like {py:func}`semantic_digital_twin.reasoning.world_reasoner.WorldReasoner.infer_semantic_annotations`.

