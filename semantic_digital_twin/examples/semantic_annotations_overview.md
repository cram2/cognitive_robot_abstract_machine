---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(semantic_annotations_overview)=
# Semantic Annotations

Semantic annotations attach meaning to the bodies and regions in a world: that a body interpreted as a
handle, or that a collection of bodies is interpreted as a double door. Because they are plain, typed Python
classes, you can reason over them with the Entity Query Language and persist them through the ORM.

This part of the guide collects the semantic-annotation tutorials:

- [](semantic_annotations) — what semantic annotations are and how to define your own.
- [](semantic_annotation_part_whole) — building annotations from structural parts via the
  part-whole relationship, using the unified `add` method and `part_whole_relationship_field`.
- [](semantic_annotation_factories) — convenience factories that spawn an annotation together with
  a body and generated geometry.
