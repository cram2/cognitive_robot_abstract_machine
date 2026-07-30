from dataclasses import dataclass


@dataclass
class LateBoundAnnotationSecondaryType:
    """
    A second field type imported only under ``TYPE_CHECKING`` by the same owner module.

    Paired with :class:`~test.krrood_test.dataset.latebound_annotation_type.LateBoundAnnotationType`
    to mimic a single module with more than one ``TYPE_CHECKING``-only annotation poisoned by the
    same circular-import event, such as ``world_entity.py``'s ``World`` and
    ``GenericSemanticAnnotation`` references.
    """

    label: str = ""
