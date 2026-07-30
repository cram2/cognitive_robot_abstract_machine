from krrood.class_diagrams.utils import get_type_hints_of_object

from ..dataset.annotation_only_referenced_type import TypeReferencedOnlyInAnnotations
from ..dataset.class_with_type_checking_only_annotation import (
    ClassWithTypeCheckingOnlyAnnotation,
)


def test_type_checking_only_field_annotation_resolves_to_referenced_type():
    """
    A field typed by a ``TYPE_CHECKING``-only import must still resolve to that exact
    type.

    ``ClassWithTypeCheckingOnlyAnnotation.annotation_only_field`` is annotated with
    :class:`TypeReferencedOnlyInAnnotations`, which is imported only inside a ``TYPE_CHECKING``
    block - the same annotation-only pattern that ``semantic_digital_twin`` and ``coraplex`` use for
    fields such as ``GraspDescription.end_effector``. The name is therefore absent from the module's
    runtime namespace, so a naive ``get_type_hints`` raises ``NameError`` and the resolver must
    recover the type from the module's type-checking imports instead of raising
    ``CouldNotResolveType``.
    """
    resolved_type_hints = get_type_hints_of_object(ClassWithTypeCheckingOnlyAnnotation)

    assert (
        resolved_type_hints["annotation_only_field"] is TypeReferencedOnlyInAnnotations
    )
