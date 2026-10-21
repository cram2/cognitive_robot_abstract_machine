"""
Specifications of semantic annotations together with the entities they are rooted in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Type

from typing_extensions import TypeVar

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from semantic_digital_twin.exceptions import (
    PartWholeCardinalityError,
    PartWholeFieldInAnnotationKwargs,
    UnknownPartWholeRelationshipField,
)
from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.specifications.base import SpawnSpecification
from semantic_digital_twin.specifications.connections import (
    FixedConnectionSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    KinematicStructureEntitySpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import (
        HasRootKinematicStructureEntity,
        PartWholeRelationship,
    )

# %% specification type parameters
TSemanticAnnotation = TypeVar(
    "TSemanticAnnotation", bound="HasRootKinematicStructureEntity"
)


# %% semantic annotation specifications


@dataclass
class PartSpecificationBinding:
    """
    Nested annotation parts together with the part-whole relationship field they fill.

    Parts are held in a list of bindings rather than keyed by field name in a mapping,
    so that the nesting is part of the persisted specification instead of being dropped
    on the way to the database.
    """

    field_name: str
    """
    The name of the part-whole relationship field the parts are mounted onto.
    """

    specifications: list[SemanticAnnotationWithRootSpecification] = field(
        default_factory=list
    )
    """
    The part specifications spawned and mounted onto the field.
    """


@dataclass
class SemanticAnnotationWithRootSpecification(SpawnSpecification[TSemanticAnnotation]):
    """
    World-independent description of a semantic annotation rooted in a single kinematic
    structure entity.

    The annotation's root entity is what attaches to the parent, so its parent
    connection lives on ``root_specification.connection_specification`` and nowhere
    else. Leaving it unset fixes the root to its parent.

    ..note:: There is deliberately no way to pass loose connection parameters here. Each
        connection family carries exactly the parameters it uses, so an inapplicable
        parameter is a construction error rather than a silently ignored field.
    """

    semantic_annotation_type: Type[TSemanticAnnotation]
    """
    The type of the semantic annotation that is a subclass of
    HasRootKinematicStructureEntity.
    """

    root_specification: KinematicStructureEntitySpecification
    """
    The specification of the root kinematic structure entity of the annotation.

    Its :attr:`connection_specification` is the annotation's parent connection.
    """

    annotation_kwargs: dict[str, Any] = field(default_factory=dict)
    """
    Inert keyword arguments passed straight to the annotation constructor, keyed by
    constructor field name.

    Nested annotation parts do not belong here; use :attr:`part_bindings`.

    .. note:: These values are of arbitrary type and are therefore not persisted with the
        specification.
    """

    part_bindings: list[PartSpecificationBinding] = field(default_factory=list)
    """
    Nested annotation parts, each bound to the part-whole relationship field it fills.

    Each part is spawned during :meth:`spawn` and mounted via the annotation's
    :meth:`PartWholeRelationship.add`.
    """

    def __post_init__(self):
        """
        Validate the annotation kwargs and part bindings so misuse fails fast, before
        any world mutation.
        """
        self._validate_annotation_kwargs()
        self._validate_part_bindings(self.semantic_annotation_type)

    def spawn(
        self,
        world: World,
        name: str | None = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> TSemanticAnnotation:
        """
        Materialize the annotation in ``world``: spawn its root entity, attach it to
        ``parent``, register the annotation, and spawn its geometry children and mounted
        part specifications.

        The root's connection is the root specification's
        :attr:`connection_specification`, falling back to a fixed connection when it is
        unset.

        :param world: The world the annotation, its root and its parts are added to.
        :param name: Overrides the specification's own name. If None, the spec's name is
            used.
        :param parent: The entity to attach the root to. If None, ``world.root`` is
            used.
        :param parent_T_self: Overrides the root specification's stored default pose.
        :return: The materialized semantic annotation.
        """
        root_entity = self.root_specification.to_domain_object(name or self.name)

        instance = self.semantic_annotation_type(
            name=self._resolved_name(name), root=root_entity, **self.annotation_kwargs
        )

        connection_specification = (
            self.root_specification.connection_specification
            or FixedConnectionSpecification()
        )

        with world.modify_world():
            self.root_specification.attach_and_spawn_children(
                world, root_entity, connection_specification, parent, parent_T_self
            )
            world.add_semantic_annotation(instance)
            self._mount_part_specifications(world, instance, root_entity)

        return instance

    def _validate_annotation_kwargs(self) -> None:
        """
        Validate that :attr:`annotation_kwargs` carries no part-whole relationship
        field.

        Such fields must be supplied via :attr:`part_bindings` so they are spawned and
        mounted.

        :raises PartWholeFieldInAnnotationKwargs: If a key names a part-whole
            relationship field.
        """
        part_whole_field_names = self._part_whole_fields_by_name()
        misplaced_field_names = [
            field_name
            for field_name in self.annotation_kwargs
            if field_name in part_whole_field_names
        ]
        if misplaced_field_names:
            raise PartWholeFieldInAnnotationKwargs(
                annotation_type_name=self.semantic_annotation_type.__name__,
                field_names=misplaced_field_names,
            )

    def _validate_part_bindings(
        self, annotation_type: type[TSemanticAnnotation]
    ) -> None:
        """
        Validate that every binding targets a part-whole relationship field of the
        annotation and that only to-many fields are given more than one part.

        :param annotation_type: The annotation type whose part-whole fields are
            validated against.
        :raises UnknownPartWholeRelationshipField: If a binding does not name a part-
            whole relationship field.
        :raises PartWholeCardinalityError: If several parts target a singular field.
        """
        part_whole_fields_by_name = self._part_whole_fields_by_name()
        for binding in self.part_bindings:
            wrapped_field = part_whole_fields_by_name.get(binding.field_name)
            if wrapped_field is None:
                raise UnknownPartWholeRelationshipField(
                    annotation=annotation_type,
                    field_name=binding.field_name,
                    available_fields=list(part_whole_fields_by_name),
                )
            if (
                len(binding.specifications) > 1
                and not wrapped_field.is_many_to_many_relationship
            ):
                raise PartWholeCardinalityError(
                    annotation_type_name=self.semantic_annotation_type.__name__,
                    field_name=binding.field_name,
                )

    def _part_whole_fields_by_name(self) -> dict[str, WrappedField]:
        """
        The annotation type's part-whole relationship fields, keyed by field name.

        :return: The wrapped part-whole relationship fields, keyed by field name.
        """
        return {
            wrapped_field.name: wrapped_field
            for wrapped_field in WrappedClass(
                self.semantic_annotation_type
            ).fields_with_metadata(IsPartWholeRelationship)
        }

    def _mount_part_specifications(
        self,
        world: World,
        instance: PartWholeRelationship,
        root_entity: KinematicStructureEntity,
    ) -> None:
        """
        Spawn each nested part and mount it onto ``instance`` via the part-whole
        :meth:`PartWholeRelationship.add`, into the field its binding names.

        .. note:: Assumes :meth:`_validate_part_bindings` has already run.

        :param world: The world the parts are added to.
        :param instance: The annotation the spawned parts are mounted onto.
        :param root_entity: The annotation's root, which the parts are attached to.
        """
        for binding in self.part_bindings:
            for part_specification in binding.specifications:
                part = part_specification.spawn(world, parent=root_entity)
                instance.add(part, field_name=binding.field_name)
