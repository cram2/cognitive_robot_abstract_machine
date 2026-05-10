from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from semantic_digital_twin.world_description.role_mixins.world_entity_role_mixins import (
    DelegatorForSemanticAnnotation,
)
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)
from typing import List, Optional, Type
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
        ProbabilisticCircuit,
    )
    from random_events.product_algebra import Event
    from semantic_digital_twin.semantic_annotations.mixins import (
        HasCaseAsRootBody,
        HasRootBody,
        HasRootKinematicStructureEntity,
        HasStorageSpace,
        HasSupportingSurface,
        TBody,
        THasRootBody,
        TKinematicStructureEntity,
    )
    from semantic_digital_twin.spatial_types.spatial_types import (
        HomogeneousTransformationMatrix,
        Point3,
    )
    from semantic_digital_twin.world_description.geometry import Scale
    from semantic_digital_twin.world_description.world_entity import (
        Body,
        KinematicStructureEntity,
        Region,
        SemanticAnnotation,
    )


@dataclass(eq=False)
class DelegatorForHasRootKinematicStructureEntity(DelegatorForSemanticAnnotation, ABC):
    @property
    @abstractmethod
    def delegatee(self) -> HasRootKinematicStructureEntity: ...
    @property
    def root(self) -> TKinematicStructureEntity:
        return self.delegatee.root

    @root.setter
    def root(self, value: TKinematicStructureEntity):
        self.delegatee.root = value

    @property
    def global_transform(self) -> HomogeneousTransformationMatrix:
        return self.delegatee.global_transform

    @property
    def min_max_points(self) -> tuple[Point3, Point3]:
        return self.delegatee.min_max_points

    @property
    def scale(self) -> Scale:
        return self.delegatee.scale

    def _attach_child_entity_in_kinematic_structure(
        self,
        child_kinematic_structure_entity: KinematicStructureEntity,
    ):
        return self.delegatee._attach_child_entity_in_kinematic_structure(
            child_kinematic_structure_entity
        )

    def _attach_parent_entity_in_kinematic_structure(
        self,
        new_parent_entity: KinematicStructureEntity,
    ):
        return self.delegatee._attach_parent_entity_in_kinematic_structure(
            new_parent_entity
        )

    def _offline_root_T_entity(
        self, entity: KinematicStructureEntity
    ) -> HomogeneousTransformationMatrix:
        return self.delegatee._offline_root_T_entity(entity)

    def get_new_grandparent(
        self,
        parent_kinematic_structure_entity: KinematicStructureEntity,
    ):
        return self.delegatee.get_new_grandparent(parent_kinematic_structure_entity)


@dataclass(eq=False)
class DelegatorForHasRootBody(DelegatorForHasRootKinematicStructureEntity, ABC):
    @property
    @abstractmethod
    def delegatee(self) -> THasRootBody: ...
    @property
    def root(self) -> TBody:
        return self.delegatee.root

    @root.setter
    def root(self, value: TBody):
        self.delegatee.root = value

    @property
    def bodies(self) -> list[Body]:
        return self.delegatee.bodies


@dataclass(eq=False)
class DelegatorForHasStorageSpace(DelegatorForHasRootBody, ABC):
    @property
    @abstractmethod
    def delegatee(self) -> HasStorageSpace: ...
    @property
    def objects(self) -> List[THasRootBody]:
        return self.delegatee.objects

    @objects.setter
    def objects(self, value: List[THasRootBody]):
        self.delegatee.objects = value

    @synchronized_attribute_modification
    def add_object(self, object: HasRootBody):
        return self.delegatee.add_object(object)

    def get_objects_of_type(
        self, object_type: Type[SemanticAnnotation]
    ) -> List[HasRootBody]:
        return self.delegatee.get_objects_of_type(object_type)


@dataclass(eq=False)
class DelegatorForHasSupportingSurface(DelegatorForHasStorageSpace, ABC):
    @property
    @abstractmethod
    def delegatee(self) -> HasSupportingSurface: ...
    @property
    def supporting_surface(self) -> Region:
        return self.delegatee.supporting_surface

    @supporting_surface.setter
    def supporting_surface(self, value: Region):
        self.delegatee.supporting_surface = value

    def _2d_gaussian_sampler_from_2d_sample_space(
        self,
        objects_of_interest: List[HasRootBody],
        variance: float,
        sample_space: Event,
    ) -> Optional[ProbabilisticCircuit]:
        return self.delegatee._2d_gaussian_sampler_from_2d_sample_space(
            objects_of_interest, variance, sample_space
        )

    def _2d_surface_sample_space_excluding_objects(self, object_bloat: float) -> Event:
        return self.delegatee._2d_surface_sample_space_excluding_objects(object_bloat)

    def _build_surface_sampler(
        self,
        category_of_interest: Optional[Type[SemanticAnnotation]] = None,
        object_bloat: float = 0.1,
    ):
        return self.delegatee._build_surface_sampler(category_of_interest, object_bloat)

    def _untruncated_2d_gaussian_sampler(
        self,
        objects_of_interest: List[HasRootBody],
        variance: float,
    ) -> ProbabilisticCircuit:
        return self.delegatee._untruncated_2d_gaussian_sampler(
            objects_of_interest, variance
        )

    @synchronized_attribute_modification
    def add_supporting_surface(self, region: Region):
        return self.delegatee.add_supporting_surface(region)

    def calculate_supporting_surface(
        self,
        upward_threshold: float = 0.95,
        clearance_threshold: float = 0.5,
        min_surface_area: float = 0.0225,  # 15cm x 15cm
    ) -> Optional[Region]:
        return self.delegatee.calculate_supporting_surface(
            upward_threshold, clearance_threshold, min_surface_area
        )

    def infer_objects_on_surface(self):
        return self.delegatee.infer_objects_on_surface()

    def sample_points_from_surface(
        self,
        body_to_sample_for: Optional[HasRootBody] = None,
        category_of_interest: Optional[Type[SemanticAnnotation]] = None,
        amount: int = 100,
    ) -> List[Point3]:
        return self.delegatee.sample_points_from_surface(
            body_to_sample_for, category_of_interest, amount
        )


@dataclass(eq=False)
class DelegatorForHasCaseAsRootBody(DelegatorForHasSupportingSurface, ABC):
    @property
    @abstractmethod
    def delegatee(self) -> HasCaseAsRootBody: ...
