"""
Annotator for writing a configured static camera transform into the CAS.
Use this in scenarios where you do not have a world to camera transform published
or when you develop the perception system in isolation.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from timeit import default_timer

from krrood.ormatic.utils import classproperty
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.datastructures.annotator_descriptor import AnnotatorDescriptor
from robokudo.datastructures.annotator_parameters import AnnotatorParameters
from robokudo.io.camera_interface import CameraInterface

# %% StaticCameraTransform parameters


@dataclass(kw_only=True)
class StaticCameraTransformParameters(AnnotatorParameters):
    """Settings specific to this annotation operation."""

    world_frame: str = "map"
    """Reference frame used for the camera-to-world transform."""

    world_T_camera: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix()
    )
    """Image-frame pose relative to the configured world frame."""


# %% StaticCameraTransform descriptor


@dataclass(kw_only=True)
class StaticCameraTransformDescriptor(
    AnnotatorDescriptor[StaticCameraTransformParameters]
):
    """Declarative configuration of :class:`StaticCameraTransformAnnotator`."""

    parameters: StaticCameraTransformParameters = field(
        default_factory=StaticCameraTransformParameters.from_defaults
    )
    """Effective settings initialized from application and annotator defaults."""


# %% StaticCameraTransform annotator


@dataclass(eq=False)
class StaticCameraTransformAnnotator(BaseAnnotator[StaticCameraTransformDescriptor]):
    """
    Write a fixed camera-to-world transform to the current CAS.
    """

    name: str = "StaticCameraTransform"
    """Name of this node in the behaviour tree."""

    descriptor: StaticCameraTransformDescriptor = field(
        default_factory=StaticCameraTransformDescriptor
    )
    """Configuration supplied to this annotator instance."""

    @classproperty
    def Descriptor(cls) -> type[StaticCameraTransformDescriptor]:
        """Specialize :meth:`robokudo.annotators.core.BaseAnnotator.Descriptor`."""
        return StaticCameraTransformDescriptor

    def update(self) -> Status:
        """
        Bind the configured world pose directly to the optical camera root.
        """
        start_timer = default_timer()

        cas = self.get_cas()
        parameters = self.descriptor.parameters
        observation = cas.require_camera_observation()

        world_T_camera = CameraInterface.bind_world_T_camera(
            world_frame=parameters.world_frame,
            camera_frame=str(observation.camera.root.name),
            world_T_camera=parameters.world_T_camera,
        )
        cas.camera_observation = observation.with_world_T_camera(world_T_camera)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
