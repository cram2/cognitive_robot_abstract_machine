from __future__ import annotations

from dataclasses import dataclass, field

from krrood_spatial_math.cartesian_frame_3d import ValueT
from krrood_spatial_math.exceptions import ReferenceFrameMismatchError
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix

from .frames import NamedFrame

# %% frames fixed on another frame


@dataclass(eq=False)
class MountedFrame(NamedFrame):
    """
    A Cartesian frame fixed on another frame, so it expresses values given in itself or
    in that frame.
    """

    mount_T_self: HomogeneousTransformationMatrix = field(kw_only=True)
    """
    The change from this frame to the frame it is fixed on, which is its reference
    frame; its child frame becomes this frame.
    """

    def __post_init__(self):
        self.mount_T_self = self.mount_T_self.copy_with_new_reference_frames(
            self.mount_T_self.reference_frame, self
        )

    def express(self, value: ValueT) -> ValueT:
        if value.reference_frame is self:
            return value
        if value.reference_frame is not self.mount_T_self.reference_frame:
            raise ReferenceFrameMismatchError(
                expected_frame=self.mount_T_self.reference_frame,
                actual_frame=value.reference_frame,
            )
        return self.mount_T_self.inverse() @ value
