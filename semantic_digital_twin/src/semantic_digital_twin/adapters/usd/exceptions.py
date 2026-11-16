from __future__ import annotations

from dataclasses import dataclass, field


from semantic_digital_twin.exceptions import ParsingError


@dataclass
class UnsupportedUsdPhysicsJointTypeError(ParsingError):
    """
    Raised when a stage contains a physics joint of a type this parser does not build a
    Connection for.

    Silently skipping it would build a World missing whatever link that joint's body1 is
    the only connection to.
    """

    joint_path: str = field(kw_only=True)
    """
    The unsupported joint prim's stage path.
    """

    joint_type: str = field(kw_only=True)
    """
    The unsupported joint prim's USD type name.
    """

    supported_types: list[str] = field(kw_only=True, default_factory=list)
    """
    The joint types that can be mapped to connections.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has a joint of unsupported type "
            f"'{self.joint_type}' at '{self.joint_path}'."
        )

    def suggest_correction(self) -> str:
        if not self.supported_types:
            return ""
        return f"Use one of the supported types: {', '.join(sorted(self.supported_types))}."


@dataclass
class UsdPhysicsJointMissingChildBodyError(ParsingError):
    """
    Raised when a physics joint's ``body1`` relationship (its child link) has no target.

    Unlike ``body0``, where an unset target is the USD convention for "the stage's own
    root frame", ``body1`` has no such meaning here - every joint is expected to connect
    a link into the world, so a joint with no child leaves nothing to build a Connection
    to.
    """

    joint_path: str = field(kw_only=True)
    """
    The joint prim's stage path.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has a joint at '{self.joint_path}' with no "
            f"body1 target."
        )

    def suggest_correction(self) -> str:
        return "Inspect the joint prim's body1 relationship on the stage."


@dataclass
class UnsupportedUsdGeometryTypeError(ParsingError):
    """
    Raised when a stage contains a renderable geometric primitive
    (:class:`~pxr.UsdGeom.Gprim`) of a type this parser does not build a Shape for.

    Silently skipping it would build a World missing that piece of a link's geometry,
    the same way silently skipping an unsupported joint would build one missing a link.
    """

    prim_path: str = field(kw_only=True)
    """
    The unsupported geometry prim's stage path.
    """

    geometry_type: str = field(kw_only=True)
    """
    The unsupported geometry prim's USD type name.
    """

    supported_types: list[str] = field(kw_only=True, default_factory=list)
    """
    The geometry types that can be mapped to shapes.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has geometry of unsupported type "
            f"'{self.geometry_type}' at '{self.prim_path}'."
        )

    def suggest_correction(self) -> str:
        if not self.supported_types:
            return ""
        return f"Use one of the supported types: {', '.join(sorted(self.supported_types))}."


@dataclass(kw_only=True)
class UnreadUsdPhysicsJointsError(ParsingError):
    """
    Raised when a stage has physics joints the parser cannot attach to anything, such as
    joints outside every component of a scene.
    """

    joint_paths: list[str]
    """
    The stage paths of the physics joint prims, at least one.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has {len(self.joint_paths)} physics joints "
            f"outside every component, which this parser does not read, the first at "
            f"'{self.joint_paths[0]}'."
        )

    def suggest_correction(self) -> str:
        return (
            "Give the prim holding the joints and the links they connect the kind "
            "'component', or parse the stage with USDParser."
        )


@dataclass(kw_only=True)
class ScaledArticulatedUsdComponentError(ParsingError):
    """
    Raised when a link a physics joint inside a component connects is scaled, by its own
    transform or an ancestor's.

    The joint's authored offsets are lengths in the link's scaled frame, which the
    link's body does not have.
    """

    joint_path: str
    """
    The stage path of the physics joint prim.
    """

    scaled_prim_path: str
    """
    The stage path of the root prim of the scaled link.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has a physics joint at '{self.joint_path}' that "
            f"connects '{self.scaled_prim_path}', which is scaled, so the joint's "
            f"offsets would not be scaled with it."
        )

    def suggest_correction(self) -> str:
        return (
            "Author the scale on the geometry below the links instead of on the "
            "component, its links or their ancestors."
        )


@dataclass(kw_only=True)
class UsdJointChildIsComponentError(ParsingError):
    """
    Raised when the ``body1`` of a physics joint inside a component is the component
    itself, which is already attached to the scene's root.
    """

    joint_path: str
    """
    The stage path of the physics joint prim.
    """

    component_path: str
    """
    The stage path of the component prim.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has a physics joint at '{self.joint_path}' "
            f"whose body1 is its own component '{self.component_path}', which is "
            f"already attached to the scene's root."
        )

    def suggest_correction(self) -> str:
        return (
            "Target a link below the component with the joint's body1, or make the "
            "part the joint moves a component of its own."
        )


@dataclass(kw_only=True)
class UsdJointLinkOutsideComponentError(ParsingError):
    """
    Raised when a physics joint inside a component targets a link outside that
    component, such as a link of another component.
    """

    joint_path: str
    """
    The stage path of the physics joint prim.
    """

    component_path: str
    """
    The stage path of the component holding the joint.
    """

    link_path: str
    """
    The stage path of the root prim of the link outside the component.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has a physics joint at '{self.joint_path}' "
            f"inside the component '{self.component_path}' that connects "
            f"'{self.link_path}', which is outside it."
        )

    def suggest_correction(self) -> str:
        return (
            "Make the joint connect links of its own component only, or make one "
            "component holding both links."
        )
