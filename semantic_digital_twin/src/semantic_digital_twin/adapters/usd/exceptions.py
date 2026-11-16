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
class UnsupportedUsdComponentSceneError(ParsingError):
    """
    Raised when a valid USD scene holds something
    :class:`~semantic_digital_twin.adapters.usd.parser.USDComponentParser` does not read
    yet.
    """


@dataclass(kw_only=True)
class UnreadUsdPhysicsJointsError(UnsupportedUsdComponentSceneError):
    """
    Raised when a scene has physics joints outside every component, such as a door
    hinged to a wall.
    """

    joint_paths: list[str]
    """
    The stage paths of the physics joint prims, at least one.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has {len(self.joint_paths)} physics joints "
            f"outside every component, the first at '{self.joint_paths[0]}'. "
            f"USDComponentParser only reads the joints inside a component so far."
        )

    def suggest_correction(self) -> str:
        return (
            "Extend USDComponentParser to read joints outside components, making each "
            "link they connect a body of its own."
        )


@dataclass(kw_only=True)
class UsdJointChildIsComponentError(UnsupportedUsdComponentSceneError):
    """
    Raised when the ``body1`` of a physics joint inside a component is the component
    itself, such as a door asset hinged to the world.
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
            f"whose body1 is its own component '{self.component_path}'. "
            f"USDComponentParser attaches every component to the scene's root so far, "
            f"never by a joint."
        )

    def suggest_correction(self) -> str:
        return (
            "Extend USDComponentParser to attach a component that is a joint's body1 "
            "by that joint instead of to the scene's root."
        )


@dataclass(kw_only=True)
class UsdJointLinkOutsideComponentError(UnsupportedUsdComponentSceneError):
    """
    Raised when a physics joint inside a component connects a link outside that
    component, such as a drawer whose slide belongs to another component.
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
            f"'{self.link_path}', which is outside it. USDComponentParser only reads "
            f"joints between the links of one component so far."
        )

    def suggest_correction(self) -> str:
        return (
            "Extend USDComponentParser to read joints that connect links of different "
            "components, or links outside every component."
        )


@dataclass(kw_only=True)
class ScaledArticulatedUsdComponentError(UnsupportedUsdComponentSceneError):
    """
    Raised when a link a physics joint inside a component connects is scaled, by its own
    transform or an ancestor's, such as in a scene exported in centimetres.
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
            f"connects '{self.scaled_prim_path}', which is scaled. USDComponentParser "
            f"does not scale a joint's offsets so far, so the link would be misplaced."
        )

    def suggest_correction(self) -> str:
        return (
            "Extend USDComponentParser to scale a joint's offsets by the scale of the "
            "link each is given in."
        )
