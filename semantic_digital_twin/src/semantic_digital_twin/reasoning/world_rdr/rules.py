"""
The rules of the world semantic-annotation classifier.

Each rule returns every annotation of one type. Everything that opens is recognised from
the kinematic structure alone, as a query over the joints; body names are read only for
the kinds no joint distinguishes, and then through the annotation classes' own vocabulary
(:meth:`~semantic_digital_twin.world_description.world_entity.WorldEntity.class_name_tokens`
and ``_synonyms``) rather than through literals spelled here.

.. note:: The rules live beside the classifier rather than inside it because the
    classifier rebuilds each rule from the imports of
    ``world_semantic_annotations_mcrdr_defs``, so a rule has to be importable to be
    callable from there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from krrood.entity_query_language.factories import (
    and_,
    entity,
    exists,
    inference,
    not_,
    variable,
)
from krrood.entity_query_language.predicate import symbolic_function
from typing_extensions import Iterable, List, Optional, Sequence, Tuple, Type, Union

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    CoffeeMachine,
    CoffeeTable,
    Cooktop,
    CounterTop,
    Dishwasher,
    Door,
    Drawer,
    Fridge,
    Handle,
    Oven,
    ShelfLayer,
    SideTable,
    Sink,
    Sofa,
    Table,
    Wall,
    Wardrobe,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    ActiveConnection1DOF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

# %% the vocabulary the rules recognise


class NamedKind(Enum):
    """
    A kind a body name can mention.

    A compound name mentions more than one: ``sink_area_left_drawer`` says where the body
    is before it says what it is. Every kind that can appear in a name has to be a member
    here for :func:`asserted_kind` to work out which one the name settles on, including
    the kinds the rules recognise from the joints rather than from a name.

    .. note:: :class:`Handle` is deliberately absent, because a handle is recognised
        entirely from how it is jointed. See :func:`handles`.
    """

    DOOR = Door
    DRAWER = Drawer
    OVEN = Oven
    SINK = Sink
    COOKTOP = Cooktop
    COUNTER_TOP = CounterTop
    SOFA = Sofa
    WALL = Wall
    SHELF_LAYER = ShelfLayer
    COFFEE_MACHINE = CoffeeMachine
    TABLE = Table
    COFFEE_TABLE = CoffeeTable
    SIDE_TABLE = SideTable

    @classmethod
    def recognised_from_the_joints(cls) -> Tuple[NamedKind, ...]:
        """
        The kinds a joint gives away, which a name is therefore never asked to decide.

        They still take part in reading a name, so that a name mentioning one of them
        settles on it rather than on the furniture it also mentions.
        """
        return cls.DOOR, cls.DRAWER

    @property
    def is_furniture(self) -> bool:
        """
        Whether nothing but a name gives this kind away.
        """
        return self not in self.recognised_from_the_joints()


class ContainerKind(Enum):
    """
    The kinds of container that are more specific than a plain :class:`Cabinet`.

    A container is named after its kind either directly or through one of its parts, so
    an appliance whose front carries the name is still recognised.
    """

    WARDROBE = Wardrobe
    DISHWASHER = Dishwasher
    FRIDGE = Fridge


AnnotationKind = Union[NamedKind, ContainerKind]
"""
A member of either vocabulary a name can be read against, each standing for one
annotation class.
"""

# %% what a body's name asserts


def _name_words(body: Body, word_separator: str = "_") -> List[str]:
    """
    The words of a body's name, in order, with numbering removed.

    Numbering distinguishes repetitions of the same thing, so ``shelf_level4`` says the
    same about what the body *is* as ``shelf_level`` does.

    :param word_separator: What separates the words of a name in the world model being
        read, as in ``handle_cabinet5_top``.
    """
    return [
        re.sub(r"\d+", "", word)
        for word in body.name.name.lower().split(word_separator)
    ]


@dataclass
class NameMatch:
    """
    What a body's name says about one kind.
    """

    kind: AnnotationKind
    """
    The kind the name mentions.
    """

    covered_words: int
    """
    How many of the name's words the kind accounts for.
    """

    last_word_index: int
    """
    How far into the name the kind is still being spoken about.
    """


def _match_name(words: Sequence[str], kind: AnnotationKind) -> Optional[NameMatch]:
    """
    What ``words`` say about ``kind``, or ``None`` when they do not mention it.

    A name mentions a kind by spelling out every word of the kind's own name, or by
    using one of its synonyms.
    """
    annotation_type = kind.value
    class_words = annotation_type.class_name_tokens()
    if not (class_words <= set(words) or annotation_type._synonyms & set(words)):
        return None
    vocabulary = class_words | annotation_type._synonyms
    matched = [index for index, word in enumerate(words) if word in vocabulary]
    return NameMatch(kind, len(matched), max(matched))


def asserted_kind(
    words: Sequence[str], candidates: Iterable[AnnotationKind]
) -> Optional[AnnotationKind]:
    """
    The kind a name settles on, or ``None`` when it mentions none of ``candidates``.

    A compound name qualifies from the left, so the kind still being spoken about at the
    end of the name is the one the body is; among kinds that reach equally far, the one
    accounting for more of the name wins, and then the more specific one.
    """
    matches = [
        match
        for match in (_match_name(words, candidate) for candidate in candidates)
        if match is not None
    ]
    if not matches:
        return None
    last_word_index = max(match.last_word_index for match in matches)
    return max(
        (match for match in matches if match.last_word_index == last_word_index),
        key=lambda match: (
            match.covered_words,
            len(match.kind.value.__mro__),
        ),
    ).kind


@symbolic_function
def is_named_after(body: Body, annotation_type: Type[SemanticAnnotation]) -> bool:
    """
    Whether the body's name settles on ``annotation_type`` rather than on any other kind
    it mentions.
    """
    kind = asserted_kind(_name_words(body), NamedKind)
    return kind is not None and kind.value is annotation_type


@symbolic_function
def names_no_recognised_kind(body: Body) -> bool:
    """
    Whether the body's name claims none of the kinds the rules recognise by name, and so
    leaves the body to be decided by how it is jointed.
    """
    return asserted_kind(_name_words(body), NamedKind) is None


@symbolic_function
def is_not_named_after_furniture(body: Body) -> bool:
    """
    Whether the body's name refrains from claiming a kind of furniture.

    Nothing about how a body is jointed tells a shelf board mounted inside a drawer
    apart from the drawer's handle, so a body the model calls furniture is left to the
    rules that go by name.
    """
    kind = asserted_kind(_name_words(body), NamedKind)
    return kind is None or not kind.is_furniture


@symbolic_function
def container_kind_of(body: Body) -> Optional[ContainerKind]:
    """
    The kind of container the body is, read from its own name and those of its parts, or
    ``None`` when nothing names a kind more specific than a cabinet.
    """
    for part in body._world.get_kinematic_structure_entities_of_branch(body):
        kind = asserted_kind(_name_words(part), ContainerKind)
        if kind is not None:
            return kind
    return None


# %% what the rules leave alone


@symbolic_function
def is_not_part_of_a_robot(body: Body) -> bool:
    """
    Whether the body belongs to the environment rather than to a robot.

    A robot is jointed exactly like furniture - a gripper finger slides as a drawer does,
    an arm link swings as a door does, and the link bolted to it looks like the handle
    that opens it - so without this a robot's own links would be annotated as furniture
    and its kinematic chain rewired around the joints that inserts.
    """
    robot_roots = {
        robot.root
        for robot in body._world.get_semantic_annotations_by_type(AbstractRobot)
    }
    if not robot_roots:
        return True
    ancestor = body
    while ancestor is not None:
        if ancestor in robot_roots:
            return False
        ancestor = ancestor.parent_kinematic_structure_entity
    return True


# %% identities the world already holds


@symbolic_function
def is_not_already_something_else(
    body: Body,
    kind: Type[SemanticAnnotation],
    annotations: Sequence[SemanticAnnotation],
) -> bool:
    """
    Whether nothing but ``kind`` has been annotated on the body yet.

    An object put away in a drawer is jointed exactly as a handle is, so a body whose
    identity the world already holds keeps it instead of being claimed by a rule that
    only looks at the joints.
    """
    return not any(
        annotation.root is body
        for annotation in annotations
        if isinstance(annotation, HasRootKinematicStructureEntity)
        and not isinstance(annotation, kind)
    )


# %% the parts a container holds


def _holder_of(body: Body) -> Optional[Body]:
    """
    The nearest body above this one that is a real part of the furniture.

    Bodies without geometry are skipped, because world models use them to carry a joint
    and nothing else, so a door reached through a pop-out helper still resolves to the
    container it belongs to.
    """
    holder = body.parent_kinematic_structure_entity
    while holder is not None and not holder.has_collision():
        holder = holder.parent_kinematic_structure_entity
    return holder


def _parts_held_by(
    body: Body,
    annotations: Sequence[SemanticAnnotation],
    part_type: Type[SemanticAnnotation],
) -> List:
    """
    The annotations of ``part_type`` whose body opens out of ``body``.

    Rules read the annotations inferred so far from what they are given rather than from
    the world, because a rule runs before its conclusions reach the world.
    """
    return [
        annotation
        for annotation in annotations
        if isinstance(annotation, part_type) and _holder_of(annotation.root) is body
    ]


@symbolic_function
def drawers_of(body: Body, annotations: Sequence[SemanticAnnotation]) -> List[Drawer]:
    """
    The drawers that slide out of the body.
    """
    return _parts_held_by(body, annotations, Drawer)


@symbolic_function
def doors_of(body: Body, annotations: Sequence[SemanticAnnotation]) -> List[Door]:
    """
    The doors that swing off the body.
    """
    return _parts_held_by(body, annotations, Door)


@symbolic_function
def holds_openable_parts(body: Body, annotations: Sequence[SemanticAnnotation]) -> bool:
    """
    Whether anything opens out of the body, which is what makes it a container.
    """
    return bool(
        _parts_held_by(body, annotations, Drawer)
        or _parts_held_by(body, annotations, Door)
    )


# %% graspable parts


def handles(world: World) -> List[Handle]:
    """
    Every body that is a grip for opening something.

    A handle is a body of its own fixed to a part that an active joint moves, so it
    travels with what it opens without moving by itself. A lever that swings on a joint
    of its own, such as a tap's, is part of the mechanism rather than a grip on it.
    """
    mount = variable(FixedConnection, world.connections)
    joint = variable(ActiveConnection, world.connections)
    grip = mount.child
    return (
        entity(inference(Handle)(root=grip))
        .where(
            joint.child == mount.parent,
            grip.has_collision(),
            is_not_part_of_a_robot(grip),
            is_not_named_after_furniture(grip),
            is_not_already_something_else(grip, Handle, world.semantic_annotations),
        )
        .tolist()
    )


# %% things that open


def drawers_with_a_handle(world: World) -> List[Drawer]:
    """
    Every body a slider pulls straight out of a container, opened by a handle.
    """
    slider = variable(PrismaticConnection, world.connections)
    mount = variable(FixedConnection, world.connections)
    handle = variable(Handle, world.semantic_annotations)
    return (
        entity(inference(Drawer)(root=slider.child, handle=handle))
        .where(
            slider.child.has_collision(),
            is_not_part_of_a_robot(slider.child),
            mount.parent == slider.child,
            mount.child == handle.root,
        )
        .tolist()
    )


def drawers_without_a_handle(world: World) -> List[Drawer]:
    """
    Every body a slider pulls straight out of a container that offers nothing to pull it
    by.
    """
    slider = variable(PrismaticConnection, world.connections)
    mount = variable(FixedConnection, world.connections)
    handle = variable(Handle, world.semantic_annotations)
    return (
        entity(inference(Drawer)(root=slider.child))
        .where(
            slider.child.has_collision(),
            is_not_part_of_a_robot(slider.child),
            not_(
                exists(
                    mount,
                    and_(mount.parent == slider.child, mount.child == handle.root),
                )
            ),
        )
        .tolist()
    )


def doors_with_a_handle(world: World) -> List[Door]:
    """
    Every body a hinge swings to uncover an opening, opened by a handle.
    """
    hinge = variable(RevoluteConnection, world.connections)
    mount = variable(FixedConnection, world.connections)
    handle = variable(Handle, world.semantic_annotations)
    return (
        entity(inference(Door)(root=hinge.child, handle=handle))
        .where(
            hinge.child.has_collision(),
            is_not_part_of_a_robot(hinge.child),
            mount.parent == hinge.child,
            mount.child == handle.root,
        )
        .tolist()
    )


def doors_without_a_handle(
    world: World, independent_joint_multiplier: float = 1.0
) -> List[Door]:
    """
    Every leaf of a folding front that carries no handle itself, but that another leaf
    follows and is opened by.

    A front of several leaves is jointed so that the others only repeat the motion of the
    one they hang off, and is opened by a single handle on one of them. Asking for that
    keeps two things out: the linkages of a mechanism, such as a tap's joints, where
    nothing is a grip at all, and a container, whose door opens by its own joint rather
    than by following it.

    :param independent_joint_multiplier: The multiplier of a joint that moves on its own.
        Any other value means the joint only repeats another joint's motion - what URDF
        calls a mimic - so the part it moves is a leaf of the same front.
    """
    hinge = variable(RevoluteConnection, world.connections)
    follower = variable(ActiveConnection1DOF, world.connections)
    mount = variable(FixedConnection, world.connections)
    handle = variable(Handle, world.semantic_annotations)
    own_mount = variable(FixedConnection, world.connections)
    return (
        entity(inference(Door)(root=hinge.child))
        .where(
            hinge.child.has_collision(),
            is_not_part_of_a_robot(hinge.child),
            follower.parent == hinge.child,
            follower.multiplier != independent_joint_multiplier,
            mount.parent == follower.child,
            mount.child == handle.root,
            not_(
                exists(
                    own_mount,
                    and_(
                        own_mount.parent == hinge.child,
                        own_mount.child == handle.root,
                    ),
                )
            ),
        )
        .tolist()
    )


# %% containers


def _containers_of_kind(world: World, kind: Optional[ContainerKind]) -> List[Cabinet]:
    """
    Every body that things open out of and whose name speaks for ``kind``.

    :param kind: The kind the body must resolve to, or ``None`` for a container that
        names no kind more specific than a cabinet, which is inferred as a plain
        :class:`Cabinet`.
    """
    container_type = Cabinet if kind is None else kind.value
    mount = variable(FixedConnection, world.connections)
    container = mount.child
    annotations = world.semantic_annotations
    return (
        entity(
            inference(container_type)(
                root=container,
                drawers=drawers_of(container, annotations),
                doors=doors_of(container, annotations),
            )
        )
        .where(
            is_not_part_of_a_robot(container),
            holds_openable_parts(container, annotations),
            container_kind_of(container) == kind,
            names_no_recognised_kind(container),
        )
        .tolist()
    )


def cabinets(world: World) -> List[Cabinet]:
    """
    Every container whose name says nothing more than that it holds things.
    """
    return _containers_of_kind(world, None)


def wardrobes(world: World) -> List[Wardrobe]:
    """
    Every container named as a wardrobe.
    """
    return _containers_of_kind(world, ContainerKind.WARDROBE)


def dishwashers(world: World) -> List[Dishwasher]:
    """
    Every container named as a dishwasher, by itself or by one of its parts.
    """
    return _containers_of_kind(world, ContainerKind.DISHWASHER)


def fridges(world: World) -> List[Fridge]:
    """
    Every container named as a fridge, by itself or by one of its parts.
    """
    return _containers_of_kind(world, ContainerKind.FRIDGE)


# %% furniture recognised by name


def _furniture_named_after(
    world: World, annotation_type: Type[SemanticAnnotation]
) -> List:
    """
    Every body with geometry that nothing moves and whose name speaks for
    ``annotation_type``.
    """
    mount = variable(FixedConnection, world.connections)
    body = mount.child
    return (
        entity(inference(annotation_type)(root=body))
        .where(
            body.has_collision(),
            is_not_part_of_a_robot(body),
            is_named_after(body, annotation_type),
        )
        .tolist()
    )


def ovens(world: World) -> List[Oven]:
    """
    Every oven.
    """
    return _furniture_named_after(world, Oven)


def sinks(world: World) -> List[Sink]:
    """
    Every sink.
    """
    return _furniture_named_after(world, Sink)


def cooktops(world: World) -> List[Cooktop]:
    """
    Every cooktop.
    """
    return _furniture_named_after(world, Cooktop)


def counter_tops(world: World) -> List[CounterTop]:
    """
    Every worktop.
    """
    return _furniture_named_after(world, CounterTop)


def sofas(world: World) -> List[Sofa]:
    """
    Every sofa.
    """
    return _furniture_named_after(world, Sofa)


def walls(world: World) -> List[Wall]:
    """
    Every wall.
    """
    return _furniture_named_after(world, Wall)


def shelf_layers(world: World) -> List[ShelfLayer]:
    """
    Every board that things are stored on.
    """
    return _furniture_named_after(world, ShelfLayer)


def coffee_machines(world: World) -> List[CoffeeMachine]:
    """
    Every coffee machine.
    """
    return _furniture_named_after(world, CoffeeMachine)


def tables(world: World) -> List[Table]:
    """
    Every table that is no more specific kind of table.
    """
    return _furniture_named_after(world, Table)


def coffee_tables(world: World) -> List[CoffeeTable]:
    """
    Every coffee table.
    """
    return _furniture_named_after(world, CoffeeTable)


def side_tables(world: World) -> List[SideTable]:
    """
    Every table that stands beside something.
    """
    return _furniture_named_after(world, SideTable)
