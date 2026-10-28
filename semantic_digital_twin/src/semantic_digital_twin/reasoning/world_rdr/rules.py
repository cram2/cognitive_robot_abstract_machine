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
from krrood.entity_query_language.predicate import (
    Predicate,
    RenderedFields,
    SymbolicFunction,
    symbolic_function,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Prepositions
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    FunctionVerbalizationTemplates,
    Noun,
    Verb,
    clause,
    predicate_clause,
)
from typing_extensions import Iterable, Sequence, Union

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
    def recognised_from_the_joints(cls) -> tuple[NamedKind, ...]:
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


def _name_words(body: Body, word_separator: str = "_") -> list[str]:
    """
    The words of a body's name, in order, with numbering removed.

    Numbering distinguishes repetitions of the same thing, so ``shelf_level4`` says the
    same about what the body *is* as ``shelf_level`` does.

    :param body: The body whose name is read.
    :param word_separator: What separates the words of a name in the world model being
        read, as in ``handle_cabinet5_top``.
    :return: The name's words, in the order they are spoken.
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


def _match_name(words: Sequence[str], kind: AnnotationKind) -> NameMatch | None:
    """
    What ``words`` say about ``kind``, or ``None`` when they do not mention it.

    A name mentions a kind by spelling out every word of the kind's own name, or by
    using one of its synonyms.

    :param words: The words of a body's name, in order.
    :param kind: The kind the words are read against.
    :return: What the words say about the kind, or ``None`` when they do not mention it.
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
) -> AnnotationKind | None:
    """
    The kind a name settles on, or ``None`` when it mentions none of ``candidates``.

    A compound name qualifies from the left, so the kind still being spoken about at the
    end of the name is the one the body is; among kinds that reach equally far, the one
    accounting for more of the name wins, and then the more specific one.

    :param words: The words of a body's name, in order.
    :param candidates: The kinds the name is read against.
    :return: The kind the name settles on, or ``None`` when it mentions none of them.
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


@dataclass(eq=False)
class IsNamedAfter(Predicate):
    """
    Whether the body's name settles on one annotation type rather than on any other kind
    it mentions.
    """

    body: Body
    """
    The body whose name is read.
    """

    annotation_type: type[SemanticAnnotation]
    """
    The type the name has to settle on.
    """

    def __call__(self) -> bool:
        kind = asserted_kind(_name_words(self.body), NamedKind)
        return kind is not None and kind.value is self.annotation_type

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :param fields: The rendered fragment for each field.
        :return: The clause *"<body> is named after <annotation type>"*.
        """
        return predicate_clause(
            cls, Noun(fields["body"]), Noun(fields["annotation_type"])
        )


@dataclass(eq=False)
class NamesARecognisedKind(Predicate):
    """
    Whether the body's name claims one of the kinds the rules recognise by name.

    A body whose name claims none of them is left to be decided by how it is jointed.
    """

    body: Body
    """
    The body whose name is read.
    """

    def __call__(self) -> bool:
        return asserted_kind(_name_words(self.body), NamedKind) is not None

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :param fields: The rendered fragment for each field.
        :return: The clause *"<body> names a recognised kind"*.
        """
        return predicate_clause(cls, Noun(fields["body"]))


@dataclass(eq=False)
class IsNamedAfterFurniture(Predicate):
    """
    Whether the body's name claims a kind of furniture.

    Nothing about how a body is jointed tells a shelf board mounted inside a drawer
    apart from the drawer's handle, so a body the model calls furniture is left to the
    rules that go by name.
    """

    body: Body
    """
    The body whose name is read.
    """

    def __call__(self) -> bool:
        kind = asserted_kind(_name_words(self.body), NamedKind)
        return kind is not None and kind.is_furniture

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :param fields: The rendered fragment for each field.
        :return: The clause *"<body> is named after furniture"*.
        """
        return predicate_clause(cls, Noun(fields["body"]))


@dataclass(eq=False)
class NamedContainerKind(SymbolicFunction):
    """
    The kind of container the names give away - the body's own and those of its parts -
    or ``None`` when none of them names a kind more specific than a cabinet.
    """

    body: Body
    """
    The body whose branch is read.
    """

    def __call__(self) -> ContainerKind | None:
        for part in self.body._world.get_kinematic_structure_entities_of_branch(
            self.body
        ):
            kind = asserted_kind(_name_words(part), ContainerKind)
            if kind is not None:
                return kind
        return None

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :param fields: The rendered fragment for each field.
        :return: The noun phrase *"the named container kind of <body>"*.
        """
        return FunctionVerbalizationTemplates.possessive(cls, *fields.values())


# %% what the rules leave alone


@dataclass(eq=False)
class IsPartOfARobot(Predicate):
    """
    Whether the body belongs to a robot rather than to the environment.

    A robot is jointed exactly like furniture - a gripper finger slides as a drawer does,
    an arm link swings as a door does, and the link bolted to it looks like the handle
    that opens it - so without this a robot's own links would be annotated as furniture
    and its kinematic chain rewired around the joints that inserts.
    """

    body: Body
    """
    The body whose ancestry is walked.
    """

    def __call__(self) -> bool:
        robot_roots = {
            robot.root
            for robot in self.body._world.get_semantic_annotations_by_type(
                AbstractRobot
            )
        }
        ancestor = self.body
        while ancestor is not None:
            if ancestor in robot_roots:
                return True
            ancestor = ancestor.parent_kinematic_structure_entity
        return False

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :param fields: The rendered fragment for each field.
        :return: The clause *"<body> is part of a robot"*.
        """
        return predicate_clause(cls, Noun(fields["body"]))


# %% the geometry a body carries


@dataclass(eq=False)
class HasCollisionGeometry(Predicate):
    """
    Whether the body carries collision geometry of its own.

    World models use a body without geometry to carry a joint and nothing else, so it is
    part of the mechanism rather than something that can be annotated.
    """

    body: Body
    """
    The body whose geometry is read.
    """

    def __call__(self) -> bool:
        return self.body.has_collision()

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :param fields: The rendered fragment for each field.
        :return: The clause *"<body> has collision geometry"*.
        """
        return predicate_clause(cls, Noun(fields["body"]))


# %% identities the world already holds


@dataclass(eq=False)
class HasAnotherAnnotation(Predicate):
    """
    Whether something other than one annotation type has been annotated on the body.

    An object put away in a drawer is jointed exactly as a handle is, so a body whose
    identity the world already holds keeps it instead of being claimed by a rule that
    only looks at the joints.
    """

    body: Body
    """
    The body whose identity is looked up.
    """

    annotation_type: type[SemanticAnnotation]
    """
    The type being inferred, which therefore does not count as another one.
    """

    annotations: Sequence[SemanticAnnotation]
    """
    The annotations to look the body up in.
    """

    def __call__(self) -> bool:
        return any(
            annotation.root is self.body
            for annotation in self.annotations
            if isinstance(annotation, HasRootKinematicStructureEntity)
            and not isinstance(annotation, self.annotation_type)
        )

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :param fields: The rendered fragment for each field.
        :return: The clause *"<body> has an annotation besides <annotation type>"*, which
            names the annotations it was looked up in no more than the rule does.
        """
        return clause(
            Noun(fields["body"]),
            Verb("have"),
            Noun("annotation"),
            Prepositions.BESIDES,
            Noun(fields["annotation_type"]),
        )


# %% the parts a container holds


def _holder_of(body: Body) -> Body | None:
    """
    The nearest body above this one that is a real part of the furniture.

    Bodies without geometry are skipped, because world models use them to carry a joint
    and nothing else, so a door reached through a pop-out helper still resolves to the
    container it belongs to.

    :param body: The body to look above.
    :return: The nearest such body, or ``None`` when nothing above this one has
        geometry.
    """
    holder = body.parent_kinematic_structure_entity
    while holder is not None and not holder.has_collision():
        holder = holder.parent_kinematic_structure_entity
    return holder


def _parts_held_by(
    body: Body,
    annotations: Sequence[SemanticAnnotation],
    part_type: type[SemanticAnnotation],
) -> list:
    """
    The annotations of ``part_type`` whose body opens out of ``body``.

    Rules read the annotations inferred so far from what they are given rather than from
    the world, because a rule runs before its conclusions reach the world.

    :param body: The body the parts have to open out of.
    :param annotations: The annotations inferred so far.
    :param part_type: The type of part to collect.
    :return: The parts of that type the body holds.
    """
    return [
        annotation
        for annotation in annotations
        if isinstance(annotation, part_type) and _holder_of(annotation.root) is body
    ]


@symbolic_function
def drawers_of(body: Body, annotations: Sequence[SemanticAnnotation]) -> list[Drawer]:
    """
    :param body: The body the drawers have to slide out of.
    :param annotations: The annotations inferred so far.
    :return: The drawers that slide out of the body.
    """
    return _parts_held_by(body, annotations, Drawer)


@symbolic_function
def doors_of(body: Body, annotations: Sequence[SemanticAnnotation]) -> list[Door]:
    """
    :param body: The body the doors have to swing off.
    :param annotations: The annotations inferred so far.
    :return: The doors that swing off the body.
    """
    return _parts_held_by(body, annotations, Door)


@dataclass(eq=False)
class HoldsOpenableParts(Predicate):
    """
    Whether anything opens out of the body, which is what makes it a container.
    """

    body: Body
    """
    The body the parts are looked for on.
    """

    annotations: Sequence[SemanticAnnotation]
    """
    The annotations inferred so far, which the parts are looked up in.
    """

    def __call__(self) -> bool:
        return bool(
            _parts_held_by(self.body, self.annotations, Drawer)
            or _parts_held_by(self.body, self.annotations, Door)
        )

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :param fields: The rendered fragment for each field.
        :return: The clause *"<body> holds openable parts"*, which names the annotations
            it looked them up in no more than the rule does.
        """
        return clause(Noun(fields["body"]), Verb("hold"), Noun.bare("openable parts"))


# %% graspable parts


def handles(world: World) -> list[Handle]:
    """
    Every body that is a grip for opening something.

    A handle is a body of its own fixed to a part that an active joint moves, so it
    travels with what it opens without moving by itself. A lever that swings on a joint
    of its own, such as a tap's, is part of the mechanism rather than a grip on it.

    :param world: The world to read.
    :return: Every handle in it.
    """
    mount = variable(FixedConnection, world.connections)
    joint = variable(ActiveConnection, world.connections)
    grip = mount.child
    return (
        entity(inference(Handle)(root=grip))
        .where(
            joint.child == mount.parent,
            HasCollisionGeometry(grip),
            not_(IsPartOfARobot(grip)),
            not_(IsNamedAfterFurniture(grip)),
            not_(HasAnotherAnnotation(grip, Handle, world.semantic_annotations)),
        )
        .tolist()
    )


# %% things that open


def drawers_with_a_handle(world: World) -> list[Drawer]:
    """
    :param world: The world to read.
    :return: Every body a slider pulls straight out of a container, opened by a handle.
    """
    slider = variable(PrismaticConnection, world.connections)
    mount = variable(FixedConnection, world.connections)
    handle = variable(Handle, world.semantic_annotations)
    return (
        entity(inference(Drawer)(root=slider.child, handle=handle))
        .where(
            HasCollisionGeometry(slider.child),
            not_(IsPartOfARobot(slider.child)),
            mount.parent == slider.child,
            mount.child == handle.root,
        )
        .tolist()
    )


def drawers_without_a_handle(world: World) -> list[Drawer]:
    """
    :param world: The world to read.
    :return: Every body a slider pulls straight out of a container that offers nothing to
        pull it by.
    """
    slider = variable(PrismaticConnection, world.connections)
    mount = variable(FixedConnection, world.connections)
    handle = variable(Handle, world.semantic_annotations)
    return (
        entity(inference(Drawer)(root=slider.child))
        .where(
            HasCollisionGeometry(slider.child),
            not_(IsPartOfARobot(slider.child)),
            not_(
                exists(
                    mount,
                    and_(mount.parent == slider.child, mount.child == handle.root),
                )
            ),
        )
        .tolist()
    )


def doors_with_a_handle(world: World) -> list[Door]:
    """
    :param world: The world to read.
    :return: Every body a hinge swings to uncover an opening, opened by a handle.
    """
    hinge = variable(RevoluteConnection, world.connections)
    mount = variable(FixedConnection, world.connections)
    handle = variable(Handle, world.semantic_annotations)
    return (
        entity(inference(Door)(root=hinge.child, handle=handle))
        .where(
            HasCollisionGeometry(hinge.child),
            not_(IsPartOfARobot(hinge.child)),
            mount.parent == hinge.child,
            mount.child == handle.root,
        )
        .tolist()
    )


def doors_without_a_handle(
    world: World, independent_joint_multiplier: float = 1.0
) -> list[Door]:
    """
    Every leaf of a folding front that carries no handle itself, but that another leaf
    follows and is opened by.

    A front of several leaves is jointed so that the others only repeat the motion of the
    one they hang off, and is opened by a single handle on one of them. Asking for that
    keeps two things out: the linkages of a mechanism, such as a tap's joints, where
    nothing is a grip at all, and a container, whose door opens by its own joint rather
    than by following it.

    :param world: The world to read.
    :param independent_joint_multiplier: The multiplier of a joint that moves on its own.
        Any other value means the joint only repeats another joint's motion - what URDF
        calls a mimic - so the part it moves is a leaf of the same front.
    :return: Every such leaf.
    """
    hinge = variable(RevoluteConnection, world.connections)
    follower = variable(ActiveConnection1DOF, world.connections)
    mount = variable(FixedConnection, world.connections)
    handle = variable(Handle, world.semantic_annotations)
    own_mount = variable(FixedConnection, world.connections)
    return (
        entity(inference(Door)(root=hinge.child))
        .where(
            HasCollisionGeometry(hinge.child),
            not_(IsPartOfARobot(hinge.child)),
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


def _containers_of_kind(world: World, kind: ContainerKind | None) -> list[Cabinet]:
    """
    :param world: The world to read.
    :param kind: The kind the body must resolve to, or ``None`` for a container that
        names no kind more specific than a cabinet, which is inferred as a plain
        :class:`Cabinet`.
    :return: Every body that things open out of and whose name speaks for the kind.
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
            not_(IsPartOfARobot(container)),
            HoldsOpenableParts(container, annotations),
            NamedContainerKind(container) == kind,
            not_(NamesARecognisedKind(container)),
        )
        .tolist()
    )


def cabinets(world: World) -> list[Cabinet]:
    """
    :param world: The world to read.
    :return: Every container whose name says nothing more than that it holds things.
    """
    return _containers_of_kind(world, None)


def wardrobes(world: World) -> list[Wardrobe]:
    """
    :param world: The world to read.
    :return: Every container named as a wardrobe.
    """
    return _containers_of_kind(world, ContainerKind.WARDROBE)


def dishwashers(world: World) -> list[Dishwasher]:
    """
    :param world: The world to read.
    :return: Every container named as a dishwasher, by itself or by one of its parts.
    """
    return _containers_of_kind(world, ContainerKind.DISHWASHER)


def fridges(world: World) -> list[Fridge]:
    """
    :param world: The world to read.
    :return: Every container named as a fridge, by itself or by one of its parts.
    """
    return _containers_of_kind(world, ContainerKind.FRIDGE)


# %% furniture recognised by name


def _furniture_named_after(
    world: World, annotation_type: type[SemanticAnnotation]
) -> list:
    """
    :param world: The world to read.
    :param annotation_type: The type the name must settle on.
    :return: Every body with geometry that nothing moves and whose name speaks for that
        type.
    """
    mount = variable(FixedConnection, world.connections)
    body = mount.child
    return (
        entity(inference(annotation_type)(root=body))
        .where(
            HasCollisionGeometry(body),
            not_(IsPartOfARobot(body)),
            IsNamedAfter(body, annotation_type),
        )
        .tolist()
    )


def ovens(world: World) -> list[Oven]:
    """
    :param world: The world to read.
    :return: Every oven.
    """
    return _furniture_named_after(world, Oven)


def sinks(world: World) -> list[Sink]:
    """
    :param world: The world to read.
    :return: Every sink.
    """
    return _furniture_named_after(world, Sink)


def cooktops(world: World) -> list[Cooktop]:
    """
    :param world: The world to read.
    :return: Every cooktop.
    """
    return _furniture_named_after(world, Cooktop)


def counter_tops(world: World) -> list[CounterTop]:
    """
    :param world: The world to read.
    :return: Every worktop.
    """
    return _furniture_named_after(world, CounterTop)


def sofas(world: World) -> list[Sofa]:
    """
    :param world: The world to read.
    :return: Every sofa.
    """
    return _furniture_named_after(world, Sofa)


def walls(world: World) -> list[Wall]:
    """
    :param world: The world to read.
    :return: Every wall.
    """
    return _furniture_named_after(world, Wall)


def shelf_layers(world: World) -> list[ShelfLayer]:
    """
    :param world: The world to read.
    :return: Every board that things are stored on.
    """
    return _furniture_named_after(world, ShelfLayer)


def coffee_machines(world: World) -> list[CoffeeMachine]:
    """
    :param world: The world to read.
    :return: Every coffee machine.
    """
    return _furniture_named_after(world, CoffeeMachine)


def tables(world: World) -> list[Table]:
    """
    :param world: The world to read.
    :return: Every table that is no more specific kind of table.
    """
    return _furniture_named_after(world, Table)


def coffee_tables(world: World) -> list[CoffeeTable]:
    """
    :param world: The world to read.
    :return: Every coffee table.
    """
    return _furniture_named_after(world, CoffeeTable)


def side_tables(world: World) -> list[SideTable]:
    """
    :param world: The world to read.
    :return: Every table that stands beside something.
    """
    return _furniture_named_after(world, SideTable)
