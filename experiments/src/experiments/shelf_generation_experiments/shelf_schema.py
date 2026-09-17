from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional, assert_never, Union

from experiments.shelf_generation_experiments.exceptions import PathError
from experiments.shelf_generation_experiments.utils import (
    MeshCandidate,
    ObjectType,
    _MeshTypeMatcher,
)
from semantic_digital_twin.api import SpawnSpecification
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    ShelfLayer,
    Cabinet,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose2D,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import (
    Mesh,
    Scale,
    VolumetricBoundingBox,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
    Body,
)


# %%
@dataclass
class RelationalCircuitExperimentObject2D(SpawnSpecification[Body]):
    """
    An object on a shelf layer -- position is 2-D since z is determined by the layer.
    """

    object_type: ObjectType
    """
    The category of the object.
    """

    scale: Scale
    """
    Physical dimensions of the object.
    """

    pose: Pose2D
    """
    Pose of the object relative to the centre of the shelf layer's content frame.
    """

    source_id: str
    """
    Identifier used to look up the PLY mesh file for this object in the dataset.
    """

    name: Optional[str] = field(default=None, kw_only=True)
    """
    Optional explicit name for the spawned body.

    Falls back to :attr:`source_id` when unset, replacing the identifier previously
    carried on a now-removed ``id`` field.
    """

    annotation: Optional[Body] = field(default=None, compare=False)
    """
    The body spawned for this object in the world.

    ``None`` until the object is spawned by :meth:`spawn`, which sets it to the body it
    creates.
    """

    @staticmethod
    def _mesh_centered_on_footprint(
        ply_file_path: Path,
        texture_file_path: Path,
        body: KinematicStructureEntity,
    ) -> Mesh:
        """
        Load a sage10k PLY mesh with its origin re-centred onto its own footprint.

        A sage10k PLY's local origin is wherever the scan happened to put it -- a
        corner, an edge, the true centre -- with no guarantee it means anything about
        the object. Placing such a mesh at identity origin would seat the body's TF
        frame on that arbitrary point instead of the object, so the same recorded
        position visually offsets different meshes by different, mesh-specific amounts.
        Re-centring here makes the body's TF frame sit at the mesh's true horizontal
        centre and lowest point instead of at that arbitrary local origin.

        :param ply_file_path: Path to the PLY geometry file.
        :param texture_file_path: Path to the PLY's texture image.
        :param body: The body the mesh's origin is expressed relative to.
        :return: The mesh, with its origin re-centred onto its own footprint.
        """
        # sage10k meshes already carry their real-world size, so spawn at
        # identity scale to keep the mesh's true proportions instead of
        # stretching it to an independently sampled scale.
        mesh = Mesh.from_ply_file(
            ply_file_path=str(ply_file_path),
            texture_file_path=str(texture_file_path),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
            scale=Scale(1.0, 1.0, 1.0),
        )
        minimum_bound, maximum_bound = mesh.mesh.bounds
        footprint_center_x = (minimum_bound[0] + maximum_bound[0]) / 2
        footprint_center_y = (minimum_bound[1] + maximum_bound[1]) / 2
        mesh.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            -footprint_center_x,
            -footprint_center_y,
            -minimum_bound[2],
            reference_frame=body,
        )
        return mesh

    def spawn(
        self,
        world: World,
        mesh_path: Optional[Path],
        name: Optional[str] = None,
        parent: Optional[KinematicStructureEntity] = None,
        parent_T_self: Optional[HomogeneousTransformationMatrix] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: float = 0.0,
    ) -> Body:
        """
        Instantiate this object in *world* at the given absolute pose.

        The mesh keeps its own native real-world size, since sage10k PLY assets already
        carry their real dimensions; stretching them to an independently sampled scale
        would distort them.

        :param world: The world the object is created in.
        :param mesh_path: Directory containing the ``objects/`` sub-folder with PLY and
            texture files for this object.
        :param name: Overrides :attr:`name` for the spawned body's naming; falls back to
            :attr:`source_id` when neither is set.
        :param parent: The parent kinematic structure entity. Defaults to the world's
            root when omitted.
        :param parent_T_self: When given, the body is placed at this pose and *x*, *y*,
            *z* are ignored, so a caller that already computed the pose can reuse it for
            both spawning and later repositioning.
        :param x: Absolute x in the parent frame (defaults to :attr:`pose`'s x).
        :param y: Absolute y in the parent frame (defaults to :attr:`pose`'s y).
        :param z: Absolute z in the parent frame.
        :raises PathError: If *mesh_path* is ``None`` or does not exist.
        :return: The created :class:`Body`.
        """
        if mesh_path is None or not mesh_path.exists():
            raise PathError(path=mesh_path)
        ply_file = mesh_path / "objects" / f"{self.source_id}.ply"
        texture_file = mesh_path / "objects" / f"{self.source_id}_texture.png"

        _parent = parent if parent is not None else world.root

        body = Body()
        body.name = PrefixedName(
            name=str(body.id), prefix=name or self.name or self.source_id
        )

        if parent_T_self is not None:
            root_T_body = parent_T_self.copy_with_new_reference_frames(
                new_reference_frame=_parent, new_child_frame=body
            )
        else:
            root_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(
                self.pose.x if x is None else x,
                self.pose.y if y is None else y,
                z,
                yaw=self.pose.yaw,
                reference_frame=_parent,
                child_frame=body,
            )

        mesh = self._mesh_centered_on_footprint(ply_file, texture_file, body)

        geometry = ShapeCollection([mesh], reference_frame=body)
        body.visual = geometry
        body.collision = geometry

        with world.modify_world():
            root_C_body = Connection6DoF.create_with_dofs(
                world=world,
                parent=_parent,
                child=body,
            )
            world.add_body(body)
            world.add_connection(root_C_body)

        # Placing the pose in the connection's degrees of freedom rather than in
        # a fixed parent expression keeps the object movable: the ``.origin``
        # setter can later reposition it in place.
        body.parent_connection.origin = root_T_body

        type_annotation = NaturalLanguageWithTypeDescription(
            root=body, description=None, type_description=self.object_type
        )

        with world.modify_world():
            world.add_semantic_annotation(type_annotation)

        self.annotation = body
        return body

    def local_bounding_box(
        self, footprint: Scale, reference_frame: KinematicStructureEntity
    ) -> VolumetricBoundingBox:
        """
        This object's axis-aligned footprint in *reference_frame*, from *footprint*'s
        x/y extents rotated by :attr:`pose`'s own yaw around :attr:`pose`'s own
        position, extended to an infinite z-column.

        Matches :meth:`~semantic_digital_twin.semantic_annotations.mixins.
        HasSupportingSurface._object_footprint_as_infinite_column`'s own convention, so
        the result composes directly with :meth:`~semantic_digital_twin.
        world_description.graph_of_convex_sets.boxes.VolumetricGraphOfBoundingBoxes.
        free_space_from_bounding_boxes` without a real mesh: an object blocks its whole
        vertical column regardless of its actual height, since which shelf layer it
        sits on -- not this bound -- is what decides its z.

        *footprint* is taken as a parameter rather than read from :attr:`scale`, since a
        caller with a matched mesh candidate's real extents should pass those instead --
        a spawned object keeps its mesh's native size, not :attr:`scale` (see
        :meth:`spawn`).

        :param footprint: The x/y extents to rotate and bound.
        :param reference_frame: The frame the returned box is expressed in.
        :return: The rotated footprint's axis-aligned bound, extended to all z.
        """
        cos_yaw = abs(math.cos(float(self.pose.yaw)))
        sin_yaw = abs(math.sin(float(self.pose.yaw)))
        half_x = (cos_yaw * footprint.x + sin_yaw * footprint.y) / 2
        half_y = (sin_yaw * footprint.x + cos_yaw * footprint.y) / 2
        origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            float(self.pose.x),
            float(self.pose.y),
            0.0,
            reference_frame=reference_frame,
        )
        return VolumetricBoundingBox(
            -half_x, -half_y, -math.inf, half_x, half_y, math.inf, origin
        )


@dataclass
class RelationalCircuitExperimentShelfLayer(SpawnSpecification[ShelfLayer]):
    """
    A shelf layer for environment generation.

    A layer's footprint is always its shelf's own width and length -- that is how layers
    are extracted and how they are spawned -- so it carries no dimensions of its own.

    It also carries where it sits vertically in its shelf. An object's own position is
    two-dimensional, since it simply rests on the slab, so without these the height at
    which a category tends to be kept -- books low, display pieces high -- is nowhere in
    the data.
    """

    objects: list[RelationalCircuitExperimentObject2D]
    """
    Objects placed on this layer, with positions relative to the shelf centre.
    """

    theme_dominant_type: ObjectType
    """
    The object type that occurs most often among this layer's shelf's objects.
    """

    height_above_shelf_base: float = 0.0
    """
    Height of the slab above the base of its shelf, in metres.

    This is the reachable height a robot has to plan for. Zero for a layer that was not
    extracted from a real shelf.
    """

    relative_height: float = 0.0
    """
    Where the slab sits between its shelf's base (0) and top (1).

    Carried alongside :attr:`height_above_shelf_base` because it is the form
    that transfers: "books sit low" holds across shelves of different heights,
    while a given height in metres does not.
    """

    vertical_clearance: float = 0.0
    """
    Space above the slab, in metres, up to the next layer or the shelf's interior
    ceiling.

    This is what decides whether an object fits, so keeping it lets that be learned from
    real shelves instead of assumed from evenly spaced layers.
    """

    slab_top_height: Optional[float] = field(default=None, compare=False)
    """
    Height of the slab's top face, in the shelf corpus's own frame.

    ``None`` until :meth:`RelationalCircuitExperimentShelf.layers_with_geometry` computes it -- a layer that was
    only recorded or only sampled, never spawned, carries no real slab yet.
    """

    maximum_object_extents: Optional[Scale] = field(default=None, compare=False)
    """
    Largest object the layer accepts, as ``(length, width, height)`` on ``(x, y, z)``.

    ``None`` until :meth:`RelationalCircuitExperimentShelf.layers_with_geometry` computes it. Its height is the
    room up to the surface above, which is infinite for a layer resting on the shelf's
    top.
    """

    annotation: Optional[ShelfLayer] = field(default=None, compare=False)
    """
    The layer's supporting-surface annotation in the world.

    ``None`` until the layer is spawned by :meth:`spawn`, which sets it to the
    annotation it creates. Each spawned object's own body lives on its
    :attr:`RelationalCircuitExperimentObject2D.annotation`, not here.
    """

    name: Optional[str] = field(default=None, kw_only=True)
    """
    Optional explicit name for the spawned layer annotation and its body.
    """

    # % ClassVars
    slab_thickness: ClassVar[float] = 0.02
    """
    Thickness, in metres, of the spawned layer slab.
    """

    def spawn(
        self,
        world: World,
        corpus: KinematicStructureEntity,
        shelf_scale: Scale,
        name: Optional[str] = None,
        parent: Optional[KinematicStructureEntity] = None,
        parent_T_self: Optional[HomogeneousTransformationMatrix] = None,
    ) -> ShelfLayer:
        """
        Spawn this layer's slab and reparent it under the shelf ``corpus`` so the whole
        shelf moves as one unit when it is repositioned.

        Calculates the slab's free space eagerly, since it is needed for every
        placement query and survives a later
        :meth:`~semantic_digital_twin.world.World.merge_world` of the shelf into
        another world intact, even though the annotation object it is calculated on
        does not.

        :param world: The world the slab is added to.
        :param name: Overrides :attr:`name` for the spawned annotation and body.
        :param parent: The frame *parent_T_self* is expressed in, before the slab is
            reparented onto *corpus*. Defaults to the world's root when omitted.
        :param parent_T_self: The slab's pose in *parent*'s frame. Mandatory --
            :attr:`height_above_shelf_base` is where this layer's objects were
            *recorded*, not an evenly-spaced slab grid position (see
            :meth:`RelationalCircuitExperimentShelf._layer_heights`), so there is no default pose that is not
            wrong by construction; the caller (normally :meth:`RelationalCircuitExperimentShelf.spawn`) must
            compute the real slab height itself.
        :param corpus: The shelf corpus this layer belongs to; the slab is reparented
            under it once spawned.
        :param shelf_scale: The owning shelf's scale -- a layer carries no footprint of
            its own.
        :raises ValueError: If *parent_T_self* is omitted.
        :return: The spawned layer's supporting-surface annotation.
        """
        if parent_T_self is None:
            raise ValueError(
                "RelationalCircuitExperimentShelfLayer.spawn requires an explicit parent_T_self: "
                "height_above_shelf_base is a recorded position, not a spawnable slab "
                "height."
            )
        _parent = parent if parent is not None else world.root
        slab_scale = Scale(x=shelf_scale.x, y=shelf_scale.y, z=self.slab_thickness)
        layer_annotation = ShelfLayer.get_annotation_specification(
            name or self.name or "layer",
            ShelfLayer.get_default_root_kinematic_structure_entity_specification(
                scale=slab_scale
            ),
        ).spawn(world, parent=_parent, parent_T_self=parent_T_self)
        # Reparent the slab under the corpus so the whole shelf moves as one
        # unit when it is repositioned at the room level; the world pose is
        # preserved by the move.
        world.move_branch(layer_annotation.root, corpus)
        self.annotation = layer_annotation
        with world.modify_world():
            layer_annotation.calculate_supporting_surface()
        return layer_annotation


@dataclass
class RelationalCircuitExperimentShelf(SpawnSpecification[Cabinet]):
    """
    A shelf and the horizontal layers its contents rest on.

    The shelf defines the frame its contents are expressed in and always sits at that
    frame's origin. Where a shelf happened to stand in the room it was extracted from
    says nothing about shelves, so carrying it would only add a near-unique coordinate
    per training row for a circuit to split on -- so, unlike
    :class:`RelationalCircuitExperimentObject2D`, a shelf carries no pose of its own; a
    caller positions it by choosing :meth:`spawn`'s ``parent`` and ``parent_T_self``.
    """

    scale: Scale
    """
    Scale of the shelf, as ``(length, width, height)`` on ``(x, y, z)``
    """

    layers: list[RelationalCircuitExperimentShelfLayer]
    """
    The layers of the Shelf.
    """

    theme_dominant_type: ObjectType
    """
    The object type that occurs most often among this shelf's own objects, which its
    dimensions, layer count and contents are conditioned on.
    """

    source_ids: Optional[list[MeshCandidate]] = field(default=None)
    """
    Pool of candidate meshes used when placing objects on shelf layers.
    """

    annotation: Optional[Cabinet] = field(default=None, compare=False)
    """
    The shelf corpus's annotation in the world.

    ``None`` until :meth:`spawn` is called, which sets it to the annotation it creates.
    """

    name: Optional[str] = field(default=None, kw_only=True)
    """
    Optional explicit name for the spawned corpus annotation and its body.
    """

    corpus_wall_thickness: ClassVar[float] = 0.03
    """
    Thickness, in metres, of the spawned shelf corpus wall.
    """

    content_frame_yaw_offset_degrees: ClassVar[float] = 90
    """
    Offset, in degrees, between a shelf's own yaw and its content frame's yaw.
    """

    @classmethod
    def content_frame_yaw(cls, shelf_yaw_radians: float = 0.0) -> float:
        """
        Absolute yaw, in radians, of the frame a shelf's layers are stored in.

        Extraction rotates a raw object's offset from its shelf into this frame;
        :meth:`spawn` builds the corpus at this same yaw relative to *parent_T_self*.

        :param shelf_yaw_radians: The raw shelf's own yaw, before extraction. Omitted at
            spawn time, since a spawned :class:`RelationalCircuitExperimentShelf`
            carries no pose of its own.
        :return: The content frame's yaw, in radians.
        """
        return shelf_yaw_radians + math.radians(cls.content_frame_yaw_offset_degrees)

    @property
    def world(self) -> Optional[World]:
        """
        The world this shelf was spawned into, or ``None`` before :meth:`spawn` is
        called.
        """
        return None if self.annotation is None else self.annotation.root._world

    @property
    def parent(self) -> Optional[KinematicStructureEntity]:
        """
        The frame this shelf's objects' poses are expressed relative to, or ``None``
        before :meth:`spawn` is called.
        """
        return (
            None
            if self.annotation is None
            else self.annotation.root.parent_connection.parent
        )

    @property
    def corpus(self) -> Optional[Body]:
        """
        The shelf corpus's body, so a caller can check objects for collision against its
        walls in addition to each other.

        ``None`` before :meth:`spawn` is called.
        """
        return None if self.annotation is None else self.annotation.root

    @property
    def corpus_footprint(self) -> Scale:
        """
        The footprint the spawned corpus occupies, in the shelf's own frame.

        Padded by twice the corpus wall thickness so the carved-out interior is exactly
        the shelf's own footprint -- otherwise a wall intrudes into the region objects
        were trained to occupy, and an object placed near the training data's edge
        margin collides with it (most visible on small shelves, where that margin is
        thinner than the wall). A caller placing a shelf against a room wall has to
        reserve this, not the bare footprint, or the corpus reaches through by the pad.

        Taken from the shelf's own dimensions -- a layer carries none of its own.
        """
        wall_margin = 2 * self.corpus_wall_thickness
        return Scale(
            x=self.scale.x + wall_margin,
            y=self.scale.y + wall_margin,
            z=self.scale.z,
        )

    @staticmethod
    def _seat_object_on_layer(
        object_: RelationalCircuitExperimentObject2D,
        body: Body,
        slab_top_z: float,
        corpus: KinematicStructureEntity,
    ) -> None:
        """
        Lower *body* so its mesh rests on the layer slab with a small contact overlap.

        Object meshes carry their own, mesh-specific origin offset, so seating them by
        their measured collision bottom -- rather than by a fixed origin height -- both
        makes them actually rest on the slab and gives the slight overlap that
        :func:`is_supported_by` needs to register support. Assumes the object is
        upright, so its body-frame vertical extent equals its corpus-frame one.

        :param object_: The object being seated, used to recompute the pose.
        :param body: The already-spawned body to lower.
        :param slab_top_z: Height of the layer slab's top face in the corpus frame.
        :param corpus: The shelf corpus body the pose is expressed relative to.
        """
        mesh_bottom = body.collision.combined_mesh.bounds[0][2]
        origin_z = slab_top_z - mesh_bottom - 0.005
        body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            object_.pose.x,
            object_.pose.y,
            origin_z,
            yaw=object_.pose.yaw,
            reference_frame=corpus,
        )

    def _rests_on_top(
        self,
        layer: RelationalCircuitExperimentShelfLayer,
        max_allowed_layer_height: float = 1.6,
    ) -> bool:
        """
        Whether *layer* describes objects standing on the shelf rather than in it.

        A shelf has one top, but layers are drawn independently and several can come
        back recorded there, so only the highest takes it; the rest are ordinary levels.
        Otherwise their slabs would land on each other with no room between them.

        :param layer: The layer to classify.
        :param max_allowed_layer_height: The maximum height, in metres, at which a layer
            can be considered to rest on the shelf's top surface. Layers above this
            height are treated as not resting on the top surface, regardless of their
            relative height.
        :return:``True`` when its objects belong on the shelf's top surface.
        """
        if self.scale.z > max_allowed_layer_height:
            return False
        highest = max(self.layers, key=lambda candidate: candidate.relative_height)
        return layer is highest and layer.relative_height >= 1.0

    def _layer_heights(self, corpus_height: float) -> list[float]:
        """
        Height of each layer's slab in the parent frame, aligned to :attr:`layers`.

        Slabs are spaced evenly across the corpus. A layer's ``relative_height`` records
        where its objects were *found*, not where a slab is: a shelf level holding
        nothing leaves no layer behind, so a measured gap is the distance to the next
        *occupied* level rather than to the next shelf. Placing slabs at those heights
        would carry that bias into the geometry and cluster them, while real shelves are
        evenly divided. What the data does supply is how many levels a kind of shelf
        has, and that decides how many slabs there are.

        :param corpus_height: Interior height of the shelf corpus, in metres.
        :return: One height per layer, in the order of :attr:`layers`.
        """
        interior_layers = [
            layer for layer in self.layers if not self._rests_on_top(layer)
        ]
        step = corpus_height / (len(interior_layers) + 1)
        heights_bottom_up = iter(
            step * (index + 1) for index in range(len(interior_layers))
        )
        height_by_layer = {
            id(layer): (
                corpus_height if self._rests_on_top(layer) else next(heights_bottom_up)
            )
            for layer in sorted(self.layers, key=lambda layer: layer.relative_height)
        }
        return [height_by_layer[id(layer)] for layer in self.layers]

    def _surface_above_height(
        self,
        layer: RelationalCircuitExperimentShelfLayer,
        own_height: float,
        layer_heights: list[float],
        corpus_height: float,
    ) -> float:
        """
        Height, in the corpus frame, of the surface a *layer*'s objects would pierce.

        :param layer: The layer whose ceiling is wanted.
        :param own_height: That layer's own slab height above the shelf base, passed in
            rather than looked up: layers compare equal whenever they hold equal
            objects, so searching for one by value finds the wrong height.
        :param layer_heights: Every layer's slab height above the shelf base.
        :param corpus_height: Interior height of the shelf corpus, in metres.
        :return: The next slab's underside, the corpus interior ceiling, or infinity
            for a layer resting on the shelf's top, which has open air above it.
        """
        if self._rests_on_top(layer):
            return math.inf
        heights_above = [height for height in layer_heights if height > own_height]
        corpus_wall_thickness = self.corpus_wall_thickness
        layer_slab_thickness = RelationalCircuitExperimentShelfLayer.slab_thickness
        if not heights_above:
            return corpus_height / 2 - corpus_wall_thickness
        return (min(heights_above) - corpus_height / 2) - layer_slab_thickness / 2

    def layers_with_geometry(self) -> list[RelationalCircuitExperimentShelfLayer]:
        """
        Copies of :attr:`layers`, each with
        :attr:`~RelationalCircuitExperimentShelfLayer.slab_top_height` and
        :attr:`~RelationalCircuitExperimentShelfLayer.maximum_object_extents` filled in,
        in the order of :attr:`layers`.

        An object taller than the room above its slab would pierce the surface above,
        which no in-plane repair can fix, so that room is what a layer accepts.

        :return: One geometry-populated layer per entry in :attr:`layers`.
        """
        corpus_height = self.corpus_footprint.z
        layer_heights = self._layer_heights(corpus_height)
        result = []
        for layer, height in zip(self.layers, layer_heights):
            slab_top_height = (
                height - corpus_height / 2
            ) + RelationalCircuitExperimentShelfLayer.slab_thickness / 2
            surface_above_height = self._surface_above_height(
                layer, height, layer_heights, corpus_height
            )
            result.append(
                dataclasses.replace(
                    layer,
                    slab_top_height=slab_top_height,
                    maximum_object_extents=Scale(
                        x=self.scale.x,
                        y=self.scale.y,
                        z=surface_above_height - slab_top_height - 0.01,  # margin,
                    ),
                )
            )
        return result

    def layer_named(self, name: str) -> RelationalCircuitExperimentShelfLayer:
        """
        Resolve one of :attr:`layers` by the name its spawned annotation carries.

        :param name: The layer's spawned name, such as one returned by a placement query
            alongside the object it placed.
        :raises StopIteration: If no layer of this shelf carries that name.
        :return: The matching layer.
        """
        return next(
            layer for layer in self.layers if str(layer.annotation.root.name) == name
        )

    def _spawn_corpus_and_slabs(
        self,
        world: World,
        name: Optional[str] = None,
        parent: Optional[KinematicStructureEntity] = None,
        parent_T_self: Optional[HomogeneousTransformationMatrix] = None,
    ) -> tuple[Body, list[RelationalCircuitExperimentShelfLayer]]:
        """
        Instantiate the shelf's corpus and every layer's slab, but none of its objects.

        Sets :attr:`annotation` to the created corpus, so :attr:`world`, :attr:`parent`
        and :attr:`corpus` read from it immediately -- a caller that wants to inspect or
        resolve the layout before any mesh is loaded can do so between this and
        :meth:`spawn_objects`.

        :param world: The world the shelf is added to.
        :param name: Overrides :attr:`name` for the spawned corpus annotation and body.
        :param parent: The parent entity the shelf is placed under. Defaults to the
            world's root when omitted.
        :param parent_T_self: Where the shelf's own origin sits in *parent*'s frame.
            Applied on top of the shelf's intrinsic offset (identity, shifted only by
            :meth:`content_frame_yaw`'s offset and half its height) rather than
            replacing it, so the corpus and every layer move together as one rigid
            placement. Identity when omitted.
        :return: The spawned corpus body, and :meth:`layers_with_geometry`.
        """
        _parent = parent if parent is not None else world.root

        footprint = self.corpus_footprint
        # Contents are stored in the shelf's content frame, so the corpus and its
        # slabs are built in that same frame -- see content_frame_yaw.
        yaw_radians = self.content_frame_yaw()

        corpus_pose_in_content_frame = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.0,
            y=0.0,
            z=footprint.z / 2,
            yaw=yaw_radians,
            reference_frame=_parent,
        )
        corpus_pose = (
            corpus_pose_in_content_frame
            if parent_T_self is None
            else parent_T_self.dot(corpus_pose_in_content_frame)
        )
        corpus_annotation = Cabinet.get_annotation_specification(
            name or self.name or "shelf_corpus",
            Cabinet.get_default_root_kinematic_structure_entity_specification(
                scale=footprint,
                wall_thickness=0.03,
            ),
        ).spawn(world, parent=_parent, parent_T_self=corpus_pose)
        corpus_body = corpus_annotation.root
        self.annotation = corpus_annotation

        layer_heights = self._layer_heights(footprint.z)

        # Every slab is created before any object is spawned onto any of them: a
        # later pass measuring a layer's own clearance against its neighbours'
        # real geometry needs every slab already standing, not just the ones
        # before it in layer order.
        for index, (layer, height) in enumerate(zip(self.layers, layer_heights)):
            layer_pose_in_content_frame = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.0,
                y=0.0,
                z=height,
                yaw=yaw_radians,
                reference_frame=_parent,
            )
            layer_pose = (
                layer_pose_in_content_frame
                if parent_T_self is None
                else parent_T_self.dot(layer_pose_in_content_frame)
            )
            layer.spawn(
                world,
                name=f"layer_{index}",
                parent=_parent,
                parent_T_self=layer_pose,
                corpus=corpus_body,
                shelf_scale=self.scale,
            )

        return corpus_body, self.layers_with_geometry()

    def match_meshes(
        self, layers: list[RelationalCircuitExperimentShelfLayer]
    ) -> dict[int, dict[int, MeshCandidate]]:
        """
        Pick a real mesh candidate for every constant-pose object this shelf's layers
        hold, without spawning anything.

        Needs no live :class:`World`: every input (*layers*, :attr:`source_ids`) is
        plain data, so a caller can match meshes -- and read off their
        :attr:`~MeshCandidate.native_extents`/:attr:`~MeshCandidate.footprint_scale` --
        before deciding whether to spawn the layout at all.

        :param layers: This shelf's own :meth:`layers_with_geometry`, passed in rather
            than recomputed so a caller that already has them (e.g. right after
            :meth:`_spawn_corpus_and_slabs`) reads the same heights spawning used.
        :return: Per layer index, per object index, the matched candidate. An object
            with a non-constant pose or no eligible candidate is simply absent.
        """
        mesh_matcher = _MeshTypeMatcher(candidates=self.source_ids or [])
        matches: dict[int, dict[int, MeshCandidate]] = {}
        for layer_index, layer in enumerate(layers):
            max_object_extents = layer.maximum_object_extents
            layer_matches: dict[int, MeshCandidate] = {}
            for object_index, object_ in enumerate(layer.objects):
                if not object_.pose.x.is_constant():
                    continue
                candidate = (
                    mesh_matcher.random_match(
                        object_.object_type,
                        max_extents=max_object_extents,
                        target_extents=object_.scale,
                    )
                    if self.source_ids
                    else None
                )
                # No mesh of this type is small enough for the layer, or none is
                # cached at all. The object is either too big for the shelf or
                # simply unrenderable, so it is left out.
                if candidate is None:
                    continue
                layer_matches[object_index] = candidate
            matches[layer_index] = layer_matches
        return matches

    @staticmethod
    def spawn_objects(
        world: World,
        corpus_body: Body,
        layers: list[RelationalCircuitExperimentShelfLayer],
        matches: dict[int, dict[int, MeshCandidate]],
    ) -> None:
        """
        Spawn the real mesh for every object *matches* names, and seat it on its layer.

        Unavoidably needs the real loaded mesh -- seating an object on its slab reads
        the mesh's own collision bottom (see :meth:`_seat_object_on_layer`) -- so this
        is the one phase of spawning that cannot run before mesh geometry exists.

        :param world: The world *corpus_body* was spawned into.
        :param corpus_body: The shelf corpus body, from :meth:`_spawn_corpus_and_slabs`.
        :param layers: This shelf's own :meth:`layers_with_geometry`.
        :param matches: Per layer index, per object index, the candidate to spawn --
            typically from :meth:`match_meshes`, possibly narrowed by a resolver. An
            object absent from its layer's matches is left unspawned.
        """
        for layer_index, layer in enumerate(layers):
            slab_top_z: float = layer.slab_top_height
            layer_matches = matches.get(layer_index, {})
            for object_index, object_ in enumerate(layer.objects):
                candidate = layer_matches.get(object_index)
                if candidate is None:
                    continue
                object_.source_id = candidate.source_id
                body = object_.spawn(
                    world=world,
                    parent=corpus_body,
                    parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                        object_.pose.x,
                        object_.pose.y,
                        slab_top_z,
                        yaw=object_.pose.yaw,
                        reference_frame=corpus_body,
                    ),
                    mesh_path=candidate.scene_directory,
                )
                RelationalCircuitExperimentShelf._seat_object_on_layer(
                    object_, body, slab_top_z, corpus_body
                )

    def spawn(
        self,
        world: World,
        name: Optional[str] = None,
        parent: Optional[KinematicStructureEntity] = None,
        parent_T_self: Optional[HomogeneousTransformationMatrix] = None,
    ) -> Cabinet:
        """
        Instantiate the shelf and its objects inside a :class:`World`.

        Mutates ``self`` in place: :attr:`annotation` is set to the created corpus (so
        :attr:`world`, :attr:`parent` and :attr:`corpus` read from it), each
        :class:`RelationalCircuitExperimentShelfLayer`'s own :attr:`~RelationalCircuitExperimentShelfLayer.annotation` is set to its slab,
        and each spawned :class:`RelationalCircuitExperimentObject2D`'s :attr:`~RelationalCircuitExperimentObject2D.annotation` is set to
        its body -- so a caller keeps using this same shelf afterwards instead of a
        separate handle.

        :param world: The world the shelf is added to.
        :param name: Overrides :attr:`name` for the spawned corpus annotation and body.
        :param parent: The parent entity the shelf is placed under. Defaults to the
            world's root when omitted.
        :param parent_T_self: Where the shelf's own origin sits in *parent*'s frame.
            Applied on top of the shelf's intrinsic offset rather than replacing it, so
            the corpus and every layer move together as one rigid placement. Identity
            when omitted.
        :return: The spawned corpus annotation.
        """
        corpus_body, layers = self._spawn_corpus_and_slabs(
            world, name, parent, parent_T_self
        )
        matches = self.match_meshes(layers)
        RelationalCircuitExperimentShelf.spawn_objects(
            self.world, corpus_body, layers, matches
        )
        return self.annotation
