from __future__ import annotations

import math
import multiprocessing
import os
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import numpy as np
import trimesh
from sklearn.cluster import DBSCAN
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from experiments.scene_generation_experiments.utils import (
    ObjectType,
    build_source_id_to_path,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine, drop_database
from experiments.scene_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
    RelationalCircuitExperimentShelf,
    RelationalCircuitExperimentShelfLayer,
)
from semantic_digital_twin.spatial_types import (
    Point2,
    Pose,
    Pose2D,
)
from semantic_digital_twin.world_description.geometry import Scale

if TYPE_CHECKING:
    from semantic_digital_twin.orm.ormatic_interface import Sage10kObjectDAO


def _rotate_offset_into_frame(offset: Point2, frame_yaw_radians: float) -> Point2:
    """
    Express *offset*, currently an offset along the world axes, in the axes of a frame
    rotated by *frame_yaw_radians*.

    Needed wherever an object's offset from a rotated parent is stored for later re-use
    *inside* that parent: keeping the offset on the world axes makes it mean something
    different once the parent's own rotation is applied again.

    :param offset: The offset along the world axes.
    :param frame_yaw_radians: Yaw of the target frame, in radians.
    :return: The same offset expressed in the target frame's axes.
    """
    cosine = math.cos(frame_yaw_radians)
    sine = math.sin(frame_yaw_radians)
    offset_x = float(offset.x)
    offset_y = float(offset.y)
    return Point2(
        x=offset_x * cosine + offset_y * sine,
        y=-offset_x * sine + offset_y * cosine,
    )


# %% classification
@dataclass(frozen=True)
class ObjectTypeClassifier:
    """
    Maps the free-form, near-instance-specific ``object_type`` strings found in the raw
    sage10k dataset (e.g. ``"book2"``, ``"bookchair8eba7fdc"``) onto the generalized
    :class:`ObjectType` categories.

    Matching is a case-insensitive, ordered keyword lookup: the raw string is tested
    against each category's keywords in turn, and the first category with a matching
    keyword wins. Furniture/surface categories (shelf, table, desk, ...) are checked
    before small-item categories, since the dataset frequently names an item together
    with the furniture it sits on (e.g. ``"bookshelf"``, ``"candletable"``) and the
    furniture is usually the more useful category for scene- layout purposes. This is a
    best-effort heuristic, not a guaranteed- correct classification -- raw strings that
    combine two plausible categories in an unusual order may be mapped to the "wrong"
    one.
    """

    _keywords_by_type: ClassVar[tuple[tuple[ObjectType, tuple[str, ...]], ...]] = (
        # -- Furniture -----------------------------------------------------
        (ObjectType.WORKBENCH, ("workbench",)),
        (ObjectType.DISPLAYCASE, ("displaycase", "showcase")),
        (ObjectType.WARDROBE, ("wardrobe", "closet")),
        (ObjectType.DRESSER, ("dresser",)),
        (ObjectType.LOCKER, ("locker",)),
        (ObjectType.PANTRY, ("pantry",)),
        (ObjectType.VANITY, ("vanity",)),
        (ObjectType.NIGHTSTAND, ("nightstand",)),
        (ObjectType.SIDEBOARD, ("sideboard", "console", "credenza")),
        (ObjectType.SHELF, ("shelf", "shelv", "rack", "bookcase")),
        (ObjectType.CABINET, ("cabinet",)),
        (ObjectType.DESK, ("desk",)),
        (ObjectType.COUNTER, ("counter", "countertop")),
        (ObjectType.SOFA, ("sofa", "couch")),
        (ObjectType.BENCH, ("bench",)),
        (ObjectType.BED, ("bed", "crib")),
        (
            ObjectType.CHAIR,
            ("chair", "stool", "armchair", "ottoman", "pouf", "barstool"),
        ),
        (ObjectType.TABLE, ("table", "island")),
        (ObjectType.CART, ("cart", "trolley")),
        (ObjectType.CRATE, ("crate", "pallet")),
        (ObjectType.TOOLBOX, ("toolbox",)),
        (ObjectType.PEDESTAL, ("pedestal", "podium", "plinth")),
        (
            ObjectType.STAND,
            ("stand", "holder", "hanger", "easel", "coatrack", "clothingrack"),
        ),
        # -- Plants (checked early: "pot" and "table" are common substrings of
        # "pottedplant"/"planttable"-style compounds, and the plant is the more
        # useful category for those) --------------------------------------
        (
            ObjectType.PLANT,
            (
                "plant",
                "succulent",
                "fern",
                "cactus",
                "ficus",
                "orchid",
                "palm",
                "bamboo",
                "flower",
                "tree",
            ),
        ),
        # -- Kitchen / dining ------------------------------------------------
        (ObjectType.CUTTING_BOARD, ("cuttingboard", "cutting_board")),
        (ObjectType.DISHWASHER, ("dishwasher",)),
        (ObjectType.REFRIGERATOR, ("fridge", "refrigerator", "freezer")),
        (ObjectType.SINK, ("sink",)),
        (ObjectType.OVEN, ("oven", "stove")),
        (ObjectType.MICROWAVE, ("microwave",)),
        (ObjectType.SMALL_APPLIANCE, ("toaster", "coffeemaker", "kettle", "blender")),
        (ObjectType.DISPENSER, ("dispenser",)),
        (ObjectType.CUTLERY, ("cutlery", "fork", "spoon", "spatula", "rollingpin")),
        (ObjectType.KNIFE, ("knife",)),
        (ObjectType.CUP, ("cup", "mug", "tumbler", "teacup")),
        (ObjectType.GLASS, ("glass", "wineglass")),
        (ObjectType.PLATE, ("plate",)),
        (ObjectType.BOWL, ("bowl",)),
        (ObjectType.BOTTLE, ("bottle",)),
        (ObjectType.JAR, ("jar", "shaker", "spicejar")),
        (ObjectType.UTENSIL, ("utensil",)),
        (ObjectType.POT, ("pot", "peppergrinder")),
        (ObjectType.TRAY, ("tray",)),
        # -- Lighting --------------------------------------------------------
        (ObjectType.CHANDELIER, ("chandelier",)),
        (ObjectType.NEON_SIGN, ("neon",)),
        (ObjectType.CANDLE, ("candle", "candelabra", "candlestick", "lantern")),
        (ObjectType.LAMP, ("lamp",)),
        (
            ObjectType.LIGHT_FIXTURE,
            ("light", "sconce", "fixture", "pendant", "ledstrip", "lightstrip"),
        ),
        # -- Electronics (checked before decor/art: "printer" and
        # "smartphone" would otherwise match ART's "print"/"art" substrings)
        # ---------------------------------------------------------------
        (ObjectType.TELEVISION, ("tv", "television")),
        (ObjectType.PROJECTOR, ("projector",)),
        (ObjectType.COMPUTER, ("computer", "laptop")),
        (ObjectType.KEYBOARD, ("keyboard",)),
        (ObjectType.MOUSE, ("mouse",)),
        (ObjectType.MONITOR, ("monitor", "screen")),
        (ObjectType.CAMERA, ("camera",)),
        (ObjectType.SPEAKER, ("speaker",)),
        (ObjectType.PHONE, ("phone", "smartphone")),
        (ObjectType.PRINTER, ("printer",)),
        (ObjectType.REMOTE_CONTROL, ("remote", "controller")),
        # -- Decor / art -------------------------------------------------------
        (ObjectType.MIRROR, ("mirror",)),
        (ObjectType.CLOCK, ("clock",)),
        (
            ObjectType.SCULPTURE,
            ("sculpture", "figurine", "statue", "bust", "mannequin"),
        ),
        (ObjectType.VASE, ("vase", "urn", "planter")),
        (ObjectType.TAPESTRY, ("tapestry", "wallhanging", "banner", "flag")),
        (ObjectType.FRAME, ("frame", "pictureframe")),
        (ObjectType.PEGBOARD, ("pegboard",)),
        (
            ObjectType.SIGN,
            ("sign", "menuboard", "whiteboard", "blackboard", "chart", "map"),
        ),
        (
            ObjectType.ART,
            (
                "art",
                "painting",
                "poster",
                "print",
                "picture",
                "canvas",
                "mural",
                "decor",
                "ornament",
                "brassdecor",
                "stainedglass",
                "globe",
                "seashell",
            ),
        ),
        # -- Food --------------------------------------------------------------
        (
            ObjectType.FOOD,
            (
                "apple",
                "fig",
                "pastry",
                "cannedgood",
                "canned",
                "condiment",
                "croissant",
                "bakingpowder",
                "flourbag",
                "bread",
                "herb",
                "spice",
            ),
        ),
        # -- Reading / office --------------------------------------------------
        (
            ObjectType.BOOK,
            (
                "book",
                "notebook",
                "magazine",
                "notepad",
                "tome",
                "volume",
                "folio",
                "textbook",
                "cookbook",
                "hardcover",
                "novel",
                "codex",
            ),
        ),
        (ObjectType.PEN, ("pen", "pencil", "crayon", "quill")),
        (
            ObjectType.OFFICE_SUPPLY,
            (
                "stapler",
                "paperclip",
                "ruler",
                "folder",
                "eraser",
                "tape",
                "scissors",
                "businesscard",
            ),
        ),
        # -- Bath / personal care ----------------------------------------------
        (ObjectType.TOILET, ("toilet",)),
        (ObjectType.BATHTUB, ("bathtub", "shower")),
        (ObjectType.TOWEL, ("towel", "napkin")),
        (
            ObjectType.PERSONAL_CARE_PRODUCT,
            (
                "soap",
                "shampoo",
                "lotion",
                "conditioner",
                "toothbrush",
                "toothpaste",
                "cosmetic",
                "perfume",
                "sanitizer",
                "bodywash",
                "hairproduct",
                "comb",
                "brush",
                "diaper",
                "syringe",
                "medicalsupply",
                "stethoscope",
            ),
        ),
        # -- Tools / hardware ----------------------------------------------------
        (
            ObjectType.TOOL,
            (
                "tool",
                "wrench",
                "hammer",
                "screwdriver",
                "drill",
                "pliers",
                "sander",
                "scale",
                "gauge",
            ),
        ),
        (
            ObjectType.HARDWARE,
            (
                "gear",
                "wire",
                "pipe",
                "hook",
                "outlet",
                "cable",
                "circuit",
                "socket",
                "cog",
                "chip",
                "sensor",
                "router",
                "key",
                "button",
            ),
        ),
        (ObjectType.LADDER, ("ladder",)),
        (ObjectType.SAFETY_EQUIPMENT, ("extinguisher", "smokedetector", "firealarm")),
        # -- Containers ----------------------------------------------------------
        (ObjectType.TRASH, ("trash", "waste")),
        (ObjectType.BASKET, ("basket",)),
        (ObjectType.BIN, ("bin",)),
        (ObjectType.BOX, ("box",)),
        (ObjectType.BUCKET, ("bucket",)),
        (
            ObjectType.CONTAINER,
            ("container", "case", "can", "barrel", "tub", "trunk", "caddy"),
        ),
        # -- Structural / architectural --------------------------------------
        (ObjectType.WINDOW, ("window",)),
        (ObjectType.DOOR, ("door",)),
        (ObjectType.FIREPLACE, ("fireplace",)),
        (ObjectType.VENT, ("vent", "radiator")),
        (
            ObjectType.PANEL,
            (
                "panel",
                "tile",
                "wallpaper",
                "molding",
                "column",
                "beam",
                "arch",
                "grille",
                "trim",
            ),
        ),
        # -- Textiles --------------------------------------------------------
        (ObjectType.PILLOW, ("pillow", "cushion")),
        (ObjectType.TEXTILE, ("textile", "fabric", "rug", "carpet", "blanket")),
        # -- Misc ---------------------------------------------------------------
        (ObjectType.APPAREL, ("shoe", "watch", "glasses")),
        (
            ObjectType.SPORTS_EQUIPMENT,
            ("dumbbell", "treadmill", "elliptical", "kettlebell"),
        ),
        (ObjectType.VEHICLE, ("car", "bike", "tire")),
        (
            ObjectType.RETAIL_FIXTURE,
            (
                "cashregister",
                "register",
                "checkout",
                "pricetag",
                "coin",
                "display",
                "kiosk",
                "station",
                "booth",
            ),
        ),
        (ObjectType.TOY, ("toy",)),
        (ObjectType.WASHING_MACHINE, ("washingmachine", "washer")),
        (ObjectType.DRYER, ("dryer",)),
    )

    def classify(self, raw_type: str) -> ObjectType:
        """
        Return the :class:`ObjectType` category whose keywords best match *raw_type*.

        :param raw_type: A raw, near-instance-specific ``object_type`` string from the
            sage10k dataset (e.g. ``"book2"``).
        :return: The best-matching generalized category, or :attr:`ObjectType.OTHER` if
            no keyword matches.
        """
        normalized = raw_type.strip().lower()
        for object_type, keywords in self._keywords_by_type:
            if any(keyword in normalized for keyword in keywords):
                return object_type
        return ObjectType.OTHER


@dataclass(frozen=True)
class ShelfMembershipClassifier:
    """
    Decides whether a free-form furniture name from the raw sage10k dataset (e.g.
    ``"bookshelf2"``, ``"storagecabinet"``) describes shelf-like storage furniture at
    all.

    Matching is a case-insensitive substring lookup against a fixed keyword set. This
    is the gate deciding which furniture enters training as a shelf -- a name outside
    the keyword set answers ``False`` rather than being admitted as some catch-all
    kind of shelf, which would let every table and chair in the dataset in.

    A shelf's kind is no longer classified from its furniture name; see
    :attr:`~experiments.scene_generation_experiments.shelf_schema.RelationalCircuitExperimentShelf.theme_dominant_type`,
    which is derived from what is actually placed on the shelf instead.
    """

    _KEYWORDS: ClassVar[tuple[str, ...]] = (
        "bookshelf",
        "bookcase",
        "book_shelf",
        "book_case",
        "cabinet",
        "sideboard",
        "console",
        "credenza",
        "shelf",
        "shelv",
        "rack",
    )
    """
    Keywords identifying shelf-like furniture, matched as substrings of the raw name.
    """

    def is_shelf_like(self, raw_type: str) -> bool:
        """
        Decide whether a raw furniture name describes shelf-like storage furniture.

        :param raw_type: The dataset's free-form name for the furniture.
        :return:``True`` when the name matches a modelled shelf-like keyword.
        """
        normalized_type = raw_type.lower()
        return any(keyword in normalized_type for keyword in self._KEYWORDS)


@dataclass(frozen=True)
class CorrectedPosition:
    """
    An object's horizontal position after the mesh-centring correction, with the
    provenance of that position.
    """

    position: Point2
    """
    The object's position in world coordinates.
    """

    is_mesh_corrected: bool
    """
    Whether :attr:`position` is the mesh's measured bounding-box centre rather than the
    dataset's unmodified recorded position.
    """


@dataclass(frozen=True)
class MeshBounds:
    """
    The measurements taken from one mesh: where its horizontal centre sits relative to
    its origin, and how far it reaches vertically.

    Kept as plain floats rather than :class:`~semantic_digital_twin.spatial_types.spatial_types.Point2`:
    this crosses process boundaries twice, as the return value of
    :meth:`MeshMeasurements._load_mesh_bounds` under
    :meth:`Sage10kPreprocessingRun._measure_meshes_in_parallel`
    and again as part of the ``bounds_by_source_id`` argument handed to
    :meth:`Sage10kPreprocessingRun._process_room_shard`, and a
    :class:`Point2`'s casadi-backed value cannot be pickled across a
    :class:`~concurrent.futures.ProcessPoolExecutor` boundary.
    """

    footprint_center_x: float
    """
    Horizontal centre of the mesh's bounding box along x, in the mesh's own frame.
    """

    footprint_center_y: float
    """
    Horizontal centre of the mesh's bounding box along y, in the mesh's own frame.
    """

    bottom: float
    """
    Lowest point of the mesh, in its own frame.
    """

    top: float
    """
    Highest point of the mesh, in its own frame.
    """

    @property
    def height(self) -> float:
        """
        The mesh's total vertical size.
        """
        return self.top - self.bottom


@dataclass
class MeshMeasurements:
    """
    Measures cached meshes, so an object's recorded position can be corrected onto its
    mesh's true horizontal centre and a shelf's real base and top can be located.

    A sage10k object's recorded position is its mesh's local origin, which the dataset
    does not guarantee to be that mesh's centre -- much as a room's recorded position is
    its lower-left corner.
    """

    source_id_to_path: dict[str, Path]
    """
    Maps a mesh's source id to the cached scene directory holding it, as returned by
    :func:`build_source_id_to_path`.
    """

    _bounds_by_source_id: dict[str, Optional[MeshBounds]] = field(default_factory=dict)
    """
    Memoizes each measured mesh, since many objects share one asset.

    ``None`` records that a mesh was not available to measure.
    """

    @property
    def measured_mesh_count(self) -> int:
        """
        How many distinct meshes were actually loaded and measured.
        """
        return sum(bounds is not None for bounds in self._bounds_by_source_id.values())

    def corrected_position(
        self, source_id: str, position: Point2, yaw_degrees: float
    ) -> CorrectedPosition:
        """
        Correct *position* onto the true centre of *source_id*'s mesh.

        Falls back to *position* unchanged, flagged as uncorrected, when the mesh is not
        cached locally, so preprocessing still runs on a partial mesh cache without
        silently passing off uncorrected data as corrected.

        :param source_id: Identifies the object's mesh asset.
        :param position: The object's recorded world position.
        :param yaw_degrees: The object's own yaw, which the mesh-local offset is rotated
            by to reach world axes.
        :return: The corrected position and whether the mesh supplied it.
        """
        bounds = self.bounds(source_id)
        if bounds is None:
            return CorrectedPosition(position=position, is_mesh_corrected=False)
        # _rotate_offset_into_frame(offset, theta) is R(-theta); negating the angle
        # gives R(+theta), the forward rotation into world axes.
        world_offset = _rotate_offset_into_frame(
            Point2(x=bounds.footprint_center_x, y=bounds.footprint_center_y),
            math.radians(-yaw_degrees),
        )
        return CorrectedPosition(
            position=Point2(
                x=position.x + world_offset.x, y=position.y + world_offset.y
            ),
            is_mesh_corrected=True,
        )

    def bounds(self, source_id: str) -> Optional[MeshBounds]:
        """
        The measurements of *source_id*'s mesh, loading it on first request.

        :param source_id: Identifies the mesh asset to measure.
        :return: The mesh's measurements, or ``None`` when it is not cached.
        """
        if source_id not in self._bounds_by_source_id:
            self._bounds_by_source_id[source_id] = self._measure(source_id)
        return self._bounds_by_source_id[source_id]

    def _measure(self, source_id: str) -> Optional[MeshBounds]:
        """
        Load *source_id*'s mesh and measure its bounding box.

        :param source_id: Identifies the mesh asset to measure.
        :return: The mesh's measurements, or ``None`` when it is not cached.
        """
        return MeshMeasurements._load_mesh_bounds(
            source_id, self.source_id_to_path.get(source_id)
        )

    @staticmethod
    def _load_mesh_bounds(
        source_id: str, scene_directory: Optional[Path]
    ) -> Optional[MeshBounds]:
        """
        Load and measure *source_id*'s mesh from *scene_directory*.

        A staticmethod so a parallel measurement pass can submit it to a worker process
        directly (it pickles by reference, like a free function, rather than needing a
        live :class:`MeshMeasurements` instance).

        :param source_id: Identifies the mesh asset to measure.
        :param scene_directory: The cached scene directory holding the mesh, or ``None``
            when it is not cached locally.
        :return: The mesh's measurements, or ``None`` when it is not cached.
        """
        if scene_directory is None:
            return None
        mesh = trimesh.load(
            str(scene_directory / "objects" / f"{source_id}.ply"), process=False
        )
        minimum_bound, maximum_bound = mesh.bounds
        # The bounds are numpy scalars, which pass for floats until PostgreSQL is
        # handed their repr instead of a number, so they are converted here rather
        # than at every place a measurement ends up in a stored field.
        return MeshBounds(
            footprint_center_x=float((minimum_bound[0] + maximum_bound[0]) / 2),
            footprint_center_y=float((minimum_bound[1] + maximum_bound[1]) / 2),
            bottom=float(minimum_bound[2]),
            top=float(maximum_bound[2]),
        )


# %% preprocessing's own raw-object representation
@dataclass
class PreprocessedObject:
    id: str
    """
    Unique identifier of the object in the dataset.

    Unlike :attr:`source_id`, which names a mesh asset that many objects can share, this
    is what :attr:`place_id` and shelf-membership matching join on to tell one placed
    object apart from another.
    """

    room_id: str
    """
    The id of the room where the object is located.
    """

    place_id: str
    """
    The id of the object where the object is located/placed on/at, e.g. wall, floor,
    anchor, or the id of a piece of furniture it stands on.
    """

    object_type: ObjectType
    """
    The type of the object.
    """

    scale: Scale
    """
    The scale of the object.
    """

    pose: Pose
    """
    Pose of the object.

    ``roll``/``pitch``/``yaw`` are in radians, as :class:`~semantic_digital_twin.spatial_types.spatial_types.Pose`
    requires -- everywhere else in this schema reasons about rotation in degrees, so it is
    converted at the boundary where a value enters or leaves this field.
    """

    source_id: str
    """
    Identifier used to look up the PLY mesh file for this object in the dataset.
    """

    description: Optional[str] = None
    """
    Free-text description of the object as written in the source dataset.
    """

    place_guidance: Optional[str] = None
    """
    Free-text description of where the object is meant to be placed, as written in the
    source dataset.
    """

    position_is_mesh_corrected: bool = True
    """
    Whether :attr:`pose`'s position was corrected to the object's true mesh bounding-box
    centre.

    ``False`` means the mesh was unavailable when the object was processed and the
    source dataset's recorded position was kept unchanged, which is not guaranteed to be
    the mesh's centre. Consumers that need centred positions should filter on this.
    """

    @staticmethod
    def from_sage10k_object(
        sage10k_object: Sage10kObjectDAO,
        classifier: ObjectTypeClassifier,
        measurements: MeshMeasurements,
    ) -> PreprocessedObject:
        """
        Build the processed :class:`PreprocessedObject` equivalent of *sage10k_object*:
        its free- form type mapped onto a unified :class:`ObjectType`, its horizontal
        position corrected onto its mesh's centre, and the dataset's free text carried
        through for later placement reasoning.

        :param sage10k_object: Raw object row, with its ``position``, ``rotation`` and
            ``dimensions`` relationships already loaded.
        :param classifier: Maps the raw type string onto an :class:`ObjectType`.
        :param measurements: Supplies the mesh-centring correction.
        :return: The processed object.
        """
        corrected = measurements.corrected_position(
            source_id=sage10k_object.source_id,
            position=Point2(x=sage10k_object.position.x, y=sage10k_object.position.y),
            yaw_degrees=sage10k_object.rotation.z,
        )
        return PreprocessedObject(
            id=sage10k_object.id,
            room_id=sage10k_object.room_id,
            place_id=sage10k_object.place_id,
            object_type=classifier.classify(sage10k_object.type),
            scale=Scale(
                x=sage10k_object.dimensions.length,
                y=sage10k_object.dimensions.width,
                z=sage10k_object.dimensions.height,
            ),
            pose=Pose.from_xyz_rpy(
                x=corrected.position.x,
                y=corrected.position.y,
                z=sage10k_object.position.z,
                roll=math.radians(sage10k_object.rotation.x),
                pitch=math.radians(sage10k_object.rotation.y),
                yaw=math.radians(sage10k_object.rotation.z),
            ),
            source_id=sage10k_object.source_id,
            description=sage10k_object.description,
            place_guidance=sage10k_object.place_guidance,
            position_is_mesh_corrected=corrected.is_mesh_corrected,
        )


@dataclass
class ShelfContents:
    """
    Holds back the objects that layer extraction reads -- the shelves and whatever
    stands on them -- while the rest of the dataset streams past.

    Extraction only ever looks at a shelf and the objects naming it as their place, so
    every other object can be written to the processed database and released straight
    away. Keeping the whole dataset instead exhausts memory long before it is written.
    """

    shelf_ids: set[str]
    """
    Ids of the raw objects classified as shelf-like.
    """

    relevant_source_ids: set[str] = field(default_factory=set)
    """
    Source ids of the shelves and of the objects standing on them.

    Mesh measurement -- and so mesh-corrected positions -- only ever feeds layer
    extraction (:meth:`Sage10kPreprocessingRun._shelves_with_layers`), which only ever
    reads shelves and their own contents. Every other mesh in the raw dataset can be
    measured for nothing, so this is the scope
    :meth:`Sage10kPreprocessingRun._measure_meshes_in_parallel` should be run against
    instead of every distinct source id in the dataset.
    """

    objects: list[PreprocessedObject] = field(default_factory=list)
    """
    The kept objects, in the order they were read.
    """

    @classmethod
    def from_raw_objects(
        cls,
        session: Session,
        classifier: ShelfMembershipClassifier,
        stream_chunk_size: int = 2000,
    ) -> ShelfContents:
        """
        Find the shelves among the raw objects, and the source ids worth measuring for
        them, reading only the four columns that decide it so the whole dataset can be
        scanned cheaply.

        The raw type string has to be read here rather than recovered later, since the
        generalized object type merges bookcases and open shelves into one category.
        Deciding which source ids are relevant needs every row's ``place_id`` compared
        against the now-complete set of shelf ids, so the rows are read into memory once
        rather than streamed straight into a single comprehension.

        :param session: Session on the raw sage10k database.
        :param classifier: Decides whether a raw type string is shelf-like.
        :return: Collector primed with the shelves it has to keep and the source ids
            worth measuring for them.
        """
        from semantic_digital_twin.orm.ormatic_interface import Sage10kObjectDAO

        rows = list(
            session.execute(
                select(
                    Sage10kObjectDAO.id,
                    Sage10kObjectDAO.type,
                    Sage10kObjectDAO.place_id,
                    Sage10kObjectDAO.source_id,
                ).execution_options(yield_per=stream_chunk_size)
            )
        )
        shelf_ids = {row.id for row in rows if classifier.is_shelf_like(row.type)}
        relevant_source_ids = {
            row.source_id
            for row in rows
            if row.id in shelf_ids or row.place_id in shelf_ids
        }
        return cls(shelf_ids=shelf_ids, relevant_source_ids=relevant_source_ids)

    def collect(self, processed_object: PreprocessedObject) -> None:
        """
        Keep *processed_object* when it is a shelf or stands on one.

        :param processed_object: An object just read from the raw dataset.
        """
        if (
            processed_object.id in self.shelf_ids
            or processed_object.place_id in self.shelf_ids
        ):
            self.objects.append(processed_object)

    def shelf_bounds(self, measurements: MeshMeasurements) -> dict[str, MeshBounds]:
        """
        Measure the kept shelves' meshes, which is what locates their base and top.

        Shelves whose mesh is not cached are absent, and are skipped by
        :meth:`Sage10kPreprocessingRun._shelves_with_layers` in turn.

        :param measurements: Supplies each mesh's measurements.
        :return: The measurements, by mesh source id.
        """
        shelf_source_ids = {
            object_.source_id
            for object_ in self.objects
            if object_.id in self.shelf_ids
        }
        return {
            source_id: bounds
            for source_id in shelf_source_ids
            if (bounds := measurements.bounds(source_id)) is not None
        }


@dataclass
class BatchedRecordWriter:
    """
    Stores records one at a time, committing and detaching them periodically.

    Detaching, rather than only expiring, is what lets a stored record be released: an
    expired instance stays registered with the session and so stays alive for the whole
    run.
    """

    session: Session
    """
    Session on the processed database.
    """

    label: str
    """
    Name used in the progress output.
    """

    stored_count: int = 0
    """
    Records stored so far.
    """

    commit_batch_size: float = 500
    """
    How many records to stage before committing and detaching them.
    """

    def store(self, record: Any) -> None:
        """
        Convert *record* to its data access object and stage it for the next commit.

        :param record: The processed record to persist.
        """
        self.session.add(to_dao(record))
        self.stored_count += 1
        if self.stored_count % self.commit_batch_size == 0:
            self._commit()
            print(f"  committed {self.stored_count} {self.label}")

    def store_all(self, records: Iterable[Any]) -> None:
        """
        Store every record in *records* and commit what is left over.

        :param records: The processed records to persist.
        """
        for record in records:
            self.store(record)
        self.finish()

    def finish(self) -> None:
        """
        Commit whatever has not been committed yet and report the total.
        """
        self._commit()
        print(f"Stored {self.stored_count} {self.label}.")

    def _commit(self) -> None:
        self.session.commit()
        self.session.expunge_all()


@dataclass
class ObjectPassShardResult:
    """
    What one worker's slice of the read-convert-write object pass produced.
    """

    stored_count: int
    """
    Objects written to the processed database by this shard.
    """

    corrected_count: int
    """
    Of those, how many had their position mesh-corrected.
    """

    shelf_count: int
    """
    Shelves this shard extracted from its own rooms and wrote to the processed database.
    """

    layer_count: int
    """
    Layers across this shard's own :attr:`shelf_count` shelves.
    """


@dataclass
class PreprocessingSummary:
    """
    The object pass's totals across every shard, and the run's overall progress report
    built from them.
    """

    shard_results: list[ObjectPassShardResult]
    """
    Every shard's own counts from
    :meth:`Sage10kPreprocessingRun._process_objects_in_parallel`.
    """

    measured_source_id_count: int
    """
    How many of the shelf-relevant source ids had a mesh actually cached and measured.
    """

    relevant_source_id_count: int
    """
    How many source ids were shelf-relevant and so worth measuring at all.
    """

    total_shelf_id_count: int
    """
    How many raw objects were classified as shelf-like, whether or not extraction kept
    them.
    """

    worker_count: int
    """
    How many worker processes the object pass ran across.
    """

    elapsed_seconds: float
    """
    Wall-clock time of the whole run, from schema setup through the object pass.
    """

    @property
    def stored_count(self) -> int:
        """
        Objects written to the processed database across every shard.
        """
        return sum(result.stored_count for result in self.shard_results)

    @property
    def corrected_count(self) -> int:
        """
        Of :attr:`stored_count`, how many had their position mesh-corrected.
        """
        return sum(result.corrected_count for result in self.shard_results)

    @property
    def shelf_count(self) -> int:
        """
        Shelves extracted and written across every shard.
        """
        return sum(result.shelf_count for result in self.shard_results)

    @property
    def layer_count(self) -> int:
        """
        Layers across every extracted shelf.
        """
        return sum(result.layer_count for result in self.shard_results)

    def report(self) -> None:
        """
        Print the run's progress summary.
        """
        print(
            f"Corrected {self.corrected_count}/{self.stored_count} object positions "
            f"across {self.worker_count} workers against "
            f"{self.measured_source_id_count}/{self.relevant_source_id_count} "
            f"shelf-relevant cached meshes."
        )
        print(
            f"Extracted {self.layer_count} layers from {self.shelf_count}/"
            f"{self.total_shelf_id_count} shelves. Shelves and contents without a "
            f"cached mesh are left out, since a layer records offsets and heights that an "
            f"unmeasured mesh would falsify."
        )
        print(f"Done in {self.elapsed_seconds:.1f}s.")


@dataclass
class Sage10kPreprocessingRun:
    """
    Orchestrates one run of the sage10k preprocessing pipeline against a pair of
    database URIs: dropping and rebuilding the processed schema, discovering shelves and
    rooms, measuring shelf-relevant meshes, and running the read-convert-write object
    pass across worker processes.

    Also carries the shelf-layer extraction algorithm itself
    (:meth:`_shelves_with_layers` down through :meth:`_layers_of_shelf` and its smaller
    geometry helpers) as staticmethods: each is used only from :meth:`_process_room_shard`
    within this class's own object pass, so none of them are free functions elsewhere in
    the module.

    Holds only the run's own configuration -- URIs and worker-pool tuning -- never a
    database session or engine: :meth:`_process_objects` and :meth:`_measure_meshes`
    delegate to :meth:`_process_objects_in_parallel` and the
    :meth:`_measure_meshes_in_parallel` staticmethod, which in turn submit
    :meth:`_process_room_shard` and (via :meth:`MeshMeasurements._load_mesh_bounds`)
    per-mesh measurement to their own worker pools -- each shard builds its own
    worker-local sessions from :attr:`sage10k_database_uri`/:attr:`processed_database_uri`
    rather than sharing a connection held here. :meth:`_process_objects_in_parallel` and
    :meth:`_process_room_shard` read those URIs straight off ``self`` instead of taking
    them as parameters: a bound instance method pickles by pickling the instance behind
    it, and this dataclass holds nothing but URIs and worker-pool tuning, so it pickles
    as cleanly as the staticmethods that submit to the same
    ``spawn``-context :class:`~concurrent.futures.ProcessPoolExecutor`.
    """

    sage10k_database_uri: str
    """
    Connection string for the raw database.
    """

    processed_database_uri: str
    """
    Connection string for the processed database this run writes to.
    """

    scenes_root: Path
    """
    Root directory that contains individual scene folders, passed to
    :func:`build_source_id_to_path` to locate shelf-relevant meshes.
    """

    mesh_measurement_worker_cap: int = 128
    """
    Upper bound on parallel workers for mesh measurement, independent of the host's core
    count.

    Measurement is disk-I/O bound, so beyond a point more workers thrash the disk
    instead of finishing faster.
    """

    object_pass_worker_cap: int = 32
    """
    Upper bound on parallel workers for the read-convert-write object pass.

    The constraint here is concurrent write throughput to the processed database, not
    CPU, so this stays far below the host's core count.
    """

    mesh_measurement_chunk_size: int = 1000
    """
    Source ids handed to one worker process per round trip during parallel mesh
    measurement.

    Too small and inter-process communication dominates; too large and work balances
    poorly across workers.
    """

    stream_chunk_size: ClassVar[int] = 2000
    """
    Rows fetched per round trip while :meth:`_streamed_raw_objects` reads the raw
    dataset.

    The dataset does not fit in memory as a whole, so it is walked in chunks and each
    object is written out and let go of before the next arrives. A class attribute
    rather than a run parameter: nothing about a single run ever has reason to tune
    this independently of the others.
    """

    layer_clustering_tolerance: ClassVar[float] = 0.05
    """
    Largest height difference, in metres, between two objects :meth:`_layers_of_shelf`
    still considers to be standing on the same shelf layer.
    """

    default_edge_margin_fraction: ClassVar[float] = 0.10
    """
    Fraction of a shelf's width and length :meth:`_shelves_with_layers` keeps free at
    its edges when deciding whether an object really stands on it, so a learned layout
    never places an object where it would protrude.

    Named with the ``default_`` prefix, unlike its :attr:`layer_clustering_tolerance`
    and :attr:`stream_chunk_size` siblings, because :meth:`_shelves_with_layers` takes
    its own same-named ``edge_margin_fraction`` parameter -- naming this identically
    would shadow that parameter in its own default-value expression.
    """

    @staticmethod
    def _available_worker_count(cap: int) -> int:
        """
        How many worker processes to use for a parallel pass, bounded by *cap*.

        :param cap: The most workers ever worth using for this pass, regardless of how
            many cores the host has.
        :return: The worker count to use.
        """
        if hasattr(os, "sched_getaffinity"):
            available = len(os.sched_getaffinity(0))
        else:
            available = os.cpu_count() or 1
        return min(available, cap)

    def run(self) -> None:
        """
        Read the raw sage10k layouts and store a processed, fitting-ready copy of them
        in the processed database, then print a summary of what it wrote.

        Every correction that shelf-layout fitting used to repeat on each run --
        unifying object types, centring positions on their meshes, discarding
        objects that overhang their shelf, grouping shelf contents into layers and
        expressing their poses in the shelf's content frame -- is applied once here
        instead.

        Mesh measurement and the read-convert-write object pass each run across
        several worker processes: the object pass splits by room, since a shelf
        and everything standing on it always share one, so each shard resolves
        both its own objects and its own shelves start to finish and writes them
        to the processed database itself. Only the summary counts
        :class:`PreprocessingSummary` reports are gathered back into this process.

        The processed database is dropped and rebuilt, so a re-run replaces the
        stored dataset rather than appending a second copy of it.

        .. note::
            Safe to run as this module's own entry point (``python -m ...preprocess_sage10k``).
            The object pass spawns worker processes that re-execute whatever module ran as
            the entry point, which gives :class:`PreprocessedObject` a second identity there, distinct
            from the one the generated DAO interface maps. :func:`~krrood.ormatic.data_access_objects.helper.get_dao_class`
            resolves that back to the same DAO regardless, by falling back to matching on
            the defining file and qualified name when a straight identity match fails and
            one of the two candidates was loaded as ``__main__`` or ``__mp_main__``.

        .. note::
            The stored ``description`` and ``place_guidance`` text is deliberately
            left unscored. Running
            :class:`~semantic_digital_twin.semantic_annotations.description_matching.DescriptionCategoryScorer`
            over it is a natural enrichment of this data, but it is transformer
            inference per object over the whole dataset and belongs in its own
            opt-in stage.
        """
        start = time.time()
        self._drop_and_create_processed_schema()

        shelf_contents, room_ids = self._discover_shelves_and_rooms()
        print(f"Found {len(shelf_contents.shelf_ids)} shelves among the raw objects.")

        # Narrowed to relevant_source_ids before it reaches either the measurement pass
        # or the object pass: MeshMeasurements.bounds() lazily re-measures anything it
        # finds a path for, so leaving the full corpus in this dict would silently
        # re-measure every other cached mesh one at a time during the object pass,
        # defeating the point of scoping measurement at all.
        relevant_source_id_to_path = {
            source_id: path
            for source_id, path in build_source_id_to_path(self.scenes_root).items()
            if source_id in shelf_contents.relevant_source_ids
        }
        bounds_by_source_id, mesh_measurement_worker_count = self._measure_meshes(
            shelf_contents.relevant_source_ids, relevant_source_id_to_path
        )
        measured_count = MeshMeasurements(
            source_id_to_path=relevant_source_id_to_path,
            _bounds_by_source_id=bounds_by_source_id,
        ).measured_mesh_count
        print(
            f"Measured {measured_count}/{len(shelf_contents.relevant_source_ids)} "
            f"shelf-relevant meshes across {mesh_measurement_worker_count} workers."
        )

        object_pass_worker_count = Sage10kPreprocessingRun._available_worker_count(
            cap=self.object_pass_worker_cap
        )
        shard_results = self._process_objects(
            room_ids,
            relevant_source_id_to_path,
            bounds_by_source_id,
            shelf_contents.shelf_ids,
            object_pass_worker_count,
        )
        PreprocessingSummary(
            shard_results=shard_results,
            measured_source_id_count=measured_count,
            relevant_source_id_count=len(shelf_contents.relevant_source_ids),
            total_shelf_id_count=len(shelf_contents.shelf_ids),
            worker_count=object_pass_worker_count,
            elapsed_seconds=time.time() - start,
        ).report()

    def _drop_and_create_processed_schema(self) -> None:
        """
        Drop and recreate the processed database's schema, so a re-run replaces the
        stored dataset rather than appending a second copy of it.
        """
        from experiments.orm.ormatic_interface import Base

        processed_engine = create_engine(self.processed_database_uri)
        drop_database(processed_engine)
        Base.metadata.create_all(bind=processed_engine)

    def _discover_shelves_and_rooms(self) -> tuple[ShelfContents, list[str]]:
        """
        Find the shelves among the raw objects and every room id to distribute across
        the object-pass shards.

        Closes its own session and disposes its own engine before returning: nothing
        after this point may hold a database connection open once the worker process
        pools are created, since a connection opened here is not safe to share with a
        forked or spawned worker.

        :return: The discovered shelf contents, and every room id in the raw dataset.
        """
        from semantic_digital_twin.orm.ormatic_interface import Sage10kObjectDAO

        sage10k_engine = create_engine(self.sage10k_database_uri)
        sage10k_session = Session(sage10k_engine)
        shelf_contents = ShelfContents.from_raw_objects(
            sage10k_session, ShelfMembershipClassifier()
        )
        room_ids = list(
            sage10k_session.execute(
                select(Sage10kObjectDAO.room_id).distinct()
            ).scalars()
        )
        sage10k_session.close()
        sage10k_engine.dispose()
        return shelf_contents, room_ids

    def _measure_meshes(
        self, relevant_source_ids: set[str], source_id_to_path: dict[str, Path]
    ) -> tuple[dict[str, Optional[MeshBounds]], int]:
        """
        Measure every shelf-relevant mesh across a capped worker pool.

        :param relevant_source_ids: Source ids of the shelves and of the objects
            standing on them -- see :attr:`ShelfContents.relevant_source_ids`. Every
            other mesh in the raw dataset is left unmeasured, since nothing downstream
            of layer extraction reads its correction.
        :param source_id_to_path: Maps a mesh's source id to its cached scene directory.
        :return: The measurements, by source id, and the worker count used.
        """
        worker_count = Sage10kPreprocessingRun._available_worker_count(
            cap=self.mesh_measurement_worker_cap
        )
        bounds_by_source_id = Sage10kPreprocessingRun._measure_meshes_in_parallel(
            source_id_to_path,
            relevant_source_ids,
            worker_count,
            chunk_size=self.mesh_measurement_chunk_size,
        )
        return bounds_by_source_id, worker_count

    @staticmethod
    def _measure_meshes_in_parallel(
        source_id_to_path: dict[str, Path],
        source_ids: Iterable[str],
        worker_count: int,
        chunk_size: int = 1000,
    ) -> dict[str, Optional[MeshBounds]]:
        """
        Measure every mesh in *source_ids* across *worker_count* processes.

        Loading and parsing hundreds of thousands of mesh files is disk-I/O and CPU
        bound but embarrassingly parallel, since each mesh is measured independently of
        every other. Running it as its own pass, ahead of the object read-convert-write
        pass, is what lets that pass share one already-measured map instead of every
        worker re- measuring meshes another worker also happens to need.

        :param source_id_to_path: Maps a mesh's source id to its cached scene directory,
            as returned by :func:`build_source_id_to_path`.
        :param source_ids: The distinct source ids to measure.
        :param worker_count: How many worker processes to measure with.
        :param chunk_size: Source ids handed to one worker per round trip.
        :return: The measurements, by source id; ``None`` for a source id whose mesh is
            not cached locally.
        """
        source_id_list = list(source_ids)
        scene_directories = [
            source_id_to_path.get(source_id) for source_id in source_id_list
        ]
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            measurements = pool.map(
                MeshMeasurements._load_mesh_bounds,
                source_id_list,
                scene_directories,
                chunksize=chunk_size,
            )
        return dict(zip(source_id_list, measurements))

    def _process_objects(
        self,
        room_ids: list[str],
        source_id_to_path: dict[str, Path],
        bounds_by_source_id: dict[str, Optional[MeshBounds]],
        shelf_ids: set[str],
        worker_count: int,
    ) -> list[ObjectPassShardResult]:
        """
        Run the read-convert-write object pass across a capped worker pool.

        :return: One result per shard.
        """
        return self._process_objects_in_parallel(
            room_ids,
            source_id_to_path,
            bounds_by_source_id,
            shelf_ids,
            worker_count,
        )

    @staticmethod
    def _streamed_raw_objects(
        session: Session, room_ids: Optional[list[str]] = None
    ) -> Iterator[Sage10kObjectDAO]:
        """
        Walk raw objects, in chunks, with the pose relationships already loaded.

        :param session: Session on the raw sage10k database.
        :param room_ids: When given, only objects in these rooms are walked -- how the
            object pass is split into independent shards.
        :return: The raw objects, one at a time.
        """
        from semantic_digital_twin.orm.ormatic_interface import Sage10kObjectDAO

        statement = (
            select(Sage10kObjectDAO)
            .options(
                joinedload(Sage10kObjectDAO.position),
                joinedload(Sage10kObjectDAO.rotation),
                joinedload(Sage10kObjectDAO.dimensions),
            )
            .execution_options(yield_per=Sage10kPreprocessingRun.stream_chunk_size)
        )
        if room_ids is not None:
            statement = statement.where(Sage10kObjectDAO.room_id.in_(room_ids))
        return iter(session.scalars(statement))

    @staticmethod
    def _wrap_angle_radians(angle: float) -> float:
        """
        Wrap *angle* into the half-open interval (-pi, pi] radians.

        :param angle: Angle in radians.
        :return: The equivalent angle in (-pi, pi].
        """
        return ((angle + math.pi) % (2 * math.pi)) - math.pi

    @staticmethod
    def _is_within_shelf_footprint(
        position: Point2,
        shelf: PreprocessedObject,
        maximum_relative_x: float,
        maximum_relative_y: float,
    ) -> bool:
        """
        Whether *position* lies within the shelf's own footprint, inset by the caller's
        edge margin.

        The offset is rotated into the shelf's own frame first, since the bounds are
        expressed along the shelf's width and length. Testing the raw world-frame offset
        instead reads the wrong axes for any shelf whose orientation is not a multiple
        of 180 degrees.

        :param position: The candidate's world-frame position.
        :param shelf: The shelf the position is tested against.
        :param maximum_relative_x: Half the shelf's width, inset by the margin.
        :param maximum_relative_y: Half the shelf's length, inset by the margin.
        :return: Whether the position falls within the inset footprint.
        """
        local_offset = _rotate_offset_into_frame(
            Point2(x=position.x - shelf.pose.x, y=position.y - shelf.pose.y),
            float(shelf.pose.yaw),
        )
        return (
            abs(float(local_offset.x)) <= maximum_relative_x
            and abs(float(local_offset.y)) <= maximum_relative_y
        )

    @staticmethod
    def _dominant_object_type(objects: Iterable[PreprocessedObject]) -> ObjectType:
        """
        The object type that occurs most often among *objects*.

        Ties break on the type's own value, ascending, so the result is deterministic
        regardless of iteration order.

        :param objects: The objects to find the mode of. Must be non-empty.
        :return: The most frequent :class:`ObjectType` among *objects*.
        """
        counts = Counter(object_.object_type for object_ in objects)
        return min(
            counts, key=lambda object_type: (-counts[object_type], object_type.value)
        )

    @staticmethod
    def _object_in_content_frame(
        shelf: PreprocessedObject, object_: PreprocessedObject
    ) -> RelationalCircuitExperimentObject2D:
        """
        Express *object_*'s pose relative to *shelf* in the shelf's content frame.

        The content frame is the shelf's own yaw plus
        :attr:`RelationalCircuitExperimentShelf.content_frame_yaw_offset_degrees`, the frame
        :meth:`RelationalCircuitExperimentShelf.spawn` builds its corpus in. Storing the pose in any other frame
        makes the contents' spread land on the corpus's shallow depth axis and overflow
        front and back.

        :param shelf: The shelf the object stands on.
        :param object_: The object whose pose is converted.
        :return: The object with a shelf-relative, content-frame pose.
        """
        content_frame_yaw_radians = RelationalCircuitExperimentShelf.content_frame_yaw(
            float(shelf.pose.yaw)
        )
        local_offset = _rotate_offset_into_frame(
            Point2(x=object_.pose.x - shelf.pose.x, y=object_.pose.y - shelf.pose.y),
            content_frame_yaw_radians,
        )
        yaw_radians = Sage10kPreprocessingRun._wrap_angle_radians(
            float(object_.pose.yaw) - content_frame_yaw_radians
        )
        return RelationalCircuitExperimentObject2D(
            object_type=object_.object_type,
            scale=object_.scale,
            pose=Pose2D(x=local_offset.x, y=local_offset.y, yaw=yaw_radians),
            source_id=object_.source_id,
        )

    @staticmethod
    def _object_bottom(
        object_: PreprocessedObject, measurements: MeshMeasurements
    ) -> float:
        """
        Height at which *object_* rests, which is the height of the slab beneath it.

        Falls back to the object's own origin when its mesh is not cached, which reads
        as an object of no height rather than inventing a reach for it.

        :param object_: The object standing on a slab.
        :param measurements: Supplies the mesh's measurements.
        :return: The height of the object's underside.
        """
        bounds = measurements.bounds(object_.source_id)
        if bounds is None:
            return float(object_.pose.z)
        return float(object_.pose.z) + bounds.bottom

    @staticmethod
    def _relative_height(
        slab_height: float, base_height: float, shelf_height: float
    ) -> float:
        """
        Where a slab sits between its shelf's base and top, as a fraction.

        A shelf mesh of no measurable height leaves the fraction undefined, so it reads
        as sitting at the base rather than dividing by zero.

        :param slab_height: The slab's height in world coordinates.
        :param base_height: The shelf's base in world coordinates.
        :param shelf_height: The shelf's total height.
        :return: The fraction, zero at the base and one at the top.
        """
        if shelf_height <= 0:
            return 0.0
        return (slab_height - base_height) / shelf_height

    @staticmethod
    def _vertical_clearance(
        index: int, slab_heights: list[float], top_height: float
    ) -> float:
        """
        Space above the slab at *index*, up to the next slab or, for the topmost, the
        shelf's own top.

        :param index: Position of the slab in *slab_heights*.
        :param slab_heights: Every slab's height, lowest first.
        :param top_height: The shelf's top in world coordinates.
        :return: The clearance, never negative.
        """
        surface_above = (
            slab_heights[index + 1] if index + 1 < len(slab_heights) else top_height
        )
        return max(surface_above - slab_heights[index], 0.0)

    @staticmethod
    def _layers_of_shelf(
        shelf: PreprocessedObject,
        members: list[PreprocessedObject],
        shelf_bounds: MeshBounds,
        edge_margin_fraction: float,
        measurements: MeshMeasurements,
    ) -> list[RelationalCircuitExperimentShelfLayer]:
        """
        Group the objects standing on *shelf* into its horizontal layers, ordered from
        the bottom up and each carrying where it sits in the shelf.

        Objects are assigned to a layer by clustering their heights, so the layer
        structure comes from the arrangement itself rather than from a fixed assumption
        about how many layers a shelf has.

        Only mesh-centred positions take part. A layer records each object's offset from
        the shelf's own origin, so an uncorrected *shelf* position shifts every offset
        on it, and an uncorrected *object* position shifts that object's own. Unlike the
        object table -- which keeps uncorrected rows and marks them -- layers are
        training data whose whole content is those offsets, so admitting an uncorrected
        one would teach a circuit an arrangement nobody built.

        :param shelf: The shelf whose contents are grouped.
        :param members: Objects declaring *shelf* as the place they stand on.
        :param shelf_bounds: The shelf mesh's own measurements, whose vertical reach
            locates its base and top and so gives the layers their heights.
        :param edge_margin_fraction: Fraction of the shelf's width and length kept free
            at its edges.
        :param measurements: Supplies each object mesh's reach, which locates the slab
            an object rests on.
        :return: The shelf's layers, lowest first; empty when nothing qualifies.
        """
        if not shelf.position_is_mesh_corrected:
            return []

        maximum_relative_x = shelf.scale.y / 2 * (1 - edge_margin_fraction)
        maximum_relative_y = shelf.scale.x / 2 * (1 - edge_margin_fraction)
        within_bounds = [
            object_
            for object_ in members
            if object_.position_is_mesh_corrected
            and Sage10kPreprocessingRun._is_within_shelf_footprint(
                Point2(x=object_.pose.x, y=object_.pose.y),
                shelf,
                maximum_relative_x,
                maximum_relative_y,
            )
        ]
        if not within_bounds:
            return []

        theme_dominant_type = Sage10kPreprocessingRun._dominant_object_type(
            within_bounds
        )
        heights = np.array(
            [float(object_.pose.z) for object_ in within_bounds]
        ).reshape(-1, 1)
        labels = DBSCAN(
            eps=Sage10kPreprocessingRun.layer_clustering_tolerance, min_samples=1
        ).fit_predict(heights)

        objects_by_label: defaultdict[int, list[PreprocessedObject]] = defaultdict(list)
        for object_, label in zip(within_bounds, labels):
            objects_by_label[label].append(object_)

        ordered_groups = sorted(
            objects_by_label.values(),
            key=lambda objects: sum(float(object_.pose.z) for object_ in objects)
            / len(objects),
        )
        # The shelf's recorded position is its mesh's origin, so its real base and
        # top follow from where the mesh reaches around that origin.
        base_height = float(shelf.pose.z) + shelf_bounds.bottom
        top_height = float(shelf.pose.z) + shelf_bounds.top
        # A slab sits at the underside of what stands on it, not at those objects'
        # centres. Averaging the centres would put every slab roughly half an object
        # height too high, and since spawning places slabs at the height recorded
        # here, that error compounds on each extract-and-regenerate round trip.
        slab_heights = [
            sum(
                Sage10kPreprocessingRun._object_bottom(object_, measurements)
                for object_ in objects
            )
            / len(objects)
            for objects in ordered_groups
        ]

        return [
            RelationalCircuitExperimentShelfLayer(
                objects=[
                    Sage10kPreprocessingRun._object_in_content_frame(shelf, object_)
                    for object_ in layer_objects
                ],
                theme_dominant_type=theme_dominant_type,
                height_above_shelf_base=slab_height - base_height,
                relative_height=Sage10kPreprocessingRun._relative_height(
                    slab_height, base_height, shelf_bounds.height
                ),
                vertical_clearance=Sage10kPreprocessingRun._vertical_clearance(
                    index, slab_heights, top_height
                ),
            )
            for index, (layer_objects, slab_height) in enumerate(
                zip(ordered_groups, slab_heights)
            )
        ]

    @staticmethod
    def _shelves_with_layers(
        objects: list[PreprocessedObject],
        bounds_by_source_id: dict[str, MeshBounds],
        shelf_ids: set[str],
        measurements: MeshMeasurements,
        edge_margin_fraction: float = default_edge_margin_fraction,
    ) -> list[RelationalCircuitExperimentShelf]:
        """
        Build one :class:`RelationalCircuitExperimentShelf` per shelf that holds
        something, carrying its own pose and its layers in order from the bottom up.

        Shelf membership comes from an object's ``place_id`` naming the shelf it stands
        on, rather than from spatial containment. Keeping the shelf itself, rather than
        loose layers, is what preserves both that grouping and the layers' order -- and
        lets a caller draw how many layers a generated shelf should have from the real
        distribution.

        Objects whose position could not be centred on their mesh are left out; see
        :meth:`_layers_of_shelf`. An object that is itself classified as a shelf-like
        parent is also left out of *another* shelf's contents -- the raw dataset records
        a smaller piece of shelf-like furniture standing on a bigger one this way, and
        counting it as ordinary content teaches the circuit that shelves commonly hold
        other shelves.

        :param objects: Processed objects.
        :param bounds_by_source_id: Each shelf mesh's own measurements, by source id. A
            shelf with no entry is skipped, since its layers' heights would be
            guesswork.
        :param shelf_ids: Ids of the raw objects classified as shelf-like; an object
            absent from it is not treated as a shelf, and one present in it is never
            treated as another shelf's content.
        :param measurements: Supplies each object mesh's reach, used to locate slabs.
        :param edge_margin_fraction: Fraction of each shelf's width and length kept free
            at its edges.
        :return: The shelves that hold at least one layer.
        """
        objects_by_place_id: defaultdict[str, list[PreprocessedObject]] = defaultdict(
            list
        )
        for object_ in objects:
            if object_.id in shelf_ids:
                continue
            objects_by_place_id[object_.place_id].append(object_)

        shelves = []
        for shelf in objects:
            if shelf.id not in shelf_ids or not objects_by_place_id[shelf.id]:
                continue
            shelf_bounds = bounds_by_source_id.get(shelf.source_id)
            if shelf_bounds is None:
                continue
            layers = Sage10kPreprocessingRun._layers_of_shelf(
                shelf,
                objects_by_place_id[shelf.id],
                shelf_bounds,
                edge_margin_fraction,
                measurements,
            )
            if not layers:
                continue
            shelves.append(
                RelationalCircuitExperimentShelf(
                    scale=Scale(
                        x=shelf.scale.x,
                        y=shelf.scale.y,
                        z=shelf_bounds.height,
                    ),
                    layers=layers,
                    theme_dominant_type=layers[0].theme_dominant_type,
                )
            )
        return shelves

    @staticmethod
    def _partition_round_robin(
        items: list[str], partition_count: int
    ) -> list[list[str]]:
        """
        Split *items* into *partition_count* roughly equal, interleaved groups.

        Round-robin, rather than contiguous slices, so an ordering correlated with how
        busy a room is (if any) does not concentrate onto one shard.

        :param items: The items to split.
        :param partition_count: How many groups to split into.
        :return: The groups, each in *items*'s original relative order.
        """
        partitions: list[list[str]] = [[] for _ in range(partition_count)]
        for index, item in enumerate(items):
            partitions[index % partition_count].append(item)
        return partitions

    def _process_room_shard(
        self,
        room_ids: list[str],
        source_id_to_path: dict[str, Path],
        bounds_by_source_id: dict[str, Optional[MeshBounds]],
        shelf_ids: set[str],
        shard_label: str,
    ) -> ObjectPassShardResult:
        """
        Read, convert and write every object in *room_ids*, then extract and store the
        shelves among them, all in one worker process.

        A shelf and everything standing on it always share a room, so a shard that owns
        a room already owns everything shelf extraction needs for it -- nothing has to
        travel back to the parent process, unlike the objects and shelves themselves,
        which this shard writes straight to the processed database through its own
        session.

        Runs in its own, freshly spawned process with its own database connections: a
        connection opened in the parent is not safe to share across processes, so this
        builds everything it needs from scratch rather than inheriting anything from the
        parent.

        :param room_ids: The rooms this shard is responsible for.
        :param source_id_to_path: Maps a mesh's source id to its cached scene directory.
        :param bounds_by_source_id: Every mesh's measurements, computed once by
            :meth:`_measure_meshes_in_parallel` ahead of the object pass, so no shard
            measures a mesh another shard also happens to need.
        :param shelf_ids: Ids of the raw objects classified as shelf-like.
        :param shard_label: Distinguishes this shard's progress output from the other
            shards running alongside it.
        :return: This shard's counts.
        """
        import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers

        sage10k_session = Session(create_engine(self.sage10k_database_uri))
        processed_session = Session(create_engine(self.processed_database_uri))
        measurements = MeshMeasurements(
            source_id_to_path=source_id_to_path,
            _bounds_by_source_id=dict(bounds_by_source_id),
        )
        classifier = ObjectTypeClassifier()
        shelf_contents = ShelfContents(shelf_ids=shelf_ids)

        object_writer = BatchedRecordWriter(
            session=processed_session, label=f"objects[{shard_label}]"
        )
        corrected_count = 0
        for sage10k_object in Sage10kPreprocessingRun._streamed_raw_objects(
            sage10k_session, room_ids
        ):
            processed_object = PreprocessedObject.from_sage10k_object(
                sage10k_object, classifier, measurements
            )
            corrected_count += processed_object.position_is_mesh_corrected
            shelf_contents.collect(processed_object)
            object_writer.store(processed_object)
        object_writer.finish()

        shelves = Sage10kPreprocessingRun._shelves_with_layers(
            shelf_contents.objects,
            shelf_contents.shelf_bounds(measurements),
            shelf_contents.shelf_ids,
            measurements,
        )
        # Layers are stored through their shelf, so the grouping and the
        # bottom-to-top order survive; they remain queryable in their own right.
        BatchedRecordWriter(
            session=processed_session, label=f"shelves[{shard_label}]"
        ).store_all(shelves)

        return ObjectPassShardResult(
            stored_count=object_writer.stored_count,
            corrected_count=corrected_count,
            shelf_count=len(shelves),
            layer_count=sum(len(shelf.layers) for shelf in shelves),
        )

    def _process_objects_in_parallel(
        self,
        room_ids: list[str],
        source_id_to_path: dict[str, Path],
        bounds_by_source_id: dict[str, Optional[MeshBounds]],
        shelf_ids: set[str],
        worker_count: int,
    ) -> list[ObjectPassShardResult]:
        """
        Read, convert and write every raw object across *worker_count* worker processes,
        each responsible for a disjoint slice of rooms, storing both the objects and the
        shelves it finds among them.

        A shelf and everything standing on it always share a room, so splitting on room
        id is what lets each shard resolve its own objects and its own shelves start to
        finish, with no coordination -- and no data -- passed back to this process
        beyond counts.

        :param room_ids: Every room id to distribute across shards.
        :param source_id_to_path: Maps a mesh's source id to its cached scene directory.
        :param bounds_by_source_id: Every mesh's measurements, computed ahead of this
            pass.
        :param shelf_ids: Ids of the raw objects classified as shelf-like.
        :param worker_count: How many worker processes to split the work across.
        :return: One result per shard.
        """
        shards = [
            shard
            for shard in Sage10kPreprocessingRun._partition_round_robin(
                room_ids, worker_count
            )
            if shard
        ]
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(
                    self._process_room_shard,
                    shard,
                    source_id_to_path,
                    bounds_by_source_id,
                    shelf_ids,
                    f"shard {index + 1}/{len(shards)}",
                )
                for index, shard in enumerate(shards)
            ]
            return [future.result() for future in futures]


if __name__ == "__main__":
    sage10k_database_uri = os.environ.get("SAGE10k_DATABASE_URI")
    processed_database_uri = os.environ.get("SAGE10K_PROCESSED_DATABASE_URI")
    scenes_root = os.environ.get("SAGE10K_SCENES_ROOT")
    assert (
        sage10k_database_uri is not None
    ), "Please set the SAGE10k_DATABASE_URI environment variable."
    assert (
        processed_database_uri is not None
    ), "Please set the SAGE10K_PROCESSED_DATABASE_URI environment variable."
    assert (
        scenes_root is not None
    ), "Please set the SAGE10K_SCENES_ROOT environment variable."

    Sage10kPreprocessingRun(
        sage10k_database_uri=sage10k_database_uri,
        processed_database_uri=processed_database_uri,
        scenes_root=Path(scenes_root),
    ).run()
