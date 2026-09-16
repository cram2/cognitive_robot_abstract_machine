from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional

from semantic_digital_twin.world_description.geometry import Scale


# %% object type
class ObjectType(StrEnum):
    """
    Generalized object categories that unify the tens of thousands of distinct, near-
    instance-specific ``object_type`` strings found in the raw sage10k dataset (for
    example ``"book1"``, ``"book_table2"`` and ``"bookchair8eba7fdc"`` all belong to the
    same real-world category of object).

    .. note::
        Mapping the raw sage10k strings onto these generalized members is
        handled separately; this enum only defines the target categories.
    """

    APPAREL = "apparel"
    ART = "art"
    BAG = "bag"
    BASKET = "basket"
    BATHTUB = "bathtub"
    BED = "bed"
    BENCH = "bench"
    BIN = "bin"
    BOOK = "book"
    BOTTLE = "bottle"
    BOWL = "bowl"
    BOX = "box"
    BUCKET = "bucket"
    CABINET = "cabinet"
    CAMERA = "camera"
    CANDLE = "candle"
    CART = "cart"
    CHAIR = "chair"
    CHANDELIER = "chandelier"
    CLOCK = "clock"
    COMPUTER = "computer"
    CONTAINER = "container"
    COUNTER = "counter"
    CRATE = "crate"
    CUP = "cup"
    CUTLERY = "cutlery"
    CUTTING_BOARD = "cutting_board"
    DESK = "desk"
    DISHWASHER = "dishwasher"
    DISPENSER = "dispenser"
    DISPLAYCASE = "displaycase"
    DOOR = "door"
    DRESSER = "dresser"
    DRYER = "dryer"
    FIREPLACE = "fireplace"
    FOOD = "food"
    FOUNTAIN = "fountain"
    FRAME = "frame"
    GLASS = "glass"
    HARDWARE = "hardware"
    JAR = "jar"
    KEYBOARD = "keyboard"
    KNIFE = "knife"
    LADDER = "ladder"
    LAMP = "lamp"
    LIGHT_FIXTURE = "light_fixture"
    LOCKER = "locker"
    MAGAZINE = "magazine"
    MICROWAVE = "microwave"
    MIRROR = "mirror"
    MONITOR = "monitor"
    MOUSE = "mouse"
    NEON_SIGN = "neon_sign"
    NIGHTSTAND = "nightstand"
    OFFICE_SUPPLY = "office_supply"
    OTHER = "other"
    OVEN = "oven"
    PANEL = "panel"
    PANTRY = "pantry"
    PEDESTAL = "pedestal"
    PEGBOARD = "pegboard"
    PEN = "pen"
    PERSONAL_CARE_PRODUCT = "personal_care_product"
    PHONE = "phone"
    PILLOW = "pillow"
    PLANT = "plant"
    PLATE = "plate"
    POT = "pot"
    PRINTER = "printer"
    PROJECTOR = "projector"
    REFRIGERATOR = "refrigerator"
    REMOTE_CONTROL = "remote_control"
    RETAIL_FIXTURE = "retail_fixture"
    SAFETY_EQUIPMENT = "safety_equipment"
    SCULPTURE = "sculpture"
    SHELF = "shelf"
    SIDEBOARD = "sideboard"
    SIGN = "sign"
    SINK = "sink"
    SMALL_APPLIANCE = "small_appliance"
    SOFA = "sofa"
    SPEAKER = "speaker"
    SPORTS_EQUIPMENT = "sports_equipment"
    STAND = "stand"
    TABLE = "table"
    TAPESTRY = "tapestry"
    TELEVISION = "television"
    TEXTILE = "textile"
    TOILET = "toilet"
    TOOL = "tool"
    TOOLBOX = "toolbox"
    TOWEL = "towel"
    TOY = "toy"
    TRASH = "trash"
    TRAY = "tray"
    UTENSIL = "utensil"
    VANITY = "vanity"
    VASE = "vase"
    VEHICLE = "vehicle"
    VENT = "vent"
    WARDROBE = "wardrobe"
    WASHING_MACHINE = "washing_machine"
    WINDOW = "window"
    WORKBENCH = "workbench"


# %% mesh candidate pool
@dataclass(frozen=True)
class MeshCandidate:
    """
    A mesh asset available for rendering a sampled object, together with the generalized
    object type it was captured from.
    """

    scene_directory: Path
    """
    Directory containing the ``objects/`` sub-folder with this mesh's PLY and texture
    files.
    """

    source_id: str
    """
    Identifier used to look up this mesh's PLY and texture files within
    :attr:`scene_directory`.
    """

    object_type: ObjectType
    """
    The generalized category of the object this mesh was captured from.
    """

    native_extents: Optional[tuple[float, float, float]] = None
    """
    The mesh's own real-world size as ``(width, length, height)``, used to decide
    whether it fits a target space.

    ``None`` when the size is unknown, in which case the candidate is treated as always
    fitting.
    """

    @property
    def footprint_scale(self) -> Optional[Scale]:
        """
        :attr:`native_extents`, re-ordered from ``(width, length, height)`` onto the
        ``(x, y, z)`` axis convention :class:`RelationalCircuitExperimentObject2D.scale`
        uses.

        ``None`` when :attr:`native_extents` is ``None``.
        """
        if self.native_extents is None:
            return None
        width, length, height = self.native_extents
        return Scale(x=length, y=width, z=height)


@dataclass
class _MeshTypeMatcher:
    """
    Selects, from a pool of candidate meshes, a random one captured from an object of
    the same :class:`ObjectType`.

    Object-type labels in the source dataset are effectively per-
    instance identifiers rather than real categories, so grouping meshes by
    their already-generalized :class:`ObjectType` -- rather than matching
    declared size -- is what keeps a randomly-drawn mesh semantically
    plausible for the category an object was sampled as.

    .. note::
        If the pool holds no mesh of the requested type, ``None`` is returned
        rather than a mesh of some other type. Substituting was what strewed
        generated rooms with arbitrary objects: the cache holds only a few
        hundred floor-capable meshes across dozens of types, so a sampled bed or
        sofa routinely became whichever mesh happened to be drawn.
    """

    candidates: list[MeshCandidate]
    """
    Pool of meshes to choose from.
    """

    def random_match(
        self,
        object_type: ObjectType,
        max_extents: Optional[Scale] = None,
        target_extents: Optional[Scale] = None,
    ) -> Optional[MeshCandidate]:
        """
        Return a candidate whose :attr:`MeshCandidate.object_type` equals *object_type*,
        or ``None`` when the pool holds none.

        *max_extents* is an upper bound: candidates larger than it on any axis are
        ineligible, which is how shelf contents are kept from piercing the layer above.
        *target_extents* is a size to aim for: candidates further than
        :attr:`MAXIMUM_SIZE_RATIO` from it on any axis are ineligible, and the closest
        remaining one is returned rather than a random one.

        :param object_type: The category of the object a mesh is selected for.
        :param max_extents: Upper bound on the mesh's width/length/height, as ``(length,
            width, height)`` on ``(x, y, z)``.
        :param target_extents: Size the mesh should match as closely as possible, in the
            same axis convention as *max_extents*.
        :return: The selected candidate, or ``None`` when nothing is eligible.
        """
        pool = [
            candidate
            for candidate in self.candidates
            if candidate.object_type == object_type
        ]
        if max_extents is not None:
            pool = [
                candidate for candidate in pool if self._fits(candidate, max_extents)
            ]
        if target_extents is None:
            return random.choice(pool) if pool else None

        scored = [
            (self._size_mismatch(candidate, target_extents), candidate)
            for candidate in pool
        ]
        eligible = [
            (mismatch, candidate)
            for mismatch, candidate in scored
            if mismatch <= math.log(2.0)
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda scored_candidate: scored_candidate[0])[1]

    @staticmethod
    def _size_mismatch(candidate: MeshCandidate, target_extents: Scale) -> float:
        """
        How far *candidate*'s real size is from *target_extents*, as the largest
        absolute log-ratio across the three axes.

        A log-ratio is used so that being twice too large and half too large count
        equally. Candidates of unknown size score as a perfect match, since there is
        nothing to judge them on and dropping them would thin an already sparse pool.

        :param candidate: The mesh candidate to score.
        :param target_extents: The size the mesh should match, as ``(length, width,
            height)`` on ``(x, y, z)``.
        :return: The mismatch, zero being an exact match.
        """
        footprint = candidate.footprint_scale
        if footprint is None:
            return 0.0
        measured_and_targets = (
            (footprint.x, target_extents.x),
            (footprint.y, target_extents.y),
            (footprint.z, target_extents.z),
        )
        return max(
            abs(math.log(measured / target))
            for measured, target in measured_and_targets
            if measured > 0 and target > 0
        )

    @staticmethod
    def _fits(candidate: MeshCandidate, max_extents: Scale) -> bool:
        """
        Whether *candidate*'s real-world size stays within *max_extents* on every axis.
        Candidates of unknown size are treated as fitting.

        :param candidate: The mesh candidate to test.
        :param max_extents: Per-axis upper bound, as ``(length, width, height)`` on
            ``(x, y, z)``.
        :return:``True`` if the candidate fits or its size is unknown.
        """
        footprint = candidate.footprint_scale
        if footprint is None:
            return True
        return (
            footprint.x <= max_extents.x
            and footprint.y <= max_extents.y
            and footprint.z <= max_extents.z
        )


def build_source_id_to_path(
    scenes_root: Path = Path.home() / "Documents" / "sage-10k-scenes",
) -> dict[str, Path]:
    """
    Scan *scenes_root* and return a mapping from source_id to its scene directory.

    Each scene directory is expected to contain an ``objects/`` sub- folder with files
    named ``{source_id}.ply``.

    :param scenes_root: Root directory that contains individual scene folders.
    :return:``{source_id: scene_directory}`` for every PLY file found under any scene.
    """
    mapping: dict[str, Path] = {}
    for scene_directory in scenes_root.iterdir():
        objects_directory = scene_directory / "objects"
        if not objects_directory.is_dir():
            continue
        for ply_file in objects_directory.glob("*.ply"):
            texture_file = objects_directory / f"{ply_file.stem}_texture.png"
            if texture_file.exists():
                mapping[ply_file.stem] = scene_directory
    return mapping
