from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from experiments.shelf_generation_experiments.utils import ObjectType


# %% object type classification
@dataclass(frozen=True)
class ClassificationRule:
    """
    Maps one :class:`ObjectType` category onto the keywords that identify it in a raw
    sage10k ``object_type`` string.
    """

    object_type: ObjectType
    """
    The category this rule's keywords identify.
    """

    keywords: tuple[str, ...]
    """
    Substrings that, when found in a normalized raw type string, identify
    :attr:`object_type`.
    """


@dataclass(frozen=True)
class ObjectTypeClassifier:
    """
    Maps the free-form, near-instance-specific ``object_type`` strings found in the raw
    sage10k dataset (e.g. ``"book2"``, ``"bookchair8eba7fdc"``) onto the generalized
    :class:`ObjectType` categories.

    Matching is a case-insensitive, ordered keyword lookup: the raw string is tested
    against each rule's keywords in turn, and the first rule with a matching keyword
    wins. Furniture/surface categories (shelf, table, desk, ...) are checked before
    small-item categories, since the dataset frequently names an item together with the
    furniture it sits on (e.g. ``"bookshelf"``, ``"candletable"``) and the furniture is
    usually the more useful category for scene- layout purposes. This is a best-effort
    heuristic, not a guaranteed- correct classification -- raw strings that combine two
    plausible categories in an unusual order may be mapped to the "wrong" one.
    """

    _rules: ClassVar[tuple[ClassificationRule, ...]] = (
        # -- Furniture -----------------------------------------------------
        ClassificationRule(ObjectType.WORKBENCH, ("workbench",)),
        ClassificationRule(ObjectType.DISPLAYCASE, ("displaycase", "showcase")),
        ClassificationRule(ObjectType.WARDROBE, ("wardrobe", "closet")),
        ClassificationRule(ObjectType.DRESSER, ("dresser",)),
        ClassificationRule(ObjectType.LOCKER, ("locker",)),
        ClassificationRule(ObjectType.PANTRY, ("pantry",)),
        ClassificationRule(ObjectType.VANITY, ("vanity",)),
        ClassificationRule(ObjectType.NIGHTSTAND, ("nightstand",)),
        ClassificationRule(ObjectType.SIDEBOARD, ("sideboard", "console", "credenza")),
        ClassificationRule(ObjectType.SHELF, ("shelf", "shelv", "rack", "bookcase")),
        ClassificationRule(ObjectType.CABINET, ("cabinet",)),
        ClassificationRule(ObjectType.DESK, ("desk",)),
        ClassificationRule(ObjectType.COUNTER, ("counter", "countertop")),
        ClassificationRule(ObjectType.SOFA, ("sofa", "couch")),
        ClassificationRule(ObjectType.BENCH, ("bench",)),
        ClassificationRule(ObjectType.BED, ("bed", "crib")),
        ClassificationRule(
            ObjectType.CHAIR,
            ("chair", "stool", "armchair", "ottoman", "pouf", "barstool"),
        ),
        ClassificationRule(ObjectType.TABLE, ("table", "island")),
        ClassificationRule(ObjectType.CART, ("cart", "trolley")),
        ClassificationRule(ObjectType.CRATE, ("crate", "pallet")),
        ClassificationRule(ObjectType.TOOLBOX, ("toolbox",)),
        ClassificationRule(ObjectType.PEDESTAL, ("pedestal", "podium", "plinth")),
        ClassificationRule(
            ObjectType.STAND,
            ("stand", "holder", "hanger", "easel", "coatrack", "clothingrack"),
        ),
        # -- Plants (checked early: "pot" and "table" are common substrings of
        # "pottedplant"/"planttable"-style compounds, and the plant is the more
        # useful category for those) --------------------------------------
        ClassificationRule(
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
        ClassificationRule(ObjectType.CUTTING_BOARD, ("cuttingboard", "cutting_board")),
        ClassificationRule(ObjectType.DISHWASHER, ("dishwasher",)),
        ClassificationRule(
            ObjectType.REFRIGERATOR, ("fridge", "refrigerator", "freezer")
        ),
        ClassificationRule(ObjectType.SINK, ("sink",)),
        ClassificationRule(ObjectType.OVEN, ("oven", "stove")),
        ClassificationRule(ObjectType.MICROWAVE, ("microwave",)),
        ClassificationRule(
            ObjectType.SMALL_APPLIANCE,
            ("toaster", "coffeemaker", "kettle", "blender"),
        ),
        ClassificationRule(ObjectType.DISPENSER, ("dispenser",)),
        ClassificationRule(
            ObjectType.CUTLERY, ("cutlery", "fork", "spoon", "spatula", "rollingpin")
        ),
        ClassificationRule(ObjectType.KNIFE, ("knife",)),
        ClassificationRule(ObjectType.CUP, ("cup", "mug", "tumbler", "teacup")),
        ClassificationRule(ObjectType.GLASS, ("glass", "wineglass")),
        ClassificationRule(ObjectType.PLATE, ("plate",)),
        ClassificationRule(ObjectType.BOWL, ("bowl",)),
        ClassificationRule(ObjectType.BOTTLE, ("bottle",)),
        ClassificationRule(ObjectType.JAR, ("jar", "shaker", "spicejar")),
        ClassificationRule(ObjectType.UTENSIL, ("utensil",)),
        ClassificationRule(ObjectType.POT, ("pot", "peppergrinder")),
        ClassificationRule(ObjectType.TRAY, ("tray",)),
        # -- Lighting --------------------------------------------------------
        ClassificationRule(ObjectType.CHANDELIER, ("chandelier",)),
        ClassificationRule(ObjectType.NEON_SIGN, ("neon",)),
        ClassificationRule(
            ObjectType.CANDLE, ("candle", "candelabra", "candlestick", "lantern")
        ),
        ClassificationRule(ObjectType.LAMP, ("lamp",)),
        ClassificationRule(
            ObjectType.LIGHT_FIXTURE,
            ("light", "sconce", "fixture", "pendant", "ledstrip", "lightstrip"),
        ),
        # -- Electronics (checked before decor/art: "printer" and
        # "smartphone" would otherwise match ART's "print"/"art" substrings)
        # ---------------------------------------------------------------
        ClassificationRule(ObjectType.TELEVISION, ("tv", "television")),
        ClassificationRule(ObjectType.PROJECTOR, ("projector",)),
        ClassificationRule(ObjectType.COMPUTER, ("computer", "laptop")),
        ClassificationRule(ObjectType.KEYBOARD, ("keyboard",)),
        ClassificationRule(ObjectType.MOUSE, ("mouse",)),
        ClassificationRule(ObjectType.MONITOR, ("monitor", "screen")),
        ClassificationRule(ObjectType.CAMERA, ("camera",)),
        ClassificationRule(ObjectType.SPEAKER, ("speaker",)),
        ClassificationRule(ObjectType.PHONE, ("phone", "smartphone")),
        ClassificationRule(ObjectType.PRINTER, ("printer",)),
        ClassificationRule(ObjectType.REMOTE_CONTROL, ("remote", "controller")),
        # -- Decor / art -------------------------------------------------------
        ClassificationRule(ObjectType.MIRROR, ("mirror",)),
        ClassificationRule(ObjectType.CLOCK, ("clock",)),
        ClassificationRule(
            ObjectType.SCULPTURE,
            ("sculpture", "figurine", "statue", "bust", "mannequin"),
        ),
        ClassificationRule(ObjectType.VASE, ("vase", "urn", "planter")),
        ClassificationRule(
            ObjectType.TAPESTRY, ("tapestry", "wallhanging", "banner", "flag")
        ),
        ClassificationRule(ObjectType.FRAME, ("frame", "pictureframe")),
        ClassificationRule(ObjectType.PEGBOARD, ("pegboard",)),
        ClassificationRule(
            ObjectType.SIGN,
            ("sign", "menuboard", "whiteboard", "blackboard", "chart", "map"),
        ),
        ClassificationRule(
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
        ClassificationRule(
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
        ClassificationRule(
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
        ClassificationRule(ObjectType.PEN, ("pen", "pencil", "crayon", "quill")),
        ClassificationRule(
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
        ClassificationRule(ObjectType.TOILET, ("toilet",)),
        ClassificationRule(ObjectType.BATHTUB, ("bathtub", "shower")),
        ClassificationRule(ObjectType.TOWEL, ("towel", "napkin")),
        ClassificationRule(
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
        ClassificationRule(
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
        ClassificationRule(
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
        ClassificationRule(ObjectType.LADDER, ("ladder",)),
        ClassificationRule(
            ObjectType.SAFETY_EQUIPMENT, ("extinguisher", "smokedetector", "firealarm")
        ),
        # -- Containers ----------------------------------------------------------
        ClassificationRule(ObjectType.TRASH, ("trash", "waste")),
        ClassificationRule(ObjectType.BASKET, ("basket",)),
        ClassificationRule(ObjectType.BIN, ("bin",)),
        ClassificationRule(ObjectType.BOX, ("box",)),
        ClassificationRule(ObjectType.BUCKET, ("bucket",)),
        ClassificationRule(
            ObjectType.CONTAINER,
            ("container", "case", "can", "barrel", "tub", "trunk", "caddy"),
        ),
        # -- Structural / architectural --------------------------------------
        ClassificationRule(ObjectType.WINDOW, ("window",)),
        ClassificationRule(ObjectType.DOOR, ("door",)),
        ClassificationRule(ObjectType.FIREPLACE, ("fireplace",)),
        ClassificationRule(ObjectType.VENT, ("vent", "radiator")),
        ClassificationRule(
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
        ClassificationRule(ObjectType.PILLOW, ("pillow", "cushion")),
        ClassificationRule(
            ObjectType.TEXTILE, ("textile", "fabric", "rug", "carpet", "blanket")
        ),
        # -- Misc ---------------------------------------------------------------
        ClassificationRule(ObjectType.APPAREL, ("shoe", "watch", "glasses")),
        ClassificationRule(
            ObjectType.SPORTS_EQUIPMENT,
            ("dumbbell", "treadmill", "elliptical", "kettlebell"),
        ),
        ClassificationRule(ObjectType.VEHICLE, ("car", "bike", "tire")),
        ClassificationRule(
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
        ClassificationRule(ObjectType.TOY, ("toy",)),
        ClassificationRule(ObjectType.WASHING_MACHINE, ("washingmachine", "washer")),
        ClassificationRule(ObjectType.DRYER, ("dryer",)),
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
        for rule in self._rules:
            if any(keyword in normalized for keyword in rule.keywords):
                return rule.object_type
        return ObjectType.OTHER


# %% shelf membership classification
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
    :attr:`~experiments.shelf_generation_experiments.shelf_schema.RelationalCircuitExperimentShelf.theme_dominant_type`,
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
