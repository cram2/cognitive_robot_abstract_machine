from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import takewhile
from typing import Optional

import enchant

from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)

logger = logging.getLogger(__name__)

from nltk.stem import WordNetLemmatizer


@dataclass(eq=False)
class NaturalLanguageDescriptionWithTypeDescription(NaturalLanguageDescription):
    """
    A natural language description of a Sage10k object including the type information of the object.
    """

    type_description: Optional[str] = field(default=None)
    """
    The cleaned description of the type of the object.
    """


@dataclass
class Sage10kTypeNameCleaner:
    """
    Clean type names from the Sage10k dataset.
    """

    word_dictionary: enchant.Dict = field(default_factory=lambda: enchant.Dict("en_US"))
    """
    The word dictionary used to check if a type name is valid.
    """

    lemmatizer: WordNetLemmatizer = field(default_factory=lambda: WordNetLemmatizer())
    """
    The lemmatizer used to convert type names to singular form.
    """

    def clean(self, type_name: str) -> Optional[str]:
        """
        Clean a type name from the sage 10k dataset.
        Types are cleaned by removing non-alphabetic characters.
        Valid type names are types that are in:
            - Singular form
            - The word dictionary

        :param type_name: The type name to clean.
        :return: The cleaned type name or None if the type name is invalid.
        """

        cleaned_type = " ".join(
            "".join(takewhile(str.isalpha, word)) for word in type_name.split("_")
        ).title()

        if not cleaned_type:
            return None
        if not self.word_dictionary.check(cleaned_type):
            return None

        singular = self.lemmatizer.lemmatize(cleaned_type, pos="n")

        # skip plural terms
        if singular != cleaned_type:
            return None

        return cleaned_type


from enum import StrEnum


class Sage10kNaturalLanguageCategory(StrEnum):
    """
    Words we can process with sage10k and wordnet
    """


class Sage10kGraspable(Sage10kNaturalLanguageCategory):
    """
    Words we can grasp
    """

    MANUAL = "Manual"
    VOLUME = "Volume"
    NOTEBOOK = "Notebook"
    SCROLL = "Scroll"
    PAPERBACK = "Paperback"
    MAGAZINE = "Magazine"
    BOOK = "Book"
    TOME = "Tome"
    CODEX = "Codex"
    COOKBOOK = "Cookbook"
    TOOTHPASTE = "Toothpaste"
    TOOTHBRUSH = "Toothbrush"
    RULER = "Ruler"
    TRAY = "Tray"
    CHARGER = "Charger"
    SCREWDRIVER = "Screwdriver"
    HAMMER = "Hammer"
    DRILL = "Drill"
    SAW = "Saw"
    TOOL = "Tool"
    WRENCH = "Wrench"
    BRUSH = "Brush"
    HAIRBRUSH = "Hairbrush"
    COMB = "Comb"
    CUP = "Cup"
    PLATE = "Plate"
    GLASS = "Glass"
    MUG = "Mug"
    UTENSIL = "Utensil"
    BOWL = "Bowl"
    POT = "Pot"
    TEAPOT = "Teapot"
    TEAKETTLE = "Teakettle"
    HANGER = "Hanger"
    SMARTPHONE = "Smartphone"
    MOBILE = "Mobile"
    HOLDER = "Holder"
    PERFUME = "Perfume"
    TOOLKIT = "Toolkit"
    LOTION = "Lotion"
    SYRINGE = "Syringe"
    MOUSE = "Mouse"
    GLOVE = "Glove"
    KEY = "Key"
    PEN = "Pen"
    FOLIO = "Folio"
    SHOE = "Shoe"
    SPEAKER = "Speaker"
    VASE = "Vase"
    URN = "Urn"
    TUBE = "Tube"
    TABLET = "Tablet"
    NAPKIN = "Napkin"
    EARBUDS = "Earbuds"
    HEADPHONE = "Headphone"
    PHONE = "Phone"
    KEYBOARD = "Keyboard"
    PHOTO = "Photo"
    FLOSS = "Floss"
    TAPE = "Tape"
    STAPLER = "Stapler"
    STATUE = "Statue"
    SCULPTURE = "Sculpture"
    FIGURINE = "Figurine"
    ORGANIZER = "Organizer"
    REMOTE = "Remote"
    LAPTOP = "Laptop"
    CUSHION = "Cushion"
    PILLOW = "Pillow"
    BOTTLE = "Bottle"
    PAPERCLIP = "Paperclip"
    CANDLE = "Candle"
    PAINTBRUSH = "Paintbrush"
    DIAPER = "Diaper"


class Sage10kNotGraspable(Sage10kNaturalLanguageCategory):
    """
    Words we cannot grasp
    """

    BASKET = "Basket"
    MEDIA = "Media"
    TOASTER = "Toaster"
    DECORATIVE = "Decorative"
    SPROUT = "Sprout"
    LAMP = "Lamp"
    PLANT = "Plant"
    COFFEEMAKER = "Coffeemaker"
    GLASSES = "Glasses"
    DISPENSER = "Dispenser"
    SHELF = "Shelf"
    SPANNER = "Spanner"
    SUPPLY = "Supply"
    WATCH = "Watch"
    MONITOR = "Monitor"
    PALETTE = "Palette"
    PEND = "Pend"
    REMOTES = "Remotes"
    COASTER = "Coaster"
    PLIERS = "Pliers"
    BOOKSHELF = "Bookshelf"
    CUTLERY = "Cutlery"
    CLOCK = "Clock"
    SINK = "Sink"
    BOOKS = "Books"
    SUCCULENT = "Succulent"
    RACK = "Rack"
    TRASHCAN = "Trashcan"
    HEADPHONES = "Headphones"


class Sage10kNonShittyScenes(StrEnum):
    GYM = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_171403_layout_26384448.zip"
    TV_STUDIO = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_180931_layout_d83fc25f.zip"
    CRAFTSMAN_LOBBY = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_205353_layout_9584241f.zip"
    FRENCH_CLOTHING_STORE = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251214_161815_layout_3644a72f.zip"
    TROPICAL_WAREHOUSE = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251214_182016_layout_a72cf11f.zip"
    VAPORWAVE = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_090236_layout_7e07a47a.zip"
    ECLECTIC_RESIDENCE = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_090413_layout_d59e4e4b.zip"
    SOUTHWESTERN_STORE = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_123747_layout_2d89d0a5.zip"
    BRUTALIST_STORE = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_153933_layout_50ffb500.zip"
