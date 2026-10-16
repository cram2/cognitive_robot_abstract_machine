"""
The faces a video writes in, found the same way on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum, auto

from matplotlib import font_manager
from PIL import ImageFont


class FontFace(Enum):
    """
    The faces a typeface sets text in.
    """

    REGULAR = auto()
    """
    Running text.
    """

    BOLD = auto()
    """
    Headings and values that stand out.
    """

    MONO = auto()
    """
    Code.
    """


class FontName(StrEnum):
    """
    The fonts that come with matplotlib, named as it finds fonts.
    """

    SANS = "DejaVu Sans"
    """
    The sans-serif font.
    """

    SANS_BOLD = "DejaVu Sans:bold"
    """
    The sans-serif font in bold.
    """

    SANS_MONO = "DejaVu Sans Mono"
    """
    The monospaced font.
    """


class FontStyle(StrEnum):
    """
    The style names a font file reports for itself.
    """

    BOOK = "Book"
    """
    The regular weight.
    """

    BOLD = "Bold"
    """
    The bold weight.
    """


@dataclass(frozen=True)
class Typeface:
    """
    The fonts a video sets each face in, named as matplotlib finds fonts.
    """

    regular: str = FontName.SANS
    """
    The font of running text.
    """

    bold: str = FontName.SANS_BOLD
    """
    The font of headings.
    """

    mono: str = FontName.SANS_MONO
    """
    The font of code.
    """

    def name_of(self, face: FontFace) -> str:
        """
        :param face: A face.
        :return: The font this typeface sets it in.
        """
        return {
            FontFace.REGULAR: self.regular,
            FontFace.BOLD: self.bold,
            FontFace.MONO: self.mono,
        }[face]

    def font(self, face: FontFace, size: int) -> ImageFont.FreeTypeFont:
        """
        The given face at the given size, found where matplotlib keeps fonts, so a
        picture is set the same on every machine that has the font; the default fonts
        come with matplotlib itself.

        :param face: Which face.
        :param size: The size in pixels.
        :return: The font.
        :raises ValueError: If no font of the face's name is installed.
        """
        found = font_manager.findfont(
            font_manager.FontProperties(self.name_of(face)), fallback_to_default=False
        )
        return ImageFont.truetype(found, size)
