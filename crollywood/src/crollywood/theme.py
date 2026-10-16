"""
The colours and fonts every scene of a video draws in, so its scenes look like one
piece.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.world_description.geometry import Color

from crollywood.frame import Frame, Resolution
from crollywood.lettering import FontFace, Typeface
from crollywood.typesetting import Typesetting


@dataclass(frozen=True)
class SyntaxColors:
    """
    What the kinds of piece of a line of code are written in, as an editor colours them;
    names take the theme's text colour, punctuation and comments its muted one.
    """

    call: Color
    """
    A name being called.
    """

    class_name: Color
    """
    A capitalised name: a class.
    """

    string: Color
    """
    A quoted string.
    """

    number: Color
    """
    A number.
    """

    keyword: Color
    """
    A word of the language itself.
    """


@dataclass(frozen=True)
class VideoTheme:
    """
    The colours and fonts every scene of a video draws in; the hues a video gives its
    own subjects are its own to choose, as :class:`Color` values.
    """

    page: Color
    """
    What a slide is drawn on.
    """

    text: Color
    """
    What text is written in on the page.
    """

    muted: Color
    """
    What secondary text is written in.
    """

    hairline: Color
    """
    What rules and frames are drawn in.
    """

    panel: Color
    """
    What a panel standing off the page is filled with.
    """

    marker: Color
    """
    What a highlighter leaves behind a stretch of code that a reader is pointed to.
    """

    accent: Color
    """
    The one colour that marks what to look at, and how far the video has played.
    """

    syntax: SyntaxColors
    """
    What code is coloured in.
    """

    typeface: Typeface = field(default_factory=Typeface)
    """
    The fonts text is set in.
    """

    @classmethod
    def neutral(cls) -> VideoTheme:
        """
        :return: Dark text on a white page, in greys, with an orange accent: the default.
        """
        return cls(
            page=Color.from_hex("#FFFFFF"),
            text=Color.from_hex("#1F2328"),
            muted=Color.from_hex("#6B7280"),
            hairline=Color.from_hex("#CFD4DC"),
            panel=Color.from_hex("#F6F7F9"),
            marker=Color.from_hex("#FFEE8C"),
            accent=Color.from_hex("#C85A19"),
            syntax=SyntaxColors(
                call=Color.from_hex("#1D4ED8"),
                class_name=Color.from_hex("#6D28D9"),
                string=Color.from_hex("#15803D"),
                number=Color.from_hex("#B45309"),
                keyword=Color.from_hex("#BE123C"),
            ),
        )

    @classmethod
    def dark(cls) -> VideoTheme:
        """
        :return: Light text on a near-black page with an orange accent: the look of an
            explainer video.
        """
        return cls(
            page=Color.from_hex("#0D1117"),
            text=Color.from_hex("#E6EDF3"),
            muted=Color.from_hex("#8B949E"),
            hairline=Color.from_hex("#6E7681"),
            panel=Color.from_hex("#161B22"),
            marker=Color.from_hex("#4D3F12"),
            accent=Color.from_hex("#F0883E"),
            syntax=SyntaxColors(
                call=Color.from_hex("#5B8DEF"),
                class_name=Color.from_hex("#BC8CFF"),
                string=Color.from_hex("#3FB97A"),
                number=Color.from_hex("#E3B341"),
                keyword=Color.from_hex("#FF7B72"),
            ),
        )

    @property
    def footage_text(self) -> Color:
        """
        What text is written in over footage, whatever the theme: white, on footage
        darkened under it.
        """
        return Color.from_hex("#FFFFFF")

    @property
    def footage_badge(self) -> Color:
        """
        What a badge over footage is filled with, whatever the theme.
        """
        return Color.from_hex("#1F2328")

    def page_of(self, resolution: Resolution) -> Frame:
        """
        :param resolution: The size of the frame.
        :return: An empty page of that size.
        """
        return Frame.of_color(resolution, self.page)

    def typesetting(self, size: int, face: FontFace = FontFace.REGULAR) -> Typesetting:
        """
        :param size: The height of the letters, in pixels.
        :param face: The face they are set in.
        :return: Text of that size and face, set in this theme's text colour and fonts.
        """
        return Typesetting(
            size=size, face=face, color=self.text, typeface=self.typeface
        )
