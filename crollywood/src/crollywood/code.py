"""
Lines of Python written on a frame, coloured piece by piece as an editor would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import StrEnum

from pygments.lexer import Lexer
from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Name, Number, Operator, String
from semantic_digital_twin.spatial_types.spatial_types import Point2
from semantic_digital_twin.world_description.geometry import Color
from typing_extensions import TYPE_CHECKING, Sequence

from crollywood.frame import Frame
from crollywood.lettering import FontFace
from crollywood.theme import VideoTheme
from crollywood.typesetting import Anchor, Typesetting

if TYPE_CHECKING:
    from pygments.token import _TokenType


class Token(StrEnum):
    """
    The kinds of piece a line of Python is coloured by.
    """

    CALL = "call"
    """
    A name followed by a parenthesis.
    """

    CLASS = "class"
    """
    A capitalised name.
    """

    STRING = "string"
    """
    A quoted string.
    """

    NUMBER = "number"
    """
    A number.
    """

    KEYWORD = "keyword"
    """
    A word of the language itself, such as ``for``, ``in`` or ``None``.
    """

    COMMENT = "comment"
    """
    A comment.
    """

    NAME = "name"
    """
    Any other name.
    """

    PUNCTUATION = "punctuation"
    """
    Anything else.
    """

    @classmethod
    def of(cls, kind: _TokenType, text: str, following: str) -> Token:
        """
        :param kind: What the lexer took a piece of a line for.
        :param text: The piece.
        :param following: The piece after it that is not a space, if any.
        :return: The kind of piece it is coloured by: a name is a class when it is
            capitalised, a call when a parenthesis follows it.
        """
        if kind in String:
            return cls.STRING
        if kind in Number:
            return cls.NUMBER
        if kind in Comment:
            return cls.COMMENT
        if kind in Keyword or kind in Operator.Word:
            return cls.KEYWORD
        if kind not in Name:
            return cls.PUNCTUATION
        if text[0].isupper():
            return cls.CLASS
        return cls.CALL if following.startswith("(") else cls.NAME


@dataclass(frozen=True)
class LineSpan:
    """
    A stretch of a line.
    """

    first: int
    """
    The index of its first character.
    """

    after_last: int
    """
    The index of the character after its last.
    """


@dataclass(frozen=True)
class CodePiece:
    """
    One piece of a line of code with the kind it is coloured by.
    """

    text: str
    """
    The characters of the piece.
    """

    token: Token
    """
    The kind of piece it is.
    """


@dataclass(frozen=True)
class CodeLine:
    """
    One line of Python.
    """

    text: str
    """
    The line.
    """

    lexer: Lexer = field(
        default_factory=lambda: PythonLexer(stripnl=False, ensurenl=False),
        compare=False,
        repr=False,
    )
    """
    What cuts the line into the pieces of the language.
    """

    @property
    def pieces(self) -> list[CodePiece]:
        """
        The line cut into pieces, each with the kind it is coloured by; neighbouring
        pieces of one kind are one piece.
        """
        lexed = list(self.lexer.get_tokens(self.text))
        pieces: list[CodePiece] = []
        for index, (kind, text) in enumerate(lexed):
            following = next(
                (later for _, later in lexed[index + 1 :] if not later.isspace()), ""
            )
            token = Token.of(kind, text, following)
            if pieces and pieces[-1].token is token:
                pieces[-1] = CodePiece(pieces[-1].text + text, token)
            else:
                pieces.append(CodePiece(text, token))
        return pieces

    def changed_spans(self, earlier: CodeLine) -> list[LineSpan]:
        """
        :param earlier: The line before this one.
        :return: The stretches of this line that differ from the earlier one: what a
            reader comparing the two would find changed, piece by piece.
        """
        before = [piece.text for piece in earlier.pieces]
        after = [piece.text for piece in self.pieces]
        starts = [0]
        for text in after:
            starts.append(starts[-1] + len(text))
        return [
            LineSpan(starts[first], starts[after_last])
            for tag, _, _, first, after_last in SequenceMatcher(
                a=before, b=after, autojunk=False
            ).get_opcodes()
            if tag != "equal" and after_last > first
        ]


@dataclass(frozen=True)
class CodeTypesetting:
    """
    How a line of Python is written on a frame: in the mono face, coloured piece by
    piece as an editor would, with any stretch a reader is pointed to marked behind it.
    """

    size: int = 22
    """
    The height of the letters, in pixels.
    """

    theme: VideoTheme = field(default_factory=VideoTheme.neutral)
    """
    What the pieces are coloured in, what a marked stretch is filled behind with, and
    the mono font.
    """

    @property
    def face(self) -> Typesetting:
        """
        The mono face at this size, in the theme's text colour.
        """
        return self.theme.typesetting(self.size, FontFace.MONO)

    def color_of(self, token: Token) -> Color:
        """
        :param token: A kind of piece.
        :return: What pieces of that kind are written in.
        """
        return {
            Token.CALL: self.theme.syntax.call,
            Token.CLASS: self.theme.syntax.class_name,
            Token.STRING: self.theme.syntax.string,
            Token.NUMBER: self.theme.syntax.number,
            Token.KEYWORD: self.theme.syntax.keyword,
            Token.COMMENT: self.theme.muted,
            Token.NAME: self.theme.text,
            Token.PUNCTUATION: self.theme.muted,
        }[token]

    def written(
        self,
        frame: Frame,
        line: CodeLine,
        at: Point2,
        marked: Sequence[LineSpan] = (),
    ) -> Frame:
        """
        :param frame: The frame written on.
        :param line: The line.
        :param at: Where its left middle goes, in pixels.
        :param marked: The stretches of the line marked behind, if any.
        :return: A copy of the frame with the line written on it.
        """
        x, y = float(at.x), float(at.y)
        for span in marked:
            left = x + self.width_of(line.text[: span.first])
            right = x + self.width_of(line.text[: span.after_last])
            frame = frame.filled(
                Frame.box(
                    left - 3, y - self.size * 0.62, right - left + 6, self.size * 1.24
                ),
                self.theme.marker,
            )
        for piece in line.pieces:
            if not piece.text.isspace():
                coloured = Typesetting(
                    size=self.size,
                    face=FontFace.MONO,
                    color=self.color_of(piece.token),
                    typeface=self.theme.typeface,
                )
                frame = coloured.written(
                    frame, piece.text, Point2(x, y), Anchor.LEFT_MIDDLE
                )
            x += self.face.width_of(piece.text)
        return frame

    def width_of(self, text: str) -> float:
        """
        :param text: A stretch of one line.
        :return: How many pixels wide it is set.
        """
        return self.face.width_of(text)
