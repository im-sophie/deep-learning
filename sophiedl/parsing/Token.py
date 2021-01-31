# Typing
from typing import Generic, TypeVar

# Internal
from ..domain.Repr import Repr

TKind = TypeVar("TKind")

class Token(Repr, Generic[TKind]):
    offset: int
    line: int
    column: int
    text: str
    kind: TKind

    def __init__(
        self,
        offset: int,
        line: int,
        column: int,
        text: str,
        kind: TKind) -> None:
        self.offset = offset
        self.line = line
        self.column = column
        self.text = text
        self.kind = kind

    def __str__(self) -> str:
        return "<{0}:{1}+{2},{3},{4!r}>".format(
            self.line,
            self.column,
            self.offset,
            self.kind,
            self.text
        )
