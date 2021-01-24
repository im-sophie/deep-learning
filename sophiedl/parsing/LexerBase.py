# Standard library
import abc

# Typing
from typing import Generic, Generator, Optional, TypeVar

# Internal
from ..domain.exceptions import LexingError
from .TextReaderBase import TextReaderBase
from .Token import Token

TTokenKind = TypeVar("TTokenKind")

class LexerBase(abc.ABC, Generic[TTokenKind]):
    text_reader: TextReaderBase
    offset_first: int
    offset_last: int
    line_first: int
    line_last: int
    column_first: int
    column_last: int
    buffer: str

    def __init__(self,
        text_reader: TextReaderBase,
        offset: int = 0,
        line: int = 1,
        column: int = 1) -> None:
        self.text_reader = text_reader
        self.offset_first = offset
        self.offset_last = offset
        self.line_first = line
        self.line_last = line
        self.column_first = column
        self.column_last = column
        self.buffer = ""

    @abc.abstractmethod
    def on_lex_next(self) -> Optional[Token[TTokenKind]]:
        pass

    def are_more_chars(self) -> bool:
        return self.text_reader.are_more_chars()

    def get_next_char(self) -> Optional[str]:
        return self.text_reader.get_next_char()

    def eat_next_char(self) -> None:
        current = self.get_next_char()

        if current == "\r":
            self.text_reader.eat_next_char()
            self.offset_last += 1

            current = self.get_next_char()
            if current == "\n":
                self.text_reader.eat_next_char()
                self.offset_last += 1

            self.line_last += 1
            self.column_last = 1
            self.buffer += "\n"
        elif current == "\n":
            self.text_reader.eat_next_char()
            self.offset_last += 1
            self.line_last += 1
            self.column_last = 1
            self.buffer += "\n"
        elif current is not None:
            self.text_reader.eat_next_char()
            self.offset_last += 1
            self.column_last += 1
            self.buffer += current

    def uneat_char(self, chr: str) -> None:
        self.text_reader.uneat_char(chr)

    def skip_token(self) -> None:
        self.offset_first = self.offset_last
        self.line_first = self.line_last
        self.column_first = self.column_last
        self.buffer = ""

    def pop_token(self, kind: TTokenKind) -> Token[TTokenKind]:
        token = Token[TTokenKind](
            self.offset_first,
            self.line_first,
            self.column_first,
            self.buffer,
            kind
        )

        self.skip_token()

        return token

    def lex(self) -> Generator[Token[TTokenKind], None, None]:
        while self.are_more_chars():
            offset_save = self.offset_last
            token = self.on_lex_next()
            if self.offset_last == offset_save:
                raise LexingError(
                    self.offset_last,
                    self.line_last,
                    self.column_last,
                    "",
                    "lexer read no characters at this location"
                )
            if token is not None:
                yield token
