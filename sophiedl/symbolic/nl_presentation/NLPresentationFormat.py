# Typing
from typing import Any, Callable, Iterable, List, Optional, Union

# Internal
from ...domain.Repr import Repr
from ...parsing.TextReaderString import TextReaderString
from ...parsing.Token import Token
from ..tree.TreeBase import TreeBase
from .LexerNLPresentationFormat import LexerNLPresentationFormat
from .TokenKindNLPresentationFormat import TokenKindNLPresentationFormat

class NLPresentationFormat(Repr):
    @staticmethod
    def _get_field_at_path(tree: TreeBase, path: str) -> Any:
        components = path.split(".")

        if len(components) == 0:
            raise Exception("cannot access field at empty path")

        cursor: Any = tree

        for component in components:
            if len(component) == 0:
                raise Exception(".. not supported in field paths")
            elif component in cursor.__dict__:
                cursor = cursor.__dict__[component]
            else:
                raise Exception("object of type {0} has no field {1!r}".format(type(cursor).__name__, component))

        return cursor

    tokens: List[Token[TokenKindNLPresentationFormat]]

    def __init__(self,
        value: Union[str, Iterable[Token[TokenKindNLPresentationFormat]]]) -> None:
        if isinstance(value, str):
            self.tokens = list(
                LexerNLPresentationFormat(
                    TextReaderString(
                        value
                    )
                ).lex()
            )
        else:
            self.tokens = list(value)
    
    @property
    def alphabet(self) -> str:
        return "".join(
            set().union(
                *[i.text for i in self.tokens if i.kind == TokenKindNLPresentationFormat.TEXT]
            )
        )

    def format(self, tree: TreeBase, field_formatter: Optional[Callable[[Any], str]] = None) -> str:
        result = ""

        for i in self.tokens:
            if i.kind == TokenKindNLPresentationFormat.TEXT:
                result += i.text.replace("{{", "{").replace("}}", "}")
            elif i.kind == TokenKindNLPresentationFormat.FIELD_PATH:
                field_value = NLPresentationFormat._get_field_at_path(tree, i.text[1:-1])
                if field_formatter:
                    result += field_formatter(field_value)
                else:
                    result += str(field_value)
            else:
                raise Exception("unexpected value of TokenKindNLPresentationFormat: {0}".format(i.kind))

        return result
