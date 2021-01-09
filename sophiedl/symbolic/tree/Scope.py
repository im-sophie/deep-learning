# Standard library
import itertools

# Typing
from typing import cast, Iterable, Iterator, List

# Internal
from .formatting import format_newline, format_color_symbol, format_color_default
from .ScopeFrame import ScopeFrame
from .TypeBase import TypeBase

class Scope(object):
    _frames: List[ScopeFrame]

    def __init__(self, **symbols: TypeBase) -> None:
        self._frames = [ScopeFrame(**symbols)]

    def __contains__(self, key: str) -> bool:
        return any(key in frame for frame in self._frames)

    def __getitem__(self, key: str) -> TypeBase:
        for frame in reversed(self._frames):
            if key in frame:
                return frame[key]

        raise KeyError(key)

    def __setitem__(self, key: str, value: TypeBase) -> None:
        self._frames[-1][key] = value

    def __delitem__(self, key: str) -> None:
        for frame in reversed(self._frames):
            if key in frame:
                del frame[key]
                return

        raise KeyError(key)
    
    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __str__(self) -> str:
        return self.format(color = False)

    def keys(self) -> Iterable[str]:
        return set(itertools.chain(*[list(i.keys()) for i in self._frames]))

    def format(self, indent: int = 0, indent_width: int = 2, color: bool = True) -> str:
        s = ""

        for i in sorted(self.keys()):
            if len(s) > 0:
                s += format_newline(indent, indent_width)

            s += "{0}{1}{2}: {3}".format(
                format_color_symbol(color),
                i,
                format_color_default(color),
                self[i].format(indent = indent + 1, indent_width = indent_width, color = color)
            )

        return s

    def push(self) -> None:
        self._frames.append(ScopeFrame())

    def pop(self) -> None:
        if len(self._frames) > 1:
            del self._frames[-1]
