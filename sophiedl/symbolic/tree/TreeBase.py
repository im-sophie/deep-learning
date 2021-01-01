# Future
from __future__ import annotations

# Standard library
import abc
import enum

# Typing
from typing import Any, Iterable, Tuple, TYPE_CHECKING

# Internal
if TYPE_CHECKING:
    from .Scope import Scope

class TreeBase(abc.ABC):
    @staticmethod
    def _is_child_member(value: Any) -> bool:
        return (isinstance(value, TreeBase) or
                isinstance(value, list) and any(isinstance(i, TreeBase) for i in value) or
                isinstance(value, dict) and any(isinstance(i, TreeBase) for i in value.values()))

    def __eq__(self, other: object) -> bool:
        if type(self) != type(other):
            return False

        for name, value in self.members:
            if not name in other.__dict__:
                return False
            elif value != other.__dict__[name]:
                return False

        return True

    def __str__(self) -> str:
        return self.format(color = False)

    @property
    def members(self) -> Iterable[Tuple[str, Any]]:
        for i in self.__dict__:
            if i[0] != "_":
                yield (i, self.__dict__[i])

    @property
    def child_members(self) -> Iterable[Tuple[str, Any]]:
        for name, value in self.members:
            if TreeBase._is_child_member(value):
                yield (name, value)

    @property
    def data_members(self) -> Iterable[Tuple[str, Any]]:
        for name, value in self.members:
            if not TreeBase._is_child_member(value):
                yield (name, value)

    @property
    def children(self) -> Iterable[TreeBase]:
        for name, value in self.members:
            if isinstance(value, TreeBase):
                yield value
            elif isinstance(value, list):
                for i in value:
                    if isinstance(i, TreeBase):
                        yield i
            elif isinstance(value, dict):
                for i in value.values():
                    if isinstance(i, TreeBase):
                        yield i

    @abc.abstractmethod
    def on_format(self, indent: int, indent_width: int, color: bool) -> str:
        pass

    def format(self, indent: int = 0, indent_width: int = 2, color: bool = True) -> str:
        return self.on_format(indent, indent_width, color)

    @abc.abstractmethod
    def on_verify(self, scope: "Scope") -> None:
        pass

    def verify(self, scope: "Scope") -> None:
        self.on_verify(scope)
