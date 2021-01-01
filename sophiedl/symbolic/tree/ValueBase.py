# Future
from __future__ import annotations

# Standard library
import abc

# Internal
from .Precedence import Precedence
from .Scope import Scope
from .TreeBase import TreeBase
from .TypeBase import TypeBase

class ValueBase(TreeBase):
    precedence: Precedence

    def __init__(self, precedence: Precedence):
        self.precedence = precedence

    @abc.abstractmethod
    def on_get_type(self, scope: Scope) -> TypeBase:
        pass

    def get_type(self, scope: Scope) -> TypeBase:
        return self.on_get_type(scope)

    def format_with_parenthesis(self, child: ValueBase, indent: int, indent_width: int, color: bool) -> str:
        return "{0}{1}{2}".format(
            "(" if self.precedence < child.precedence else "",
            child.format(indent, indent_width, color),
            ")" if self.precedence < child.precedence else ""
        )
