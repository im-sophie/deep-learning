# Typing
from typing import Iterable, List

# Internal
from ...domain.exceptions import TreeVerificationError
from .formatting import format_color_type, format_color_default
from .Scope import Scope
from .TypeBase import TypeBase

class TypeAtomic(TypeBase):
    name: str

    def __init__(self, name: str):
        self.name = name

    def on_format(self, indent: int, indent_width: int, color: bool) -> str:
        return "{0}{1}{2}".format(
            format_color_type(color),
            self.name,
            format_color_default(color)
        )

    def on_verify(self, scope: Scope) -> None:
        pass

class TypeWildcard(TypeAtomic):
    def __init__(self) -> None:
        super().__init__("*")

class TypeBool(TypeAtomic):
    def __init__(self) -> None:
        super().__init__("Bool")

class TypeInt(TypeAtomic):
    def __init__(self) -> None:
        super().__init__("Int")

class TypeAbstract(TypeAtomic):
    def __init__(self) -> None:
        super().__init__("Abstract")

class TypeFunction(TypeBase):
    return_type: TypeBase
    arg_types: List[TypeBase]

    def __init__(self, return_type: TypeBase, arg_types: Iterable[TypeBase]) -> None:
        self.return_type = return_type
        self.arg_types = list(arg_types)

    def on_format(self, indent: int, indent_width: int, color: bool) -> str:
        return "{0} <- ({1})".format(
            self.return_type.format(indent, indent_width, color),
            ", ".join([i.format(indent, indent_width, color) for i in self.arg_types])
        )

    def on_verify(self, scope: Scope) -> None:
        if isinstance(self.return_type, TypeFunction):
            raise TreeVerificationError("functions are not first-class objects")

        if any(isinstance(i, TypeFunction) for i in self.arg_types):
            raise TreeVerificationError("functions are not first-class objects")
