# Standard library
import abc

# Typing
from typing import Any, Iterable, List

# Internal
from ...domain.exceptions import TreeVerificationError
from .formatting import format_color_literal, format_color_default, format_color_symbol
from .Precedence import Precedence
from .Scope import Scope
from .TypeBase import TypeBase
from .types import TypeBool, TypeInt, TypeFunction
from .ValueBase import ValueBase

class ValueWildcard(ValueBase):
    def __init__(self) -> None:
        super().__init__(Precedence.TERM)

    def on_format(self, indent: int, indent_width: int, color: bool) -> str:
        return "*"

    def on_get_type(self, scope: Scope) -> TypeBase:
        raise Exception("wildcard value has no type")

    def on_verify(self, scope: Scope) -> None:
        pass

class ValueLiteral(ValueBase):
    type: TypeBase
    value: Any

    def __init__(self, type: TypeBase, value: Any):
        super().__init__(Precedence.TERM)
        self.type = type
        self.value = value

    @abc.abstractmethod
    def on_format_literal(self) -> str:
        pass

    def on_format(self, indent: int, indent_width: int, color: bool) -> str:
        return "{0}{1}{2}".format(
            format_color_literal(color),
            self.on_format_literal(),
            format_color_default(color)
        )

    def on_get_type(self, scope: Scope) -> TypeBase:
        return self.type

    def on_verify(self, scope: Scope) -> None:
        pass

class ValueBool(ValueLiteral):
    def __init__(self, value: bool):
        super().__init__(TypeBool(), value)

    def on_format_literal(self) -> str:
        return "true" if self.value else "false"

class ValueInt(ValueLiteral):
    def __init__(self, value: int):
        super().__init__(TypeInt(), value)

    def on_format_literal(self) -> str:
        return str(self.value)

class ValueSymbol(ValueBase):
    name: str

    def __init__(self, name: str):
        super().__init__(Precedence.TERM)
        self.name = name

    def on_format(self, indent: int, indent_width: int, color: bool) -> str:
        return "{0}{1}{2}".format(
            format_color_symbol(color),
            self.name,
            format_color_default(color)
        )

    def on_get_type(self, scope: Scope) -> TypeBase:
        return scope[self.name]

    def on_verify(self, scope: Scope) -> None:
        if not self.name in scope:
            raise TreeVerificationError("use of undeclared symbol {0}".format(
                repr(self.name)
            ))

class ValueCall(ValueBase):
    callee: ValueBase
    args: List[ValueBase]

    def __init__(self, callee: ValueBase, args: Iterable[ValueBase]) -> None:
        super().__init__(Precedence.TERM)
        self.callee = callee
        self.args = list(args)

    def on_format(self, indent: int, indent_width: int, color: bool) -> str:
        return "{0}({1})".format(
            self.format_with_parenthesis(self.callee, indent, indent_width, color),
            ", ".join([i.format(indent, indent_width, color) for i in self.args])
        )

    def on_get_type(self, scope: Scope) -> TypeBase:
        callee_type = self.callee.get_type(scope)

        if not isinstance(callee_type, TypeFunction):
            raise Exception("cannot call value of non-functional type: {0}".format(self.callee))

        return callee_type.return_type

    def on_verify(self, scope: Scope) -> None:
        callee_type = self.callee.get_type(scope)

        if not isinstance(callee_type, TypeFunction):
            raise TreeVerificationError("cannot call value of non-functional type: {0}".format(self.callee))

        if len(callee_type.arg_types) != len(self.args):
            raise TreeVerificationError("call to {0} expects {1} argument{2}, not {3}".format(
                self.callee,
                len(callee_type.arg_types),
                "" if len(callee_type.arg_types) == 1 else "s",
                len(self.args)
            ))

        for i in range(len(callee_type.arg_types)):
            arg_type = self.args[i].get_type(scope)
            if callee_type.arg_types[i] != arg_type:
                raise TreeVerificationError("argument {0} of {1} expected to be of type {2}, not {3}".format(
                    i,
                    self.callee,
                    callee_type.arg_types[i],
                    arg_type
                ))

class ValueUnary(ValueBase):
    prefix: str
    suffix: str
    arg: ValueBase

    def __init__(self, prefix: str, suffix: str, precedence: Precedence, arg: ValueBase) -> None:
        super().__init__(precedence)
        self.prefix = prefix
        self.suffix = suffix
        self.arg = arg

    def on_format(self, indent: int, indent_width: int, color: bool) -> str:
        return "{0}{1}{2}".format(
            self.prefix,
            self.format_with_parenthesis(self.arg, indent, indent_width, color),
            self.suffix
        )

class ValueNot(ValueUnary):
    def __init__(self, arg: ValueBase) -> None:
        super().__init__("not ", "", Precedence.NOT, arg)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return TypeBool()

    def on_verify(self, scope: Scope) -> None:
        arg_type = self.arg.get_type(scope)
        if arg_type != TypeBool():
            raise TreeVerificationError("expected argument of 'not' to be of type Bool, not {0}".format(
                arg_type
            ))

class ValueNegate(ValueUnary):
    def __init__(self, arg: ValueBase) -> None:
        super().__init__("-", "", Precedence.NEGATE, arg)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return self.arg.get_type(scope)

    def on_verify(self, scope: Scope) -> None:
        arg_type = self.arg.get_type(scope)
        if arg_type != TypeInt():
            raise TreeVerificationError("expected argument of - to be of type Int, not {0}".format(
                arg_type
            ))

class ValueBinary(ValueBase):
    infix: str
    lhs: ValueBase
    rhs: ValueBase

    def __init__(self, infix: str, precedence: Precedence, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(precedence)
        self.infix = infix
        self.lhs = lhs
        self.rhs = rhs

    def on_format(self, indent: int, indent_width: int, color: bool) -> str:
        return "{0}{1}{2}".format(
            self.format_with_parenthesis(self.lhs, indent, indent_width, color),
            self.infix,
            self.format_with_parenthesis(self.rhs, indent, indent_width, color)
        )

class ValueOr(ValueBinary):
    def __init__(self, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(" or ", Precedence.OR, lhs, rhs)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return TypeBool()

    def on_verify(self, scope: Scope) -> None:
        lhs_type = self.lhs.get_type(scope)
        if lhs_type != TypeBool():
            raise TreeVerificationError("expected left-hand side argument of 'or' to be of type Bool, not {0}".format(
                lhs_type
            ))

        rhs_type = self.rhs.get_type(scope)
        if rhs_type != TypeBool():
            raise TreeVerificationError("expected right-hand side argument of 'or' to be of type Bool, not {0}".format(
                rhs_type
            ))

class ValueAnd(ValueBinary):
    def __init__(self, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(" and ", Precedence.AND, lhs, rhs)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return TypeBool()

    def on_verify(self, scope: Scope) -> None:
        lhs_type = self.lhs.get_type(scope)
        if lhs_type != TypeBool():
            raise TreeVerificationError("expected left-hand side argument of 'and' to be of type Bool, not {0}".format(
                lhs_type
            ))

        rhs_type = self.rhs.get_type(scope)
        if rhs_type != TypeBool():
            raise TreeVerificationError("expected right-hand side argument of 'and' to be of type Bool, not {0}".format(
                rhs_type
            ))

class ValueImplies(ValueBinary):
    def __init__(self, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(" => ", Precedence.IMPLIES, lhs, rhs)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return TypeBool()

    def on_verify(self, scope: Scope) -> None:
        lhs_type = self.lhs.get_type(scope)
        if lhs_type != TypeBool():
            raise TreeVerificationError("expected left-hand side argument of => to be of type Bool, not {0}".format(
                lhs_type
            ))

        rhs_type = self.rhs.get_type(scope)
        if rhs_type != TypeBool():
            raise TreeVerificationError("expected right-hand side argument of => to be of type Bool, not {0}".format(
                rhs_type
            ))

class ValueAdd(ValueBinary):
    def __init__(self, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(" + ", Precedence.ADD, lhs, rhs)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return self.lhs.get_type(scope)

    def on_verify(self, scope: Scope) -> None:
        lhs_type = self.lhs.get_type(scope)
        if lhs_type != TypeInt():
            raise TreeVerificationError("expected left-hand side argument of + to be of type Int, not {0}".format(
                lhs_type
            ))

        rhs_type = self.rhs.get_type(scope)
        if lhs_type != rhs_type:
            raise TreeVerificationError("expected right-hand side argument of + to match type of left-hand side, not {0}".format(
                rhs_type
            ))

class ValueSub(ValueBinary):
    def __init__(self, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(" - ", Precedence.ADD, lhs, rhs)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return self.lhs.get_type(scope)

    def on_verify(self, scope: Scope) -> None:
        lhs_type = self.lhs.get_type(scope)
        if lhs_type != TypeInt():
            raise TreeVerificationError("expected left-hand side argument of - to be of type Int, not {0}".format(
                lhs_type
            ))

        rhs_type = self.rhs.get_type(scope)
        if lhs_type != rhs_type:
            raise TreeVerificationError("expected right-hand side argument of - to match type of left-hand side, not {0}".format(
                rhs_type
            ))

class ValueMul(ValueBinary):
    def __init__(self, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(" * ", Precedence.MUL, lhs, rhs)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return self.lhs.get_type(scope)

    def on_verify(self, scope: Scope) -> None:
        lhs_type = self.lhs.get_type(scope)
        if lhs_type != TypeInt():
            raise TreeVerificationError("expected left-hand side argument of * to be of type Int, not {0}".format(
                lhs_type
            ))

        rhs_type = self.rhs.get_type(scope)
        if lhs_type != rhs_type:
            raise TreeVerificationError("expected right-hand side argument of * to match type of left-hand side, not {0}".format(
                rhs_type
            ))

class ValueDiv(ValueBinary):
    def __init__(self, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(" / ", Precedence.MUL, lhs, rhs)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return self.lhs.get_type(scope)

    def on_verify(self, scope: Scope) -> None:
        lhs_type = self.lhs.get_type(scope)
        if lhs_type != TypeInt():
            raise TreeVerificationError("expected left-hand side argument of / to be of type Int, not {0}".format(
                lhs_type
            ))

        rhs_type = self.rhs.get_type(scope)
        if lhs_type != rhs_type:
            raise TreeVerificationError("expected right-hand side argument of / to match type of left-hand side, not {0}".format(
                rhs_type
            ))

class ValueLT(ValueBinary):
    def __init__(self, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(" < ", Precedence.RELATION, lhs, rhs)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return TypeBool()

    def on_verify(self, scope: Scope) -> None:
        lhs_type = self.lhs.get_type(scope)
        if lhs_type != TypeInt():
            raise TreeVerificationError("expected left-hand side argument of < to be of type Int, not {0}".format(
                lhs_type
            ))

        rhs_type = self.rhs.get_type(scope)
        if lhs_type != rhs_type:
            raise TreeVerificationError("expected right-hand side argument of < to match type of left-hand side, not {0}".format(
                rhs_type
            ))

class ValueLE(ValueBinary):
    def __init__(self, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(" <= ", Precedence.RELATION, lhs, rhs)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return TypeBool()

    def on_verify(self, scope: Scope) -> None:
        lhs_type = self.lhs.get_type(scope)
        if lhs_type != TypeInt():
            raise TreeVerificationError("expected left-hand side argument of <= to be of type Int, not {0}".format(
                lhs_type
            ))

        rhs_type = self.rhs.get_type(scope)
        if lhs_type != rhs_type:
            raise TreeVerificationError("expected right-hand side argument of <= to match type of left-hand side, not {0}".format(
                rhs_type
            ))

class ValueGT(ValueBinary):
    def __init__(self, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(" > ", Precedence.RELATION, lhs, rhs)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return TypeBool()

    def on_verify(self, scope: Scope) -> None:
        lhs_type = self.lhs.get_type(scope)
        if lhs_type != TypeInt():
            raise TreeVerificationError("expected left-hand side argument of > to be of type Int, not {0}".format(
                lhs_type
            ))

        rhs_type = self.rhs.get_type(scope)
        if lhs_type != rhs_type:
            raise TreeVerificationError("expected right-hand side argument of > to match type of left-hand side, not {0}".format(
                rhs_type
            ))

class ValueGE(ValueBinary):
    def __init__(self, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(" >= ", Precedence.RELATION, lhs, rhs)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return TypeBool()

    def on_verify(self, scope: Scope) -> None:
        lhs_type = self.lhs.get_type(scope)
        if lhs_type != TypeInt():
            raise TreeVerificationError("expected left-hand side argument of >= to be of type Int, not {0}".format(
                lhs_type
            ))

        rhs_type = self.rhs.get_type(scope)
        if lhs_type != rhs_type:
            raise TreeVerificationError("expected right-hand side argument of >= to match type of left-hand side, not {0}".format(
                rhs_type
            ))

class ValueNE(ValueBinary):
    def __init__(self, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(" /= ", Precedence.COMPARE, lhs, rhs)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return TypeBool()

    def on_verify(self, scope: Scope) -> None:
        lhs_type = self.lhs.get_type(scope)
        rhs_type = self.rhs.get_type(scope)

        if lhs_type != rhs_type:
            raise TreeVerificationError("expected right-hand side argument of /= to match type of left-hand side, not {0}".format(
                rhs_type
            ))

class ValueEQ(ValueBinary):
    def __init__(self, lhs: ValueBase, rhs: ValueBase) -> None:
        super().__init__(" = ", Precedence.COMPARE, lhs, rhs)

    def on_get_type(self, scope: Scope) -> TypeBase:
        return TypeBool()

    def on_verify(self, scope: Scope) -> None:
        lhs_type = self.lhs.get_type(scope)
        rhs_type = self.rhs.get_type(scope)

        if lhs_type != rhs_type:
            raise TreeVerificationError("expected right-hand side argument of = to match type of left-hand side, not {0}".format(
                rhs_type
            ))
