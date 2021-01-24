# Standard library
import itertools

# Typing
from typing import Any, Callable, cast, Collection, Dict, Generator, Iterable, List, Optional, Tuple

# Internal
from ..tree.Scope import Scope
from ..tree.TypeBase import TypeBase
from ..tree.types import TypeBool, TypeInt, TypeFunction, TypeAbstract
from ..tree.ValueBase import ValueBase
from ..tree.values import ValueBool, ValueInt, ValueSymbol, ValueCall, ValueNot, ValueNegate, ValueOr, ValueAnd, ValueImplies, ValueAdd, ValueSub, ValueMul, ValueDiv, ValueLT, ValueLE, ValueGT, ValueGE, ValueNE, ValueEQ
from .TreeSynthesizerBase import TreeSynthesizerBase

class TreeSynthesizerEnumerative(TreeSynthesizerBase):
    scope: Scope
    max_depth: int

    def __init__(self, scope: Scope, max_depth: int):
        self.scope = scope
        self.max_depth = max_depth

    def _get_symbols_of_type(self, type_: TypeBase) -> List[ValueSymbol]:
        return [ValueSymbol(i) for i in self.scope if self.scope[i] == type_]

    def _get_literals_of_type(self, type_: TypeBase) -> List[ValueBase]:
        if type_ == TypeBool():
            return [ValueBool(True), ValueBool(False)]
        elif type_ == TypeInt():
            return [ValueInt(-5), ValueInt(-1), ValueInt(0), ValueInt(1), ValueInt(5)]
        else:
            return []

    def _get_callable_symbols_with_return_type(self, type_: TypeBase) -> List[ValueSymbol]:
        return [ValueSymbol(i) for i in self.scope if isinstance(self.scope[i], TypeFunction) and cast(TypeFunction, self.scope[i]).return_type == type_]

    def _create_factories_of_type(self, type_: TypeBase) -> Generator[Tuple[List[TypeBase], Callable[[Iterable[ValueBase]], ValueBase]], None, None]:
        for i in self._get_literals_of_type(type_):
            yield [], lambda args: i

        for i in self._get_symbols_of_type(type_):
            yield [], lambda args: i

        for i in self._get_callable_symbols_with_return_type(type_):
            yield cast(TypeFunction, self.scope[i.name]).arg_types, lambda args: ValueCall(i, args)

        if isinstance(type_, TypeBool):
            yield [TypeBool()], lambda args: ValueNot(*args)
            yield [TypeBool(), TypeBool()], lambda args: ValueOr(*args)
            yield [TypeBool(), TypeBool()], lambda args: ValueAnd(*args)
            yield [TypeBool(), TypeBool()], lambda args: ValueImplies(*args)
            yield [TypeBool(), TypeBool()], lambda args: ValueNE(*args)
            yield [TypeBool(), TypeBool()], lambda args: ValueEQ(*args)
            yield [TypeInt(), TypeInt()], lambda args: ValueLT(*args)
            yield [TypeInt(), TypeInt()], lambda args: ValueLE(*args)
            yield [TypeInt(), TypeInt()], lambda args: ValueGT(*args)
            yield [TypeInt(), TypeInt()], lambda args: ValueGE(*args)
            yield [TypeInt(), TypeInt()], lambda args: ValueNE(*args)
            yield [TypeInt(), TypeInt()], lambda args: ValueEQ(*args)
            yield [TypeAbstract(), TypeAbstract()], lambda args: ValueNE(*args)
            yield [TypeAbstract(), TypeAbstract()], lambda args: ValueEQ(*args)

        if isinstance(type_, TypeInt):
            yield [TypeInt()], lambda args: ValueNegate(*args)
            yield [TypeInt(), TypeInt()], lambda args: ValueAdd(*args)
            yield [TypeInt(), TypeInt()], lambda args: ValueSub(*args)
            yield [TypeInt(), TypeInt()], lambda args: ValueMul(*args)
            yield [TypeInt(), TypeInt()], lambda args: ValueDiv(*args)

    def on_can_use_factory(self, result_type: TypeBase, arg_types: Collection[TypeBase]) -> bool:
        return True

    def on_synthesize(self) -> Generator[ValueBase, None, None]:
        assert self.max_depth > 0

        # Initialize synthesis
        cache: Dict[type, List[ValueBase]] = {}

        # Seed base layer of cache
        for type_ in [TypeBool(), TypeInt(), TypeAbstract()]:
            cache[type(type_)] = []
            for arg_types, factory in self._create_factories_of_type(type_):
                if len(arg_types) == 0 and self.on_can_use_factory(type_, arg_types):
                    cache[type(type_)].append(factory([]))

        # Use factories to synthesize intermediate layers
        if self.max_depth > 2:
            for i in range(self.max_depth-2):
                cache_next: Dict[type, List[ValueBase]] = {i: [] for i in cache}

                for type_ in [TypeBool(), TypeInt(), TypeAbstract()]:
                    for arg_types, factory in self._create_factories_of_type(type_):
                        if len(arg_types) > 0 and self.on_can_use_factory(type_, arg_types):
                            arg_value_sets = []
                            for arg_type in arg_types:
                                arg_value_sets.append(cache[type(arg_type)])

                            for arg_valuation in itertools.product(*arg_value_sets):
                                cache_next[type(type_)].append(factory(arg_valuation))

                for j in cache:
                    cache[j] += cache_next[j]

        # Use factories to synthesize final layer
        if self.max_depth > 1:
            for arg_types, factory in self._create_factories_of_type(TypeBool()):
                if len(arg_types) > 0 and self.on_can_use_factory(type_, arg_types):
                    arg_value_sets = []
                    for arg_type in arg_types:
                        arg_value_sets.append(cache[type(arg_type)])

                    for arg_valuation in itertools.product(*arg_value_sets):
                        yield factory(arg_valuation)
