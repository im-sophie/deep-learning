from __future__ import annotations
from collections.abc import Iterable
from functools import reduce
import operator
import copy as C
from typing import Any, List

class Shape(List[int]):
    @staticmethod
    def to_shape(value: Any, copy: bool = True) -> Shape:
        if isinstance(value, Shape):
            if copy:
                return C.deepcopy(value)
            else:
                return value
        else:
            try:
                return Shape(value)
            except TypeError:
                raise TypeError("cannot convert non-iterable value to Shape")
    
    @property
    def flat_size(self) -> int:
        return reduce(operator.mul, self, 1)
