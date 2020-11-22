from collections.abc import Iterable
from functools import reduce
import operator
import copy as C

class Shape(tuple):
    @staticmethod
    def to_shape(value, copy = True):
        if isinstance(value, Shape):
            if copy:
                return C.deepcopy(value)
            else:
                return value
        else:
            try:
                return Shape(list(value))
            except TypeError:
                raise TypeError("cannot convert non-iterable value to Shape")
    
    @property
    def flat_size(self):
        return reduce(operator.mul, self, 1)
