import itertools

from ..domain import Repr

class Hyperparameter(Repr):
    def __init__(self, name, value_list):
        self.name = name
        self.value_list = value_list
        self.value = value_list[0] if len(value_list) == 1 else None

class HyperparameterSet(object):
    def __init__(self):
        self._hyperparameters = {}
    
    def add(self, name, *values):
        if len(values) == 0:
            raise Exception("must specify at least one value for hyperparameter")

        self._hyperparameters[name] = Hyperparameter(name, list(values))

    @property
    def needs_permutation(self):
        return any(type(i.value) == type(None) for i in self._hyperparameters.values())

    def permute(self):
        keys = list(self._hyperparameters.keys())

        permuted_values = itertools.product(*[self._hyperparameters[i].value_list for i in keys])

        for valuation in permuted_values:
            result = HyperparameterSet()
            for i in range(len(keys)):
                result.add(keys[i], valuation[i])            
            yield result
