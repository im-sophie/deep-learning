from __future__ import annotations
import itertools
from typing import List, Any, Dict, Generator
from .domain.Repr import Repr

class Hyperparameter(Repr):
    name: str
    value_list: List[Any]
    value: Any

    def __init__(self, name: str, value_list: List[Any]) -> None:
        self.name = name
        self.value_list = value_list
        self.value = value_list[0] if len(value_list) == 1 else None

class HyperparameterSet(object):
    _hyperparameters: Dict[str, Hyperparameter]

    def __init__(self) -> None:
        self._hyperparameters = {}
    
    def __str__(self) -> str:
        return "{{{0}}}".format(
            ", ".join(
                [
                    "{0}={1}".format(
                        key,
                        self._hyperparameters[key].value if self._hyperparameters[key].value else self._hyperparameters[key].value_list
                    ) for key in self._hyperparameters
                ]
            )
        )
    
    def add(self, name: str, *values: Any) -> None:
        if len(values) == 0:
            raise Exception("must specify at least one value for hyperparameter")

        self._hyperparameters[name] = Hyperparameter(name, list(values))

    @property
    def needs_permutation(self) -> bool:
        return any(type(i.value) == type(None) for i in self._hyperparameters.values())

    def permute(self) -> Generator[HyperparameterSet, None, None]:
        keys = list(self._hyperparameters.keys())

        permuted_values = itertools.product(*[self._hyperparameters[i].value_list for i in keys])

        for valuation in permuted_values:
            result = HyperparameterSet()
            for i in range(len(keys)):
                result.add(keys[i], valuation[i])            
            yield result

    def __getitem__(self, key: str) -> Any:
        if not key in self._hyperparameters:
            raise KeyError(key)
    
        if type(self._hyperparameters[key].value) == type(None):
            raise Exception("hyperparameter set needs permutation before it can be used")
    
        return self._hyperparameters[key].value
