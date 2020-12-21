# Future
from __future__ import annotations

# Standard library
import itertools

# Typing
from typing import Any, Dict, Generator

# Internal
from .Hyperparameter import Hyperparameter

class HyperparameterSet(object):
    _hyperparameters: Dict[str, Hyperparameter]

    def __init__(self) -> None:
        self._hyperparameters = {}
    
    def __str__(self) -> str:
        return "{{{0}}}".format(
            ", ".join(
                [str(i) for i in self._hyperparameters.values()]
            )
        )
    
    def add(self, name: str, *values: Any) -> None:
        if len(values) == 0:
            raise Exception("must specify at least one value for hyperparameter")

        if name in self._hyperparameters:
            raise Exception("hyperparameter with key {0} already exists".format(name))

        self._hyperparameters[name] = Hyperparameter(name, *values)

    @property
    def needs_permutation(self) -> bool:
        return any(len(i) > 1 for i in self._hyperparameters.values())

    def permute(self) -> Generator[HyperparameterSet, None, None]:
        keys = list(self._hyperparameters.keys())

        permuted_values = itertools.product(*[self._hyperparameters[i] for i in keys])

        for valuation in permuted_values:
            result = HyperparameterSet()
            for i in range(len(keys)):
                result.add(keys[i], valuation[i])            
            yield result

    def __contains__(self, key: str) -> bool:
        return key in self._hyperparameters

    def __getitem__(self, key: str) -> Any:
        if not key in self._hyperparameters:
            raise KeyError(key)
    
        return self._hyperparameters[key].single

    def __setitem__(self, key: str, value: Any) -> None:
        if not key in self._hyperparameters:
            raise KeyError(key)
    
        self._hyperparameters[key].assign(value)
