# Typing
from typing import Any, Iterator, List

# Internal
from ..domain.Repr import Repr

class Hyperparameter(Repr):
    name: str
    _value_list: List[Any]

    def __init__(self, name: str, *value_list: Any) -> None:
        self.name = name
        self._value_list = list(value_list)

    def __str__(self) -> str:
        return "{0}={1}".format(
            self.name,
            self._value_list[0] if len(self._value_list) == 1 else self._value_list
        )

    def __iter__(self) -> Iterator[Any]:
        return iter(self._value_list)
    
    def __len__(self) -> int:
        return len(self._value_list)
    
    def __getitem__(self, key: int) -> Any:
        return self._value_list[key]

    @property
    def single(self) -> Any:
        if len(self._value_list) == 0:
            raise Exception("no value specified for hyperparameter")
        elif len(self._value_list) > 1:
            raise Exception("hyperparameter needs permutation before it can be used")
    
        return self._value_list[0]

    def assign(self, *value_list: Any) -> None:
        self._value_list = list(value_list)
