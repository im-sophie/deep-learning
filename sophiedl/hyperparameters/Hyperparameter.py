# Typing
from typing import Any, List

# Internal
from ..domain.Repr import Repr

class Hyperparameter(Repr):
    name: str
    value_list: List[Any]
    value: Any

    def __init__(self, name: str, value_list: List[Any]) -> None:
        self.name = name
        self.value_list = value_list
        self.value = value_list[0] if len(value_list) == 1 else None
