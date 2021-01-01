# Typing
from typing import Dict, Iterable

# Internal
from .TypeBase import TypeBase

class ScopeFrame(object):
    _symbols: Dict[str, TypeBase]

    def __init__(self, **symbols: TypeBase):
        self._symbols = symbols

    def __contains__(self, key: str) -> bool:
        return key in self._symbols

    def __getitem__(self, key: str) -> TypeBase:
        return self._symbols[key]

    def __setitem__(self, key: str, value: TypeBase) -> None:
        self._symbols[key] = value

    def __delitem__(self, key: str) -> None:
        del self._symbols[key]

    def keys(self) -> Iterable[str]:
        return self._symbols.keys()