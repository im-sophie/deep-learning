# Typing
from typing import List, Optional, Tuple

# Internal
from .TreeBase import TreeBase
from .types import TypeWildcard
from .values import ValueWildcard

def tree_glob_match(a: TreeBase, b: TreeBase) -> Tuple[bool, List[TreeBase]]:
    wildcard_matches: List[TreeBase] = []

    def comparator(a: TreeBase, b: TreeBase) -> Optional[bool]:
        if isinstance(a, TypeWildcard):
            wildcard_matches.append(b)
            return True
        elif isinstance(a, ValueWildcard):
            wildcard_matches.append(b)
            return True
        else:
            return None

    return a.compare(b, comparator), wildcard_matches
