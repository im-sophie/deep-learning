# Typing
from typing import Any, Generator, List

# Internal
from ..tree.tree_glob_match import tree_glob_match
from ..tree.TreeBase import TreeBase
from ..tree.ValueBase import ValueBase
from .NLPresentationRule import NLPresentationRule

class NLPresenter(object):
    rules: List[NLPresentationRule]

    def __init__(self, *args: NLPresentationRule):
        self.rules = list(args)

    def present(self, value: Any) -> str:
        if isinstance(value, TreeBase):
            for rule in self.rules:
                status, _ = tree_glob_match(rule.pattern, value)
                if status:
                    return rule.format.format(
                        value,
                        lambda i: self.present(i)
                    )

            return "[{0}]".format(
                value
            )
        else:
            return str(value)
