# Standard library
import random

# Typing
from typing import Collection

# Internal
from ..tree.Scope import Scope
from ..tree.TypeBase import TypeBase
from .TreeSynthesizerEnumerative import TreeSynthesizerEnumerative

class TreeSynthesizerStochastic(TreeSynthesizerEnumerative):
    probability: float

    def __init__(self, scope: Scope, max_depth: int, probability: float):
        super().__init__(scope, max_depth)
        self.probability = probability

        assert 0 <= self.probability
        assert self.probability <= 1

    def on_can_use_factory(self, result_type: TypeBase, arg_types: Collection[TypeBase]) -> bool:
        return random.random() < self.probability
