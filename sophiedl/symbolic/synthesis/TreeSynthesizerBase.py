# Standard library
import abc
import itertools
import random

# Typing
from typing import Generator, Iterable

# Internal
from ..tree.ValueBase import ValueBase

class TreeSynthesizerBase(abc.ABC):
    @abc.abstractmethod
    def on_synthesize(self) -> Generator[ValueBase, None, None]:
        pass

    def synthesize(self) -> Generator[ValueBase, None, None]:
        return self.on_synthesize()

    def take(self, n: int) -> Iterable[ValueBase]:
        return itertools.islice(self.synthesize(), n)

    def sample(self, n: int) -> Iterable[ValueBase]:
        population = list(self.synthesize())
        if len(population) < n:
            random.shuffle(population)
            return population
        else:
            return random.sample(population, n)
