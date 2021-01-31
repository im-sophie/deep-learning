# Standard library
import string

# Typing
from typing import Optional, Union

# Internal
from ..tree.ValueBase import ValueBase
from .NLPresentationFormat import NLPresentationFormat

class NLPresentationRule(object):
    pattern: ValueBase
    format: NLPresentationFormat

    def __init__(self, pattern: ValueBase, format_: Union[str, NLPresentationFormat]):
        self.pattern = pattern

        if isinstance(format_, str):
            self.format = NLPresentationFormat(format_)
        elif isinstance(format_, NLPresentationFormat):
            self.format = format_
        else:
            raise Exception("unexpected type of format: {0}".format(type(format_).__name__))

    @property
    def alphabet(self) -> str:
        return self.format.alphabet
