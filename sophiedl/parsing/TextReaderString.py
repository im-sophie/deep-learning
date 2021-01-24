# Typing
from typing import Union

# Internal
from .TextReaderBase import TextReaderBase

class TextReaderString(TextReaderBase):
    value: str
    offset: int

    def __init__(self, value: str, offset: int = 0) -> None:
        super().__init__()
        self.value = value
        self.offset = offset

    def on_eat_char(self) -> Union[int, str, None]:
        if self.offset < len(self.value):
            result = self.value[self.offset]
            self.offset += 1
            return result
        else:
            return None
