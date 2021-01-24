# Standard library
import abc

# Typing
from typing import Optional, Union

class TextReaderBase(abc.ABC):
    _buffer: str
    _eos: bool

    def __init__(self) -> None:
        self._buffer = ""
        self._eos = False

    @abc.abstractmethod
    def on_eat_char(self) -> Union[int, str, None]:
        pass

    def _fill_buffer(self) -> None:
        assert not self._eos
        assert len(self._buffer) == 0

        if self._eos:
            return

        result = self.on_eat_char()

        if result is None:
            self._eos = True
        elif isinstance(result, int):
            if result >= 0:
                self._buffer = chr(result)
            else:
                self._eos = True
        elif isinstance(result, str):
            if len(result) == 0:
                self._eos = True
            elif len(result) == 1:
                self._buffer = result
            else:
                raise Exception("expected on_eat_char() to return string of length 1")
        else:
            raise Exception("unexpected type returned by on_eat_char()")

    def _ensure_buffer_filled(self) -> None:
        if not self._eos and len(self._buffer) == 0:
            self._fill_buffer()

    def are_more_chars(self) -> bool:
        self._ensure_buffer_filled()
        return len(self._buffer) > 0

    def get_next_char(self) -> Optional[str]:
        self._ensure_buffer_filled()

        if len(self._buffer) > 0:
            return self._buffer[0]
        else:
            return None

    def eat_next_char(self) -> None:
        if not self._eos and len(self._buffer) > 0:
            self._buffer = self._buffer[1:]

    def uneat_char(self, chr: str) -> None:
        self._buffer += chr
