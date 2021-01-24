# Standard library
import enum

@enum.unique
class TokenKindNLPresentationFormat(enum.Enum):
    TEXT = 0
    FIELD_PATH = 1