# Standard library
import enum

@enum.unique
class Precedence(enum.IntEnum):
    TERM = 0
    NEGATE = 1
    MUL = 2
    ADD = 3
    RELATION = 4
    COMPARE = 5
    NOT = 6
    OR = 7
    AND = 8
    IMPLIES = 9
