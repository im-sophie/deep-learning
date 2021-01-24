class TreeVerificationError(Exception):
    pass

class LexingError(Exception):
    offset: int
    line: int
    column: int
    text: str

    def __init__(self,
        offset: int,
        line: int,
        column: int,
        text: str,
        *args: object) -> None:
        super().__init__(*args)
        self.offset = offset
        self.line = line
        self.column = column
        self.text = text
