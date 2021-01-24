class Repr(object):
    def __repr__(self) -> str:
        props = [
            "{0}={1}".format(
                key,
                repr(self.__dict__[key])
            ) for key in self.__dict__ if self.__dict__[key] is not None
        ]

        return "<{0}{1}{2}>".format(
            type(self).__name__,
            " " if len(props) > 0 else "",
            ", ".join(props)
        )

if __name__ == "__main__":
    class Snek(Repr):
        name: str
        length: int
        age: str

        def __init__(self) -> None:
            self.name = "Sneky"
            self.length = 3
            self.age = "rotary phone"

    print(Snek())
