class Repr(object):
    def __repr__(self):
        props = [
            "{0}={1}".format(
                key,
                repr(self.__dict__[key])
            ) for key in self.__dict__ if type(self.__dict__[key]) != type(None)
        ]

        return "<{0}{1}{2}>".format(
            type(self).__name__,
            " " if len(props) > 0 else "",
            ", ".join(props)
        )

if __name__ == "__main__":
    class Snek(Repr):
        def __init__(self):
            self.name = "Sneky"
            self.length = 3
            self.age = "rotary phone"

    print(Snek())
