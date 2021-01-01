import sophiedl.symbolic as SS

scope = SS.Scope(
    b0 = SS.TypeBool(),
    b1 = SS.TypeBool(),
    b2 = SS.TypeBool(),
    a0 = SS.TypeFunction(
        SS.TypeBool(),
        [
            SS.TypeAbstract()
        ]
    ),
    a1 = SS.TypeAbstract(),
)

b0 = SS.ValueSymbol("b0")
b1 = SS.ValueSymbol("b1")
b2 = SS.ValueSymbol("b2")

a0 = SS.ValueSymbol("a0")
a1 = SS.ValueSymbol("a1")

# e = SS.ValueImplies(b0, SS.ValueAnd(b1, b2))
e = SS.ValueCall(a0, [a1])

print(e.format())

e.verify(scope)
