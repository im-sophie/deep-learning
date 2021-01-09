import sophiedl.symbolic as SS

scope = SS.Scope(
    b0 = SS.TypeBool(),
    b1 = SS.TypeBool(),
    b2 = SS.TypeBool(),
    a0 = SS.TypeAbstract(),
    a1 = SS.TypeAbstract(),
    f0 = SS.TypeFunction(
        SS.TypeBool(),
        [
            SS.TypeAbstract()
        ]
    )
)

synth = SS.TreeSynthesizerStochastic(
    scope,
    3,
    0.4
)

n = 0

for i in synth.synthesize():
    n += 1

print(n)
