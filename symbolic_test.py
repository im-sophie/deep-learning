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

nl_presenter = SS.NLPresenter(
    SS.NLPresentationRule(
        SS.ValueAnd(
            SS.ValueWildcard(),
            SS.ValueWildcard()
        ),
        "{lhs} and {rhs}"
    ),
    SS.NLPresentationRule(
        SS.ValueOr(
            SS.ValueWildcard(),
            SS.ValueWildcard()
        ),
        "{lhs} or {rhs}"
    ),
    SS.NLPresentationRule(
        SS.ValueImplies(
            SS.ValueWildcard(),
            SS.ValueWildcard()
        ),
        "if {lhs}, then {rhs}"
    ),
    SS.NLPresentationRule(
        SS.ValueNot(
            SS.ValueWildcard()
        ),
        "{arg} is false"
    ),
    SS.NLPresentationRule(
        SS.ValueLT(
            SS.ValueWildcard(),
            SS.ValueWildcard()
        ),
        "{lhs} is less than {rhs}"
    ),
    SS.NLPresentationRule(
        SS.ValueLE(
            SS.ValueWildcard(),
            SS.ValueWildcard()
        ),
        "{lhs} is less than or equal to {rhs}"
    ),
    SS.NLPresentationRule(
        SS.ValueGT(
            SS.ValueWildcard(),
            SS.ValueWildcard()
        ),
        "{lhs} is greater than {rhs}"
    ),
    SS.NLPresentationRule(
        SS.ValueGE(
            SS.ValueWildcard(),
            SS.ValueWildcard()
        ),
        "{lhs} is greater than or equal to {rhs}"
    ),
    SS.NLPresentationRule(
        SS.ValueNE(
            SS.ValueWildcard(),
            SS.ValueWildcard()
        ),
        "{lhs} is not equal to {rhs}"
    ),
    SS.NLPresentationRule(
        SS.ValueEQ(
            SS.ValueWildcard(),
            SS.ValueWildcard()
        ),
        "{lhs} is equal to {rhs}"
    )
)

for i in synth.sample(20):
    print(
        "{0} \033[0;90m{1}\033[0;0m".format(
            i.format(),
            nl_presenter.present(i)
        )
    )
