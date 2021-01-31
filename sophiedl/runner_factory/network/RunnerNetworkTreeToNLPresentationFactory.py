# Standard library
import random

# Typing
from typing import List, Optional

# PyTorch
import torch.nn as nn

# Internal
from ...hyperparameters.HyperparameterSet import HyperparameterSet
from ...network.OptimizedModule import OptimizedModule
from ...network.OptimizedSequential import OptimizedSequential
from ...running.RunnerBase import RunnerBase
from ...running.RunnerNetworkTreeToNLPresentation import RunnerNetworkTreeToNLPresentation
from ...symbolic.nl_presentation.NLPresenter import NLPresenter
from ...symbolic.nl_presentation.NLPresentationRule import NLPresentationRule
from ...symbolic.synthesis.TreeSynthesizerBase import TreeSynthesizerBase
from ...symbolic.synthesis.TreeSynthesizerStochastic import TreeSynthesizerStochastic
from ...symbolic.tree.Scope import Scope
from ...symbolic.tree.TypeBase import TypeBase
from ...symbolic.tree.types import TypeAbstract, TypeBool, TypeFunction, TypeInt
from ...symbolic.tree.values import ValueAnd, ValueOr, ValueImplies, ValueNot, ValueLT, ValueLE, ValueGT, ValueGE, ValueNE, ValueEQ, ValueWildcard
from ..base.RunnerFactoryBase import RunnerFactoryBase

class RunnerNetworkTreeToNLPresentationFactory(RunnerFactoryBase):
    def on_create_default_hyperparameter_set(self) -> HyperparameterSet:
        hyperparameter_set = HyperparameterSet()
        hyperparameter_set.add("batch_size_training", 10 ** 3)
        hyperparameter_set.add("batch_size_testing", 10 ** 3)
        hyperparameter_set.add("epoch_count", 25)
        hyperparameter_set.add("learning_rate", 0.001)
        return hyperparameter_set

    def _create_network(
        self,
        hyperparameter_set: HyperparameterSet) -> OptimizedModule:
        return OptimizedSequential(
            nn.Linear(
                in_features = 256,
                out_features = 256
            ),
            optimizer_factory = OptimizedSequential.optimizer_factory_adam,
            learning_rate = hyperparameter_set["learning_rate"]
        )

    def _get_function_argument_type_permutations(
        self,
        max_count: int) -> List[List[TypeBase]]:
        assert max_count > 0

        if max_count == 1:
            return [[TypeBool(), TypeAbstract(), TypeInt()]]
        else:
            next_results = self._get_function_argument_type_permutations(max_count - 1)
            current_results = list(next_results)

            for i in self._get_function_argument_type_permutations(1):
                for j in next_results:
                    current_results.append(i + j)

            return current_results

    def _get_function_type_permutations(
        self,
        max_argument_count: int,
        max_type_count: int) -> List[TypeFunction]:
        assert max_argument_count > 0
        assert max_type_count >= 0

        results = []

        for i in self._get_function_argument_type_permutations(1):
            for j in self._get_function_argument_type_permutations(max_argument_count):
                assert len(i) == 1
                results.append(TypeFunction(i[0], j))

        if len(results) > max_type_count:
            return random.sample(results, max_type_count)
        else:
            return results

    def _create_scope(
        self,
        hyperparameter_set: HyperparameterSet) -> Scope:
        scope = Scope()

        for i in range(hyperparameter_set["boolean_symbol_count"]):
            scope["b{0}".format(i)] = TypeBool()

        for i in range(hyperparameter_set["abstract_symbol_count"]):
            scope["a{0}".format(i)] = TypeAbstract()

        for i in range(hyperparameter_set["int_symbol_count"]):
            scope["i{0}".format(i)] = TypeInt()

        for i, j in enumerate(self._get_function_type_permutations(hyperparameter_set["max_function_argument_count"], hyperparameter_set["function_symbol_count"])):
            scope["f{0}".format(i)] = j

        return scope

    def _create_synthesizer(
        self,
        hyperparameter_set: HyperparameterSet,
        scope: Scope) -> TreeSynthesizerBase:
        return TreeSynthesizerStochastic(
            scope,
            hyperparameter_set["max_value_depth"],
            hyperparameter_set["tree_synthesis_probability"]
        )
    
    def _create_nl_presenter(self) -> NLPresenter:
        return NLPresenter(
            NLPresentationRule(
                ValueAnd(
                    ValueWildcard(),
                    ValueWildcard()
                ),
                "{lhs} and {rhs}"
            ),
            NLPresentationRule(
                ValueOr(
                    ValueWildcard(),
                    ValueWildcard()
                ),
                "{lhs} or {rhs}"
            ),
            NLPresentationRule(
                ValueImplies(
                    ValueWildcard(),
                    ValueWildcard()
                ),
                "if {lhs}, then {rhs}"
            ),
            NLPresentationRule(
                ValueNot(
                    ValueWildcard()
                ),
                "{arg} is false"
            ),
            NLPresentationRule(
                ValueLT(
                    ValueWildcard(),
                    ValueWildcard()
                ),
                "{lhs} is less than {rhs}"
            ),
            NLPresentationRule(
                ValueLE(
                    ValueWildcard(),
                    ValueWildcard()
                ),
                "{lhs} is less than or equal to {rhs}"
            ),
            NLPresentationRule(
                ValueGT(
                    ValueWildcard(),
                    ValueWildcard()
                ),
                "{lhs} is greater than {rhs}"
            ),
            NLPresentationRule(
                ValueGE(
                    ValueWildcard(),
                    ValueWildcard()
                ),
                "{lhs} is greater than or equal to {rhs}"
            ),
            NLPresentationRule(
                ValueNE(
                    ValueWildcard(),
                    ValueWildcard()
                ),
                "{lhs} is not equal to {rhs}"
            ),
            NLPresentationRule(
                ValueEQ(
                    ValueWildcard(),
                    ValueWildcard()
                ),
                "{lhs} is equal to {rhs}"
            )
        )

    def on_create_runner(
        self,
        hyperparameter_set: HyperparameterSet,
        tensorboard_output_dir: Optional[str]) -> RunnerBase:
        return RunnerNetworkTreeToNLPresentation(
            hyperparameter_set,
            self._create_network(hyperparameter_set),
            self._create_synthesizer(
                hyperparameter_set,
                self._create_scope(
                    hyperparameter_set
                )
            ),
            self._create_nl_presenter()
        )
