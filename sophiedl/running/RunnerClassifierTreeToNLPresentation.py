# Standard library
import random

# Typing
from typing import List, Sequence, Tuple

# PyTorch
import torch as T

# Internal
from ..hyperparameters.HyperparameterSet import HyperparameterSet
from ..network.OptimizedModule import OptimizedModule
from ..symbolic.synthesis.TreeSynthesizerBase import TreeSynthesizerBase
from ..symbolic.synthesis.TreeSynthesizerStochastic import TreeSynthesizerStochastic
from ..symbolic.tree.Scope import Scope
from ..symbolic.tree.TypeBase import TypeBase
from ..symbolic.tree.types import TypeAbstract, TypeBool, TypeFunction, TypeInt
from .RunnerClassifierBase import RunnerClassifierBase


class RunnerClassifierTreeToNLPresentation(RunnerClassifierBase):
    synthesizer: TreeSynthesizerBase
    nl_presenter: NLPresenter

    def __init__(self,
        hyperparameter_set: HyperparameterSet,
        network: OptimizedModule,
        synthesizer: TreeSynthesizerBase,
        nl_presenter: NLPresenter) -> None:
        super().__init__(
            hyperparameter_set,
            network
        )

        self.synthesizer = synthesizer
        self.nl_presenter = nl_presenter
    
    def _get_tensor_from_string(self,
        alphabet: str,
        value: str) -> T.Tensor:
        return T.as_tensor(
            [alphabet.index(i) for i in value],
            dtype = T.int
        )

    def _get_tensor_from_tree(
        value: TreeBase) -> T.Tensor:
        return self._get_tensor_from_string(
            self.nl_presenter.present(
                value
            )
        )
    
    def _get_tree_from_tensor(
        value: T.Tensor) -> TreeBase:
        

    def on_get_training_data(self) -> Sequence[Tuple[T.Tensor, T.Tensor]]:
        return [
            i for i
            in self.synthesizer.sample(self.hyperparameter_set["training_sample_size"])
        ]

    def on_get_testing_data(self) -> Sequence[Tuple[T.Tensor, T.Tensor]]:
        return [
            i for i
            in self.synthesizer.sample(self.hyperparameter_set["testing_sample_size"])
        ]

    def on_get_loss(self,
        predictions: T.Tensor,
        labels: T.Tensor) -> T.Tensor:
        pass