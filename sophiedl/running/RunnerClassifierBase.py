# Standard library
import abc

# Typing
from typing import Generator, List, Optional, Sequence, Tuple

# PyTorch
import torch as T
from torch.utils.tensorboard import SummaryWriter

# TQDM
import tqdm # type: ignore

# Internal
from ..hyperparameters.HyperparameterSet import HyperparameterSet
from ..network.OptimizedModule import OptimizedModule
from .RunnerBase import RunnerBase

class RunnerClassifierBase(RunnerBase):
    def __init__(self,
        hyperparameter_set: HyperparameterSet,
        network: OptimizedModule) -> None:
        super().__init__(hyperparameter_set)
        self.network = network
    
    @abc.abstractmethod
    def on_get_training_data(self) -> Sequence[Tuple[T.Tensor, T.Tensor]]:
        pass

    @abc.abstractmethod
    def on_get_testing_data(self) -> Sequence[Tuple[T.Tensor, T.Tensor]]:
        pass

    @abc.abstractmethod
    def on_get_loss(self, predictions: T.Tensor, labels: T.Tensor) -> T.Tensor:
        pass

    def _train(self) -> None:
        print("Training for {0} epoch{1}...".format(self.hyperparameter_set["epoch_count"], "" if self.hyperparameter_set["epoch_count"] == 1 else "s"))
        self.network.train()

        accuracy_history: List[float] = []
        training_data_length = len(self.on_get_training_data())

        with tqdm.tqdm(
            total = self.hyperparameter_set["epoch_count"] * training_data_length,
            bar_format = "{percentage:.1f} {bar} Accuracy: {postfix[0]:8.6f}, moving average ({postfix[1]}): {postfix[2]:8.6f}, elapsed: {postfix[3]}/{postfix[4]}",
            postfix = [0, self.moving_average_window, 0, 0, self.hyperparameter_set["epoch_count"]]) as t:
            for epoch_index in range(self.hyperparameter_set["epoch_count"]):
                epoch_loss = 0.

                training_data = self.on_get_training_data()

                for batch_index, (inputs, labels) in enumerate(training_data):
                    self.network.optimizer.zero_grad()
                    labels = labels.to(self.network.device)
                    predictions = self.network.forward(inputs.to(self.network.device))
                    classes = T.argmax(predictions, dim = 1)
                    wrong = T.where(
                        classes != labels,
                        T.Tensor([1.]).to(self.network.device),
                        T.Tensor([0.]).to(self.network.device)
                    )
                    accuracy = 1 - (T.sum(wrong) / self.hyperparameter_set["batch_size_training"])
                    loss = self.on_get_loss(predictions, labels)
                    epoch_loss += loss.item()
                    loss.backward() # type: ignore
                    self.network.optimizer.step()
                    t.postfix[0] = accuracy
                    t.postfix[2] = sum(accuracy_history) / len(accuracy_history) if len(accuracy_history) > 0 else 0
                    t.postfix[3] = epoch_index
                    accuracy_history.append(accuracy)
                    while len(accuracy_history) > self.moving_average_window:
                        del accuracy_history[0]
                    t.update()

    def _test(self) -> None:
        print("Testing...")
        self.network.eval()

        accuracy_history: List[float] = []

        for batch_index, (inputs, labels) in enumerate(self.on_get_testing_data()):
            self.network.optimizer.zero_grad()
            labels = labels.to(self.network.device)
            predictions = self.network.forward(inputs.to(self.network.device))
            classes = T.argmax(predictions, dim = 1)
            wrong = T.where(
                classes != labels,
                T.Tensor([1.]).to(self.network.device),
                T.Tensor([0.]).to(self.network.device)
            )
            accuracy = 1 - (T.sum(wrong) / self.hyperparameter_set["batch_size_testing"])
            accuracy_history.append(accuracy)
        
        print("Accuracy: {0:8.6f}".format(sum(accuracy_history) / len(accuracy_history)))

    def on_run(self, tensorboard_summary_writer: Optional[SummaryWriter]) -> None:
        self._train()
        self._test()
