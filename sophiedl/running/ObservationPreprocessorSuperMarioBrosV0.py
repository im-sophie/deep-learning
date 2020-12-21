# Typing
from typing import Any

# NumPy
import numpy as np # type: ignore

# PyTorch
import torch as T
import torchvision.transforms as transforms # type: ignore

# Internals
from .ObservationPreprocessorBase import ObservationPreprocessorBase

class ObservationPreprocessorSuperMarioBrosV0(ObservationPreprocessorBase):
    def __init__(self) -> None:
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(64),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def on_preprocess(self, observation: Any) -> T.Tensor:
        return self.transforms( # type: ignore
            observation.transpose(
                (2, 0, 1)
            ).copy()
        )
