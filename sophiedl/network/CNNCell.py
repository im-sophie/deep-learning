# PyTorch
import torch.nn as nn

class CNNCell(nn.Sequential):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels = in_channels,
                kernel_size = kernel_size,
                out_channels = out_channels,
                stride = stride
            ),
            nn.BatchNorm2d(
                num_features = out_channels
            ), # type: ignore
            nn.ReLU()
        )
