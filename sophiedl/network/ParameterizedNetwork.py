import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O

class ParameterizedNetwork(nn.Module):
    def __init__(
        self,
        learning_rate,
        observation_space_shape,
        output_feature_count,
        layer_dimensions):
        assert len(layer_dimensions) > 0, "at least one layer dimension must be specified"

        super().__init__()

        self.learning_rate = learning_rate
        self.observation_space_shape = observation_space_shape
        self.output_feature_count = output_feature_count
        self.layer_dimensions = layer_dimensions

        self.input_layer = nn.Linear(*observation_space_shape, layer_dimensions[0])

        self.hidden_layers = []

        for i in range(len(layer_dimensions) - 1):
            hidden_layer = nn.Linear(layer_dimensions[i], layer_dimensions[i + 1])
            self.add_module("hidden_layers[{0}]".format(i), hidden_layer)
            self.hidden_layers.append(hidden_layer)

        self.output_layer = nn.Linear(layer_dimensions[-1], self.output_feature_count)

        self.optimizer = O.Adam(self.parameters(), lr = learning_rate)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu:0")
        self.to(self.device)

    def forward(self, observation):
        x = T.as_tensor(observation)
        x = F.relu(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        
        return self.output_layer(x)
