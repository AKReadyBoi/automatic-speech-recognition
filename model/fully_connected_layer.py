import math

import torch
import torch.nn as nn


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features und out_features müssen größer als 0 sein.")

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        output = torch.matmul(inputs, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output
