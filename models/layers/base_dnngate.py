"""
Multilayer perceptron torch module.
"""
from collections import OrderedDict

import torch.nn as nn
from utils.utils import weight_init
import torch

class HiddenGate(nn.Module):
    """Multilayer perceptron torch module.

    Parameters
    ----------
    input_size : int
        Size of input.

    hidden_layers : iterable
        Hidden layer sizes.

    dropout : float
        Dropout rate.

    activation : str
        Name of activation function. ReLU, PReLU and Sigmoid are supported.
    """
    def __init__(self, input_size,num_layers,
                 dropout=0.0, activation='tanh',use_bias=False):
        super(HiddenGate, self).__init__()
        previous_size = input_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [nn.Linear(previous_size, previous_size,bias=use_bias) for _ in range(num_layers)]
        )
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        if dropout:
            self.dropout = nn.Dropout(dropout)

        weight_init(self.layers)

    def forward(self, input):

        for layer in range(self.num_layers):
            input = torch.mul(input,self.dropout(self.activation(self.layers[layer](input))))

        return input
