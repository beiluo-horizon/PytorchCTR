
import torch.nn as nn
from collections import OrderedDict
from utils.utils import weight_init
class Linear(nn.Module):
    def __init__(self, input_features,dropout=0.0, use_bias=False):
        super(Linear, self).__init__()
        modules = OrderedDict()
        previous_size = input_features
        modules["out"] = nn.Linear(previous_size, 1,bias=use_bias)
        if dropout:
            modules["drop"] = nn.Dropout(dropout)
        self._sequential = nn.Sequential(modules)
        weight_init(self._sequential)
    def forward(self, input):
        return self._sequential(input)
