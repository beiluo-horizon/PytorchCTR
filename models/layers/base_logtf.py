import torch
import torch.nn as nn

class LogTransformLayer(nn.Module):
  
    def __init__(self, field_size, embedding_size, ltl_hidden_size):
        super(LogTransformLayer, self).__init__()

        self.ltl_weights = nn.Parameter(torch.Tensor(field_size, ltl_hidden_size))
        self.ltl_biases = nn.Parameter(torch.Tensor(1, 1, ltl_hidden_size))
        self.bn = nn.ModuleList([nn.BatchNorm1d(embedding_size) for i in range(2)])
        nn.init.normal_(self.ltl_weights, mean=0.0, std=0.1)
        nn.init.zeros_(self.ltl_biases, )

    def forward(self, inputs):
        afn_input = torch.clamp(torch.abs(inputs), min=1e-7, max=float("Inf"))
        afn_input_trans = torch.transpose(afn_input, 1, 2)
        ltl_result = torch.log(afn_input_trans)
        ltl_result = self.bn[0](ltl_result)
        ltl_result = torch.matmul(ltl_result, self.ltl_weights) + self.ltl_biases
        ltl_result = torch.exp(ltl_result)
        ltl_result = self.bn[1](ltl_result)
        ltl_result = torch.flatten(ltl_result, start_dim=1)
        return ltl_result