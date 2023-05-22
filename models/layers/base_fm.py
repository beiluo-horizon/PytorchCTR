import torch.nn as nn
import torch

class BaseFM(nn.Module):
    """
      参考deepCTR
      输入
        (batch_size,field_size,embedding_size).
      输出
        logitc
    """
    def __init__(self):
        super(BaseFM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        return cross_term