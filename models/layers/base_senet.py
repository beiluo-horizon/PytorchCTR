import torch
import torch.nn as nn
from utils.utils import weight_init
import itertools

class SENET(nn.Module):
    def __init__(self, filed_size, reduction_ratio=3):
        super(SENET, self).__init__()
        self.filed_size = filed_size
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.senet = nn.Sequential(
            nn.Linear(self.filed_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, self.filed_size, bias=False),
            nn.ReLU()
        )
        weight_init(self.senet)

    def forward(self, inputs):
        '''
        b,f,d
        '''
        reduce_input = torch.mean(inputs, dim=-1)
        weight = self.senet(reduce_input)
        out = torch.mul(inputs, torch.unsqueeze(weight, dim=2))

        return out
    

class BilinearInteraction(nn.Module):

    def __init__(self, filed_size, embedding_size, bilinear_type="interaction"):
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        self.bilinear = nn.ModuleList()
        if self.bilinear_type == "all":
            self.bilinear = nn.Linear(
                embedding_size, embedding_size, bias=False)  #(vi,w)*vj
        elif self.bilinear_type == "each":
            for _ in range(filed_size):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        elif self.bilinear_type == "interaction":
            for _, _ in itertools.combinations(range(filed_size), 2):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        else:
            raise NotImplementedError
        weight_init(self.bilinear)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        inputs = torch.split(inputs, 1, dim=1)
        if self.bilinear_type == "all":
            p = [torch.mul(self.bilinear(v_i), v_j)
                 for v_i, v_j in itertools.combinations(inputs, 2)]
        elif self.bilinear_type == "each":
            p = [torch.mul(self.bilinear[i](inputs[i]), inputs[j])
                 for i, j in itertools.combinations(range(len(inputs)), 2)]
        elif self.bilinear_type == "interaction":
            p = [torch.mul(bilinear(v[0]), v[1])
                 for v, bilinear in zip(itertools.combinations(inputs, 2), self.bilinear)]
        else:
            raise NotImplementedError
        return torch.cat(p, dim=1)