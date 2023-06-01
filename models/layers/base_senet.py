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
    
class SENETPlus(nn.Module):
    def __init__(self, filed_size,num_dim,num_group = 2,reduction_ratio=3):
        super(SENETPlus, self).__init__()
        self.filed_size = filed_size
        self.num_group = num_group
        input_size = 2*filed_size*num_group
        out_size = filed_size*num_dim
        self.reduction_size = max(1, input_size // reduction_ratio)

        self.senet = nn.Sequential(
            nn.Linear(input_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, out_size, bias=False),
        )
        weight_init(self.senet) 

        self.ln = nn.LayerNorm(num_dim)

    def squeeze(self,inputs):
        '''
        b,f,d
        '''
        b,f,d = inputs.shape
        assert inputs.shape[2]%self.num_group == 0
        inputs = inputs.reshape(b,f,self.num_group,d//self.num_group)  #b,f,g,k
        max_statistic = torch.max(inputs,dim=-1,keepdim=True)[0]
        avg_statistic = torch.mean(inputs,dim=-1,keepdim=True)
        z = torch.cat([max_statistic,avg_statistic],dim=-1) #b,f,2g
        z = z.reshape(b,-1)  #b,2fg
        return z

    def forward(self, inputs):
        '''
        b,f,d
        '''
        b,f,d = inputs.shape
        x = self.squeeze(inputs)
        weight = self.senet(x)
        out = torch.mul(inputs.reshape(b,-1), weight)
        out = out.reshape(b,f,-1)
        out = out + inputs
        out = self.ln(out)
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
    

class BilinearInteractionPlus(nn.Module):

    def __init__(self, filed_size, embedding_size, mlp_dim = 128,bilinear_type="interaction"):
        super(BilinearInteractionPlus, self).__init__()
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

        input_size = filed_size*(filed_size-1)//2
        self.mlp = nn.Linear(input_size,mlp_dim,bias=False)


    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        inputs = torch.split(inputs, 1, dim=1)
        if self.bilinear_type == "all":
            p = [torch.matmul(self.bilinear(v_i), v_j.transpose(1, 2))
                 for v_i, v_j in itertools.combinations(inputs, 2)]
        elif self.bilinear_type == "each":
            p = [torch.matmul(self.bilinear[i](inputs[i]), inputs[j].transpose(1, 2))
                 for i, j in itertools.combinations(range(len(inputs)), 2)]
        elif self.bilinear_type == "interaction":
            p = [torch.matmul(bilinear(v[0]), v[1].transpose(1, 2))
                 for v, bilinear in zip(itertools.combinations(inputs, 2), self.bilinear)]
        else:
            raise NotImplementedError
        
        out = torch.cat(p, dim=1).squeeze()
        return self.mlp(out)
    

if __name__ == '__main__':
    
    model = BilinearInteractionPlus(32,10)
    x = torch.randn((1000,32,10))
    model(x)