import torch.nn as nn
import torch
from utils.utils import weight_init

'''
为DCN以及DCNv2的基本层
'''

class CrossLayer(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    """
    def __init__(self, input_dim, num_layers=2):
        super(CrossLayer, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor((input_dim,))) for _ in range(num_layers)
        ])
        weight_init(self.w,xav_para=True)
        weight_init(self.b)
    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class CrossLayerMatrix(nn.Module):
    '''
    矩阵形式交互
    '''
    def __init__(self, input_dim, num_layers=2):
        super(CrossLayerMatrix, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(input_dim,input_dim)) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(input_dim,)) for _ in range(num_layers)
        ])
        weight_init(self.w,xav_para=True)
        weight_init(self.b)

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            xw = torch.mm(self.w[i],x.T)
            x = x0 * (xw.T + self.b[i]) + x
        return x
    

class CrossLayerLowRank(nn.Module):
    '''
    矩阵分解形式交互
    '''
    def __init__(self, input_dim, num_layers=2,reduction = 8):
        super(CrossLayerLowRank, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        reduce_size = self.input_dim//reduction
        self.w1 = torch.nn.ModuleList([
             torch.nn.Linear(input_dim, reduce_size, bias=False) for _ in range(num_layers)
        ])
        self.w2 = torch.nn.ModuleList([
             torch.nn.Linear(reduce_size, input_dim, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(input_dim,)) for _ in range(num_layers)
        ])
        weight_init(self.w1)
        weight_init(self.w2)
        weight_init(self.b)

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            xw = self.w1[i](x)
            xw = self.w2[i](xw)
            x = x0 * (xw + self.b[i]) + x
        return x
    

class CrossNetMix(nn.Module):
    '''
    Moe形式交互
    '''
    def __init__(self, input_dim, num_experts=4, num_layers=2,reduction = 8):
        super(CrossNetMix, self).__init__()

        self.num_layers = num_layers
        self.num_experts = num_experts
        reduce_size = input_dim//reduction

        self.U = torch.nn.ParameterList([
             torch.nn.Parameter(torch.Tensor(num_experts,input_dim,reduce_size)) for _ in range(num_layers)
        ])
        self.V = torch.nn.ParameterList([
             torch.nn.Parameter(torch.Tensor(num_experts,input_dim,reduce_size)) for _ in range(num_layers)
         ])
        self.C = torch.nn.ParameterList([
             torch.nn.Parameter(torch.Tensor(num_experts,reduce_size,reduce_size)) for _ in range(num_layers)
         ])
        self.G = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for i in range(self.num_experts)])

        self.b = nn.ParameterList([torch.nn.Parameter(torch.Tensor(input_dim,)) for _ in range(num_layers)])

        weight_init(self.U,xav_para=True)
        weight_init(self.V,xav_para=True)
        weight_init(self.C,xav_para=True)
        weight_init(self.G)
        weight_init(self.b)

    def forward(self, inputs):
        b = inputs.shape[0]
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.num_layers):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                gating_score_of_experts.append(self.G[expert_id](x_l.squeeze(2)))  
                v_x = torch.matmul(self.V[i][expert_id].T, x_l) 
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C[i][expert_id], v_x)
                v_x = torch.tanh(v_x)
                uv_x = torch.matmul(self.U[i][expert_id], v_x) 
                dot_ = uv_x + self.b[i].unsqueeze(1).unsqueeze(0).repeat(b,1,1)
                dot_ = x_0 * dot_  

                output_of_experts.append(dot_.squeeze(2))
            output_of_experts = torch.stack(output_of_experts, 2) 
            gating_score_of_experts = torch.stack(gating_score_of_experts, 1)  
            moe_out = torch.matmul(output_of_experts, gating_score_of_experts.softmax(1))
            x_l = moe_out + x_l
        x_l = x_l.squeeze()
        return x_l