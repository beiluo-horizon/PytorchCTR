
import torch.nn as nn
import torch
from collections import OrderedDict
from .base_mlp import MLP
from utils.utils import weight_init

class FeatureSelect(nn.Module):
    def __init__(self,all_dim,input_dim,hidden_dim,batch_normal):
        super(FeatureSelect, self).__init__()
        self.all_dim = all_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = MLP(self.input_dim, 
                    hidden_dim,
                    batchnorm = batch_normal, 
                    use_bias= False)

        
        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim[-1],self.all_dim,bias=False),
            nn.ReLU()
        )
        weight_init(self.out)

    def forward(self,x,emb):
        '''
        b,f,e
        '''
        b,f = x.shape
        x_ = emb.reshape(b,-1)
        weight = self.net(x_)
        weight = self.out(weight)*2
        out = torch.mul(x.reshape(b,-1),weight)
        return out


class BilinearFusion(nn.Module):
    def __init__(self,input_dim,head):
        super(BilinearFusion, self).__init__()
        assert input_dim%head == 0
        
        self.head = head
        self.dim = input_dim//head
        self.w1 = nn.ModuleList()
        self.w2 = nn.ModuleList()
        self.w3 = nn.ModuleList()
        for _ in range(head):
            self.w1.append(nn.Linear(self.dim,1,bias=True))
            self.w2.append(nn.Linear(self.dim,1,bias=False))
            self.w3.append(nn.Linear(self.dim,self.dim,bias=False))

        # 
        weight_init(self.w1)
        weight_init(self.w2)
        weight_init(self.w3)

    def forward(self,x1,x2):
        '''
        b,f*e
        '''
        b,d = x1.shape
        x1 = x1.reshape(b,self.head,-1)
        x2 = x2.reshape(b,self.head,-1)
        ret = []
        for index in range(self.head):
            ret.append((self.w1[index](x1[:,index,:])+self.w2[index](x2[:,index,:])+torch.mm(self.w3[index](x1[:,index,:]),x2[:,index,:].T).diag().unsqueeze(1)))
        
        ret = torch.cat(ret,dim=-1)
        out = torch.sum(ret,dim=1)
        return out
    
if __name__ == '__main__':
    
    model = BilinearFusion(320,10)
    x1 = torch.randn((1000,320))
    x2 = torch.randn((1000,320))
    model(x1,x2)