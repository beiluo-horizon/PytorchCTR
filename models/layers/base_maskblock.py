
import torch.nn as nn
import torch
from collections import OrderedDict
from utils.utils import weight_init

class InstanceMask(nn.Module):
    def __init__(self,input_dim,increase=3):
        super(InstanceMask, self).__init__()
        self.input_dim = input_dim
        self.wide_size = input_dim*increase

        self.net = nn.Sequential(
            nn.Linear(self.input_dim,self.wide_size,bias=True),
            nn.ReLU(),
            nn.Linear(self.wide_size,self.input_dim)
        )

        weight_init(self.net)

    def forward(self,x,emb):
        '''
        b,f,e
        输入是没有经过LN的原始emb
        '''
        b,f,e = x.shape
        x_ = emb.reshape(b,-1)
        weight = self.net(x_)
        out = torch.mul(x.reshape(b,-1),weight)
        return out.reshape(b,f,e)


class MaskBlock(nn.Module):
    def __init__(self,input_dim,increase,feature_dim):
        super(MaskBlock, self).__init__()
        self.instance_mask = InstanceMask(
            input_dim =input_dim,
            increase = increase,
        )
        self.net = nn.Sequential(
            nn.Linear(feature_dim,feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )
        weight_init(self.net)


    def forward(self,x,emb):
        '''
        b,f,e
        '''
        b,f,e = x.shape
        out = self.instance_mask(x,emb)
        out = self.net(out)
        return out
    
if __name__ == '__main__':
    
    model = MaskBlock(320,3,10)
    x = torch.randn((1000,32,10))
    model(x)