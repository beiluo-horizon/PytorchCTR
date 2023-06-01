
import torch.nn as nn
import torch
from collections import OrderedDict
from utils.utils import weight_init

class ContextualEmbedding(nn.Module):
    def __init__(self,num_fields,feature_dim,increase=3):
        super(ContextualEmbedding, self).__init__()

        self.feature_dim = feature_dim
        self.num_fields = num_fields
        input_size = feature_dim*num_fields*increase
        proj_dict = nn.ModuleDict(
            {
                str(feat): nn.Linear(input_size, self.feature_dim,bias=False) for feat in range(num_fields)
            }
        )
        weight_init(proj_dict)

        self.proj_dict = proj_dict
        self.proj_share = nn.Sequential(
            nn.Linear(feature_dim*num_fields, feature_dim*num_fields*increase,bias=False),
            nn.ReLU()
        )
        weight_init(self.proj_share)

    def forward(self,x):
        '''
        b,f,e
        '''
        b,f,e = x.shape
        x_ = x.reshape(b,-1)
        share_hidden = self.proj_share(x_)
        ret = []
        for index in range(self.num_fields):
            ret.append(torch.mul(self.proj_dict[str(index)](share_hidden),x[:,index,:]).unsqueeze(1))

        outputs = torch.cat(ret,dim=1)

        return outputs


class ContextNetBlock(nn.Module):
    def __init__(self,num_fields,feature_dim,net_model='PFFN'):
        super(ContextNetBlock, self).__init__()

        self.net_model = net_model
        input_size = num_fields*feature_dim
        if self.net_model == 'PFFN':
            self.net = nn.Sequential(
                nn.Linear(feature_dim,feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim,feature_dim)
            )
            self.ln = nn.LayerNorm(feature_dim)

        elif self.net_model == 'SFFN':
            self.net = nn.Sequential(
                nn.Linear(feature_dim,feature_dim)
            )
            self.ln = nn.LayerNorm(feature_dim)

        else:
            raise ValueError('Unexpected model set')
        

        weight_init(self.net)


    def forward(self,x):
        '''
        b,f,e
        '''
        b,f,e = x.shape
        if self.net_model == 'PFFN':
            out = self.net(x)
            out += x
            out = self.ln(out)
        elif self.net_model == 'SFFN':
            out = self.net(x)
            out = self.ln(out)
        return out
    
if __name__ == '__main__':
    
    model = ContextualEmbedding(32,10)
    x = torch.randn((1000,32,10))
    model(x)