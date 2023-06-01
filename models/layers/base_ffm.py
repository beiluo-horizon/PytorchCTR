import torch.nn as nn
import torch
from collections import OrderedDict
from utils.utils import weight_init

class FFM(nn.Module):
    def __init__(self, num_fields):
        super(FFM, self).__init__()
        self.num_fields = num_fields

    def forward(self,x,feat_embedding):
        '''
        b,f,e
        f,f,k
        '''
 
        field_aware_interaction_list = []
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                inter = feat_embedding[i,j,:] * feat_embedding[j, i, :]
                field_aware_out = torch.mul(x[:,i,:],x[:,j,:])
                field_aware_interaction_list.append(torch.mul(inter.reshape(1,-1).repeat(field_aware_out.shape[0],1),field_aware_out))

        ffm_out = torch.cat(field_aware_interaction_list,dim=-1)

        return ffm_out
    

if __name__ == '__main__':
    
    model= FFM(32,10)
    x = torch.randn(1000,32,10)
    fiele = torch.randn(32,32,10)
    model(x,fiele)