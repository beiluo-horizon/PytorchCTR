
import torch.nn as nn
import torch
from collections import OrderedDict
from utils.utils import weight_init

class CENet(nn.Module):
    def __init__(self,num_fields,feature_dim, reduction=8):
        super(CENet, self).__init__()

        self.feature_dim = feature_dim
        self.num_fields = num_fields
    
        embedding_dict = nn.ModuleDict(
            {
                str(feat): nn.Embedding(self.num_fields, self.feature_dim) for feat in range(num_fields)
            }
        )
        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=0.001)

        self.embedding = embedding_dict

        inputs_num_fields =  num_fields*num_fields
        reduced_num_fields = inputs_num_fields // reduction

        # self.pooling = nn.layer.AdaptiveAvgPool1D(output_size=1)
        self.pooling = nn.AdaptiveMaxPool1d(output_size=1)

        self.fc = nn.Sequential(
            nn.Linear(inputs_num_fields,reduced_num_fields),
            nn.ReLU(),
            nn.Linear(reduced_num_fields,inputs_num_fields),
            nn.ReLU()
        )
        weight_init(self.fc)

    def forward(self):
        '''
        b,f,e
        '''


        pool_embs = []
        embs = []
        for index in range(self.num_fields):
            pool_embs.append(self.pooling(self.embedding[str(index)].weight))
            embs.append(self.embedding[str(index)].weight)
        fc_embs = torch.cat(pool_embs,dim=-1)
        fc_embs = torch.flatten(fc_embs)
        embs = torch.cat(embs,dim=0)

        attn_w = self.fc(fc_embs)
        outputs = torch.mul(attn_w.reshape(-1,1).repeat(1,self.feature_dim),embs)
        outputs = outputs.reshape(self.num_fields,self.num_fields,self.feature_dim)

        return outputs


if __name__ == '__main__':
    
    model = CENet(32,10)
    model()