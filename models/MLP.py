
import torch
import torch.nn as nn
from .basemodel import BaseModel
from .layers.base_mlp import MLP

class FullMLP(BaseModel):

    def __init__(self, fix_SparseFeat, fix_DenseFeat,args):
        super(FullMLP, self).__init__(fix_SparseFeat, fix_DenseFeat,args)
        _ = self.get_input_dim()
        _ = self.get_field_index()

        if args.model_params['loss'] == 'binary_crossentropy':
            self.pediction_layer = nn.Sequential(
                nn.Linear(self.args.model_params['mlp1_hidden_units'][-1],1),
                nn.Sigmoid()
            )
        
        self.net = MLP(self.input_dim, 
                        args.model_params['mlp1_hidden_units'],
                        dropout=args.model_params['mlp1_dropout'], 
                        batchnorm=args.model_params['batch_normal'], 
                        activation=args.model_params['hidden_activations'])
        
        self.add_regularization_net_weight(self.net.parameters())
        self.add_regularization_net_weight(self.pediction_layer.parameters())

    def forward(self, x):
        sparse_dict,dense_dict = self.get_embeddings(x)
        all_emb = [values for res,values in sparse_dict.items()]
        all_emb += [values for res,values in dense_dict.items()]
        all_emb = torch.cat(all_emb,dim=-1)

        fm_emb = [values.unsqueeze(1) for res,values in sparse_dict.items()]
        fm_emb = torch.cat(fm_emb,dim=1)
        logit = self.net(all_emb)
        y_pred = self.pediction_layer(logit)
        reg_loss = self.get_regularization_loss()
        return y_pred,reg_loss
