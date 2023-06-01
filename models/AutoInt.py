import torch
import torch.nn as nn
from .basemodel import BaseModel
from .layers.base_mlp import MLP
from .layers.base_liner import Linear
from .layers.base_tf import AutoIntBlock

class AutoInt(BaseModel):

    def __init__(self, fix_SparseFeat, fix_DenseFeat,args):
        super(AutoInt, self).__init__(fix_SparseFeat, fix_DenseFeat,args)

        _ = self.get_input_dim()
        _ = self.get_field_index()

        if args.model_params['loss'] == 'binary_crossentropy':
            self.pediction_layer = nn.Sequential(
                nn.Sigmoid()
            )

        self.add_regularization_net_weight(self.pediction_layer.parameters())

        out_feature = 0
        if self.args.model_params['use_dnn']:
            self.dnn = MLP(self.input_dim, 
                            args.model_params['hidden_units'],
                            dropout=args.model_params['dropout'], 
                            batchnorm=args.model_params['batch_normal'], 
                            activation=args.model_params['hidden_activations'],
                            use_bias=False)
            self.add_regularization_net_weight(self.dnn.parameters())
            out_feature += args.model_params['hidden_units'][-1]
        
        if self.args.model_params['use_linear']:
            self.linear = Linear(
                            self.input_dim,
                            dropout=args.model_params['linear_dropout'],
                            use_bias=True
                            )
            self.add_regularization_net_weight(self.linear.parameters())
        
        self.att_layer = nn.ModuleList(
            [AutoIntBlock(args) for _ in range(args.model_params['att_layer'])])
        out_feature += args.model_params['embedding_dim']*len(self.field_index)
        self.add_regularization_net_weight(self.att_layer.parameters())

        self.att_linear = nn.Linear(out_feature, 1, bias=False)
        self.add_regularization_net_weight(self.att_linear.parameters())
        nn.init.xavier_normal_(self.att_linear.weight)
        

    def forward(self, x):
        
        sparse_dict,dense_dict = self.get_embeddings(x)
        all_emb = [values for res,values in sparse_dict.items()]
        all_emb += [values for res,values in dense_dict.items()]
        all_emb = torch.cat(all_emb,dim=-1)

        fm_emb = [values.unsqueeze(1) for res,values in sparse_dict.items()]
        fm_emb = torch.cat(fm_emb,dim=1)
        att_logit = torch.cat(fm_emb,dim=1)

        for layer in range(self.args.model_params['att_layer']):
            att_logit = self.att_layer[layer](att_logit)
        att_logit = att_logit.reshape(att_logit.shape[0],-1)
        if self.args.model_params['use_dnn']:
            dnn_logit = self.dnn(all_emb)
            att_logit = torch.cat([dnn_logit,att_logit],dim=-1)
        
        att_logit = self.att_linear(att_logit)
        if self.args.model_params['use_linear']:
            linear_logit = self.linear(all_emb)
            att_logit += linear_logit
        y_pred = self.pediction_layer(att_logit)
        reg_loss = self.get_regularization_loss()
        return y_pred,reg_loss
