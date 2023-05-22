import torch
import torch.nn as nn
from .basemodel import BaseModel
from .layers.base_mlp import MLP
from.layers.base_liner import Linear
from .layers.base_cl import CrossLayer
from utils.utils import weight_init

class DCN(BaseModel):

    def __init__(self, fix_SparseFeat, fix_DenseFeat,args):
        super(DCN, self).__init__(fix_SparseFeat, fix_DenseFeat,args)

        _ = self.get_input_dim()
        _ = self.get_field_index()

        if args.model_params['loss'] == 'binary_crossentropy':
            self.pediction_layer = nn.Sequential(
                nn.Sigmoid()
            )
        self.add_regularization_net_weight(self.pediction_layer.parameters())

        self.dnn = MLP(self.input_dim, 
                        args.model_params['hidden_units'],
                        dropout=args.model_params['dropout'], 
                        batchnorm=args.model_params['batch_normal'], 
                        activation=args.model_params['hidden_activations'],
                        use_bias=False)
        self.add_regularization_net_weight(self.dnn.parameters())

        self.linear = Linear(
            self.input_dim,
            dropout=args.model_params['linear_dropout'],
            use_bias=True
        )

        self.add_regularization_net_weight(self.linear.parameters())

        self.cl = CrossLayer(
            self.input_dim, num_layers=self.args.model_params['dcn_layers']
        )
        self.add_regularization_net_weight(self.cl.parameters())

        self.stack_linear = nn.Linear(args.model_params['hidden_units'][-1]+self.input_dim, 1, bias=False)
        self.add_regularization_net_weight(self.stack_linear.parameters())
        nn.init.xavier_normal_(self.stack_linear.weight)
    def forward(self, x):
        
        sparse_dict,dense_dict = self.get_embeddings(x)
        all_emb = [values for res,values in sparse_dict.items()]
        all_emb += [values for res,values in dense_dict.items()]
        all_emb = torch.cat(all_emb,dim=-1)

        fm_emb = [values.unsqueeze(1) for res,values in sparse_dict.items()]
        fm_emb = torch.cat(fm_emb,dim=1)

        linear_logit = self.linear(all_emb)
        dnn_logit = self.dnn(all_emb)
        cl_logit = self.cl(all_emb)
        stack_emb = torch.cat([dnn_logit,cl_logit],dim=-1)
        stack_logit = self.stack_linear(stack_emb)

        logit = linear_logit + stack_logit
        y_pred = self.pediction_layer(logit)
        reg_loss = self.get_regularization_loss()
        return y_pred,reg_loss
