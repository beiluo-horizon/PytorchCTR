import torch
import torch.nn as nn
from .basemodel import BaseModel
from .layers.base_mlp import MLP
from.layers.base_liner import Linear
from .layers.base_cl import CrossLayerMatrix,CrossLayerLowRank,CrossNetMix
from utils.utils import weight_init

class DCNv2(BaseModel):

    def __init__(self, fix_SparseFeat, fix_DenseFeat,args):
        super(DCNv2, self).__init__(fix_SparseFeat, fix_DenseFeat,args)

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

        if self.args.model_params['use_linear']:
            self.linear = Linear(
                self.input_dim,
                dropout=args.model_params['linear_dropout'],
                use_bias=True
            )

            self.add_regularization_net_weight(self.linear.parameters())

        if args.model_params['net_model'] == 'Matrix':
            self.cl = CrossLayerMatrix(
                self.input_dim, num_layers=self.args.model_params['dcn_layers']
            )
        elif args.model_params['net_model'] == 'LowRank':
            self.cl = CrossLayerLowRank(
                self.input_dim, 
                num_layers= self.args.model_params['dcn_layers'],
                reduction = self.args.model_params['reduction'],
            )
        elif args.model_params['net_model'] == 'Mix':
            self.cl = CrossNetMix(
                input_dim = self.input_dim,
                num_experts = self.args.model_params['num_experts'],
                num_layers = self.args.model_params['dcn_layers'],
                reduction = self.args.model_params['reduction'],
            )
        else:
            raise ValueError('set error')
        self.add_regularization_net_weight(self.cl.parameters())

        if args.model_params['structure'] == 'stack':
            self.stack_linear = nn.Linear(args.model_params['hidden_units'][-1], 1, bias=False)
        elif args.model_params['structure'] == 'parallel':
            self.stack_linear = nn.Linear(args.model_params['hidden_units'][-1]+self.input_dim, 1, bias=False)
        else:
            raise ValueError('set error')
        self.add_regularization_net_weight(self.stack_linear.parameters())
        nn.init.xavier_normal_(self.stack_linear.weight)
    def forward(self, x):
        
        sparse_dict,dense_dict = self.get_embeddings(x)
        all_emb = [values for res,values in sparse_dict.items()]
        all_emb += [values for res,values in dense_dict.items()]
        all_emb = torch.cat(all_emb,dim=-1)

        fm_emb = [values.unsqueeze(1) for
                   res,values in sparse_dict.items()]
        fm_emb = torch.cat(fm_emb,dim=1)
        if self.args.model_params['structure'] == 'stack':
            cl_logit = self.cl(all_emb)
            dnn_logit = self.dnn(cl_logit)
            dnn_logit = self.stack_linear(dnn_logit)
        elif self.args.model_params['structure'] == 'parallel':
            cl_logit = self.cl(all_emb)
            dnn_logit = self.dnn(all_emb)
            dnn_logit = torch.cat([cl_logit,dnn_logit],dim=-1)
            dnn_logit = self.stack_linear(dnn_logit)

        if self.args.model_params['use_linear']:
            linear_logit = self.linear(all_emb)
            dnn_logit += linear_logit
        y_pred = self.pediction_layer(dnn_logit)
        reg_loss = self.get_regularization_loss()
        return y_pred,reg_loss
