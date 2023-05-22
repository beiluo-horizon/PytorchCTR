import torch
import torch.nn as nn
from .basemodel import BaseModel
from .layers.base_mlp import MLP
from.layers.base_fm import BaseFM
from .layers.base_liner import Linear

class DeepFM(BaseModel):

    def __init__(self, fix_SparseFeat, fix_DenseFeat,args):
        super(DeepFM, self).__init__(fix_SparseFeat, fix_DenseFeat,args)

        _ = self.get_input_dim()
        _ = self.get_field_index()

        if args.model_params['loss'] == 'binary_crossentropy':
            self.pediction_layer = nn.Sequential(
                nn.Sigmoid()
            )

        self.dnn = MLP(self.input_dim, 
                        args.model_params['hidden_units'],
                        dropout=args.model_params['dropout'], 
                        batchnorm=args.model_params['batch_normal'], 
                        activation=args.model_params['hidden_activations'],
                        use_bias=False)
        self.dnn_linear = nn.Linear(args.model_params['hidden_units'][-1], 1, bias=False)
        self.add_regularization_net_weight(self.dnn.parameters())
        self.add_regularization_net_weight(self.dnn_linear.parameters())
        

        self.fm = BaseFM()
        self.linear = Linear(
            self.input_dim,
            use_bias=True
        )

        self.add_regularization_net_weight(self.linear.parameters())

    def forward(self, x):

        sparse_dict,dense_dict = self.get_embeddings(x)

        all_emb = [values for res,values in sparse_dict.items()]
        all_emb += [values for res,values in dense_dict.items()]
        all_emb = torch.cat(all_emb,dim=-1)

        fm_emb = [values.unsqueeze(1) for res,values in sparse_dict.items()]
        fm_emb = torch.cat(fm_emb,dim=1)

        fm_logit = self.fm(fm_emb)
        # linear_logit = self.linear(all_emb)
        dnn_logit = self.dnn(all_emb)
        dnn_logit = self.dnn_linear(dnn_logit)

        # logit = linear_logit + dnn_logit + fm_logit
        logit =  dnn_logit + fm_logit
        y_pred = self.pediction_layer(logit)
        reg_loss = self.get_regularization_loss()
        return y_pred,reg_loss
