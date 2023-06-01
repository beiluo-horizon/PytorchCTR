import torch
import torch.nn as nn
from .basemodel import BaseModel
from .layers.base_mlp import MLP
from.layers.base_liner import Linear
from .layers.base_cenet import CENet
from .layers.base_ffm import FFM

class FATDeepFFM(BaseModel):

    def __init__(self, fix_SparseFeat, fix_DenseFeat,args):
        super(FATDeepFFM, self).__init__(fix_SparseFeat, fix_DenseFeat,args)

        _ = self.get_input_dim()
        _ = self.get_field_index()

        if args.model_params['loss'] == 'binary_crossentropy':
            self.pediction_layer = nn.Sequential(
                nn.Sigmoid()
            )
        self.add_regularization_net_weight(self.pediction_layer.parameters())

        self.cenet = CENet(
            num_fields = len(self.field_index),
            feature_dim = self.args.model_params['ce_dim'],
            reduction= self.args.model_params['reduction']
        )
        self.add_regularization_net_weight(self.cenet.parameters())

        self.ffm = FFM(
            num_fields = len(self.field_index)
        )
        self.add_regularization_net_weight(self.ffm.parameters())

        input_feature = int(self.args.model_params['ce_dim']*len(self.field_index)*(len(self.field_index)-1)/2)
        
        self.dnn = MLP(input_feature, 
                        args.model_params['hidden_units'],
                        dropout=args.model_params['dropout'], 
                        batchnorm=args.model_params['batch_normal'], 
                        activation=args.model_params['hidden_activations'],
                        use_bias=False)
        
        self.dnn_linear = nn.Linear(args.model_params['hidden_units'][-1], 1, bias=False)
        self.add_regularization_net_weight(self.dnn.parameters())
        self.add_regularization_net_weight(self.dnn_linear.parameters())
        
        if self.args.model_params['use_linear']:
            self.linear = Linear(
                self.input_dim,
                dropout=args.model_params['linear_dropout'],
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

        ce_embs = self.cenet()
        dnn_input = self.ffm(fm_emb,ce_embs)
        dnn_logit = self.dnn(dnn_input)
        logit = self.dnn_linear(dnn_logit)

        if self.args.model_params['use_linear']:
            linear_logit = self.linear(all_emb)
            logit += linear_logit
        y_pred = self.pediction_layer(logit)
        reg_loss = self.get_regularization_loss()
        return y_pred,reg_loss
