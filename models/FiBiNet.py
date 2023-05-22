import torch
import torch.nn as nn
from .basemodel import BaseModel
from .layers.base_mlp import MLP
from.layers.base_liner import Linear
from .layers.base_senet import SENET,BilinearInteraction

class FiBiNet(BaseModel):

    def __init__(self, fix_SparseFeat, fix_DenseFeat,args):
        super(FiBiNet, self).__init__(fix_SparseFeat, fix_DenseFeat,args)

        _ = self.get_input_dim()
        _ = self.get_field_index()

        if args.model_params['loss'] == 'binary_crossentropy':
            self.pediction_layer = nn.Sequential(
                nn.Sigmoid()
            )
        self.add_regularization_net_weight(self.pediction_layer.parameters())

        self.senet = SENET(
            filed_size= len(self.field_index)
        )
        self.add_regularization_net_weight(self.pediction_layer.parameters())
        self.bi_inter = BilinearInteraction(
            filed_size= len(self.field_index),
            embedding_size = args.model_params['embedding_dim'],
            bilinear_type = args.model_params['bilinear_type']
        )
        self.add_regularization_net_weight(self.pediction_layer.parameters())

        input_feature = 0
        input_feature = len(self.field_index) * (len(self.field_index) - 1) * args.model_params['embedding_dim']
        
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

        senet_output = self.senet(fm_emb)
        senet_bilinear_out = self.bi_inter(senet_output)
        bilinear_out = self.bi_inter(fm_emb)

        dnn_logit = self.dnn(torch.cat([senet_bilinear_out.reshape(senet_bilinear_out.shape[0],-1),bilinear_out.reshape(bilinear_out.shape[0],-1)],dim=-1))
        logit = self.dnn_linear(dnn_logit)

        if self.args.model_params['use_linear']:
            linear_logit = self.linear(all_emb)
            logit += linear_logit
        y_pred = self.pediction_layer(logit)
        reg_loss = self.get_regularization_loss()
        return y_pred,reg_loss
