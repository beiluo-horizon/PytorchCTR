import torch
import torch.nn as nn
from .basemodel import BaseModel
from .layers.base_mlp import MLP
from.layers.base_liner import Linear
from .layers.base_senet import SENETPlus,BilinearInteractionPlus

class FiBiNetPlus(BaseModel):

    def __init__(self, fix_SparseFeat, fix_DenseFeat,args):
        super(FiBiNetPlus, self).__init__(fix_SparseFeat, fix_DenseFeat,args)

        _ = self.get_input_dim()
        _ = self.get_field_index()

        if args.model_params['loss'] == 'binary_crossentropy':
            self.pediction_layer = nn.Sequential(
                nn.Sigmoid()
            )
        self.add_regularization_net_weight(self.pediction_layer.parameters())

        self.bn = nn.BatchNorm1d(args.model_params['embedding_dim']*len(self.field_index))
        self.ln = nn.LayerNorm(args.model_params['embedding_dim'])

        self.senet = SENETPlus(
            filed_size = len(self.field_index),
            num_dim = args.model_params['embedding_dim'],
            num_group = args.model_params['num_group'],
            reduction_ratio = args.model_params['reduction_ratio'],
        )
        self.add_regularization_net_weight(self.senet.parameters())

        self.bi_inter = BilinearInteractionPlus(
            filed_size= len(self.field_index),
            embedding_size = args.model_params['embedding_dim'],
            mlp_dim = args.model_params['bi_mlp_dim'],
            bilinear_type = args.model_params['bilinear_type']
        )
        self.add_regularization_net_weight(self.bi_inter.parameters())

        input_feature = 0
        input_feature += args.model_params['bi_mlp_dim']
        input_feature += len(self.field_index) * args.model_params['embedding_dim']
        
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


    def feat_norm(self,sparse_dict,dense_dict):

        sparse_features = [values.unsqueeze(1) for res,values in sparse_dict.items()]
        dense_features = [values.unsqueeze(1) for res,values in dense_dict.items()]
        sparse_features = torch.cat(sparse_features,dim=1)

        b,f,d = sparse_features.shape
        sparse_features = self.bn(sparse_features.reshape(b,-1))
        if len(dense_features) != 0:
            dense_features = torch.cat(dense_features,dim=1)
            dense_features = self.ln(dense_features)
            all_emb = torch.cat([sparse_features,dense_features.reshape(b,-1)],dim=-1)
        else:
            all_emb = torch.cat([sparse_features],dim=-1)
        fm_emb = sparse_features.reshape(b,f,-1)

        return all_emb,fm_emb
    def forward(self, x):
        
        sparse_dict,dense_dict = self.get_embeddings(x)
        all_emb,fm_emb = self.feat_norm(sparse_dict,dense_dict)

        senet_output = self.senet(fm_emb)
        bilinear_out = self.bi_inter(fm_emb)
        b = senet_output.shape[0]
        dnn_logit = self.dnn(torch.cat([senet_output.reshape(b,-1),bilinear_out.reshape(b,-1)],dim=-1))
        logit = self.dnn_linear(dnn_logit)

        if self.args.model_params['use_linear']:
            linear_logit = self.linear(all_emb)
            logit += linear_logit
        y_pred = self.pediction_layer(logit)
        reg_loss = self.get_regularization_loss()
        return y_pred,reg_loss
