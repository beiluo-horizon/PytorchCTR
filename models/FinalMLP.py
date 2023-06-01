import torch
import torch.nn as nn
from .basemodel import BaseModel
from .layers.base_mlp import MLP
from.layers.base_liner import Linear
from .layers.base_final import FeatureSelect,BilinearFusion


class FinalMLP(BaseModel):

    def __init__(self, fix_SparseFeat, fix_DenseFeat,args):
        super(FinalMLP, self).__init__(fix_SparseFeat, fix_DenseFeat,args)

        _ = self.get_input_dim()
        _ = self.get_field_index()

        if args.model_params['loss'] == 'binary_crossentropy':
            self.pediction_layer = nn.Sequential(
                nn.Sigmoid()
            )
        self.add_regularization_net_weight(self.pediction_layer.parameters())

        assert self.input_dim%2 == 0
        self.select1 = FeatureSelect(
            all_dim = self.input_dim,
            input_dim = self.input_dim//2,
            hidden_dim = args.model_params['feature_select_hidden_units'],
            batch_normal = args.model_params['feature_select_batch_normal']
        )
        self.add_regularization_net_weight(self.select1.parameters())
        self.select2 = FeatureSelect(
            all_dim = self.input_dim,
            input_dim = self.input_dim//2,
            hidden_dim = args.model_params['feature_select_hidden_units'],
            batch_normal = args.model_params['feature_select_batch_normal']
        )
        self.add_regularization_net_weight(self.select2.parameters())
        
        self.dnn1 = MLP(self.input_dim, 
                        args.model_params['hidden_units'],
                        dropout=args.model_params['dropout'], 
                        batchnorm=args.model_params['batch_normal'], 
                        activation=args.model_params['hidden_activations'],
                        use_bias=False)
        self.add_regularization_net_weight(self.dnn1.parameters())
        self.dnn2 = MLP(self.input_dim, 
                        args.model_params['hidden_units'],
                        dropout=args.model_params['dropout'], 
                        batchnorm=args.model_params['batch_normal'], 
                        activation=args.model_params['hidden_activations'],
                        use_bias=False)
        self.add_regularization_net_weight(self.dnn2.parameters())

        self.bf = BilinearFusion(
            input_dim = args.model_params['hidden_units'][-1],
            head = args.model_params['num_heads']
        )
        self.add_regularization_net_weight(self.bf.parameters())
        
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

        a,b = torch.split(all_emb,[self.input_dim//2,self.input_dim//2],dim=1)
        out1 = self.select1(all_emb,a)
        out2 = self.select2(all_emb,b)

        out1 = self.dnn1(out1)
        out2 = self.dnn2(out2)
        
        logit = self.bf(out1,out2)
        if self.args.model_params['use_linear']:
            linear_logit = self.linear(all_emb)
            logit += linear_logit
        y_pred = self.pediction_layer(logit)
        reg_loss = self.get_regularization_loss()
        return y_pred,reg_loss
