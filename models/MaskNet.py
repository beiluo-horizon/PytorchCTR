import torch
import torch.nn as nn
from .basemodel import BaseModel
from .layers.base_mlp import MLP
from .layers.base_liner import Linear
from .layers.base_maskblock import MaskBlock

class MaskNet(BaseModel):

    def __init__(self, fix_SparseFeat, fix_DenseFeat,args):
        super(MaskNet, self).__init__(fix_SparseFeat, fix_DenseFeat,args)

        _ = self.get_input_dim()
        _ = self.get_field_index()

        if args.model_params['loss'] == 'binary_crossentropy':
            self.pediction_layer = nn.Sequential(
                nn.Sigmoid()
            )
        self.add_regularization_net_weight(self.pediction_layer.parameters())

        self.ln = nn.LayerNorm(args.model_params['embedding_dim'])
        self.mask_blocks = nn.ModuleList()
        for _ in range(args.model_params['block_num']):
            self.mask_blocks.append(MaskBlock(
                        input_dim = self.input_dim,
                        increase = args.model_params['increase_ratio'],
                        feature_dim = args.model_params['embedding_dim'],
                    ))
        self.add_regularization_net_weight(self.mask_blocks.parameters())
        if args.model_params['net_model'] == 'stack':
            self.dnn_linear = nn.Linear(len(self.field_index)*args.model_params['embedding_dim'], 1, bias=False)
            self.add_regularization_net_weight(self.dnn_linear.parameters())
        elif args.model_params['net_model'] == 'parallel':
            self.dnn = MLP(self.input_dim*args.model_params['block_num'], 
                        args.model_params['hidden_units'],
                        dropout=args.model_params['dropout'], 
                        batchnorm=args.model_params['batch_normal'], 
                        activation=args.model_params['hidden_activations'],
                        use_bias=False)
            self.dnn_linear = nn.Linear(args.model_params['hidden_units'][-1], 1, bias=False)
            self.add_regularization_net_weight(self.dnn.parameters())
            self.add_regularization_net_weight(self.dnn_linear.parameters())
        else:
            raise ValueError('Unexpected model set')

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
        b = fm_emb.shape[0]
        
        out = self.ln(fm_emb)
        if self.args.model_params['net_model'] == 'stack':
            for index in range(self.args.model_params['block_num']):
                out = self.mask_blocks[index](out,fm_emb)
            logit = self.dnn_linear(out.reshape(b,-1))
        elif self.args.model_params['net_model'] == 'parallel':
            ret = []
            for index in range(self.args.model_params['block_num']):
                ret.append(self.mask_blocks[index](out,fm_emb))
            ret = torch.cat(ret,dim=-1)
            logit = self.dnn(ret.reshape(b,-1))
            logit = self.dnn_linear(logit)

        if self.args.model_params['use_linear']:
            linear_logit = self.linear(all_emb)
            logit += linear_logit
        y_pred = self.pediction_layer(logit)
        reg_loss = self.get_regularization_loss()
        return y_pred,reg_loss
