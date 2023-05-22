# -*- coding:utf-8 -*-

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *
from tqdm import tqdm


class BaseModel(nn.Module):
    def __init__(self, fix_SparseFeat, fix_DenseFeat,args):

        super(BaseModel, self).__init__()
        self.reg_loss = torch.zeros((1,))
        self.aux_loss = torch.zeros((1,))
        self.args = args
        self.fix_SparseFeat = fix_SparseFeat
        self.fix_DenseFeat = fix_DenseFeat
        self.create_embedding_matrix(fix_SparseFeat)
        if args.data_params['dense_emb']:
            self.create_embedding_matrix_dense(fix_DenseFeat)
        self.regularization_weight = []
        self.regularization_net_weight = []
        self.add_regularization_embedding_weight(self.embedding_dict.parameters())
        if args.data_params['dense_emb']:
            self.add_regularization_embedding_weight(self.embedding_dict_dense.parameters())


    def create_embedding_matrix(self,fix_SparseFeat):
        embedding_dict = nn.ModuleDict(
            {
                feat.name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim) for feat in fix_SparseFeat
            }
        )
        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=0.001)

        self.embedding_dict = embedding_dict

    def create_embedding_matrix_dense(self,fix_DenseFeat):
        embedding_dict = nn.ModuleDict(
            {
                feat.name: nn.Embedding(1, feat.embedding_dim) for feat in fix_DenseFeat
            }
        )
        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=0.001)

        self.embedding_dict_dense = embedding_dict
    

    def add_regularization_embedding_weight(self, weight_list):
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append(weight_list)

    def add_regularization_net_weight(self, weight_list):
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        else:
            weight_list = list(weight_list)
        self.regularization_net_weight.append(weight_list)

    def get_input_dim(self):
        
        input_dim = 0
        for feat in self.fix_SparseFeat:
            input_dim += feat.embedding_dim
        
        if self.args.data_params['dense_emb']:
            for feat in self.fix_DenseFeat:
                input_dim += feat.embedding_dim
        else:
            input_dim += len(self.fix_DenseFeat)

        self.input_dim = input_dim
        return input_dim

    
    def get_field_index(self):

        field_index = {}
        start = 0
        for feat in self.fix_SparseFeat:
            field_index[feat.name] = (start,start+feat.embedding_dim)
            start = start+feat.embedding_dim

        for feat in self.fix_DenseFeat:
            field_index[feat.name] = (start,start+feat.embedding_dim)
            start = start+feat.embedding_dim

        self.field_index = field_index
        return field_index

    def get_regularization_loss(self):
        total_reg_loss = torch.zeros((1,), device=self.args.device)
        param_cnt = 0
        for weight_list in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                total_reg_loss += torch.sum(self.args.model_params['embedding_regularizer'] * torch.square(parameter))
        
        for weight_list in self.regularization_net_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                total_reg_loss += torch.sum(self.args.model_params['net_regularizer'] * torch.square(parameter))
        return total_reg_loss

    def get_embeddings(self,x):

        sparse_dict = {}
        dense_dict = {}
        start = 0
        for feat in self.fix_SparseFeat:
            sparse_dict[feat.name] = self.embedding_dict[feat.name](x[:,start])
            start += 1
        
        batch_size = x[:,start].shape[0]
        index = torch.zeros((batch_size)).long().to(self.args.device)
        for feat in self.fix_DenseFeat:
            if self.args.data_params['dense_emb']:
                sparse_dict[feat.name] = self.embedding_dict_dense[feat.name](index)*x[:,start].reshape(batch_size,-1)
                start += 1
            else:
                dense_dict[feat.name] = x[:,start].reshape(batch_size,-1)
                start += 1
        return sparse_dict,dense_dict

    # @staticmethod
    # def _accuracy_score(y_true, y_pred):
    #     return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))

    # def _get_metrics(self, metrics, set_eps=False):
    #     metrics_ = {}
    #     if metrics:
    #         for metric in metrics:
    #             if metric == "binary_crossentropy" or metric == "logloss":
    #                 if set_eps:
    #                     metrics_[metric] = self._log_loss
    #                 else:
    #                     metrics_[metric] = log_loss
    #             if metric == "auc":
    #                 metrics_[metric] = roc_auc_score
    #             if metric == "mse":
    #                 metrics_[metric] = mean_squared_error
    #             if metric == "accuracy" or metric == "acc":
    #                 metrics_[metric] = self._accuracy_score
    #             self.metrics_names.append(metric)
    #     return metrics_


    # @property
    # def embedding_size(self, ):
    #     feature_columns = self.dnn_feature_columns
    #     sparse_feature_columns = list(
    #         filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
    #         feature_columns) else []
    #     embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
    #     if len(embedding_size_set) > 1:
    #         raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
    #     return list(embedding_size_set)[0]