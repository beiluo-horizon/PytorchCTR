import pandas as pd 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import torch
class SparseFeature():

    def __init__(self,name,vocabulary_size,embedding_dim):
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        if embedding_dim == "auto":
            self.embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
    
class DenseFeature():

    def __init__(self,name,dense_emb,embedding_dim):
        self.name = name
        if dense_emb:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = 1
        # self.vocabulary_size = 1
    
def read_data(args):
    try:
        train_data = pd.read_csv(args.data_params['data_root']+'train_.csv')
        test_data = pd.read_csv(args.data_params['data_root']+'test_.csv')
        if args.data_params['use_valid']:
            valid_data = pd.read_csv(args.data_params['data_root']+'valid_.csv')
        sparse_features = args.data_params['feature_cols']['sparse_features']['name']
        dense_features = args.data_params['feature_cols']['dense_features']['name']
        target = ['label']
    except:
        train_path = args.data_params['train_data']
        test_path = args.data_params['test_data']
        valid_path = args.data_params['valid_data']

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        if args.data_params['use_valid']:
            valid_data = pd.read_csv(valid_path)

        sparse_features = args.data_params['feature_cols']['sparse_features']['name']
        dense_features = args.data_params['feature_cols']['dense_features']['name']
        
        #空值处理
        train_data[sparse_features] = train_data[sparse_features].fillna('-1', )
        train_data[dense_features] = train_data[dense_features].fillna(0, )
        test_data[sparse_features] = test_data[sparse_features].fillna('-1', )
        test_data[dense_features] = test_data[dense_features].fillna(0, )
        if args.data_params['use_valid']:
            valid_data[sparse_features] = valid_data[sparse_features].fillna('-1', )
            valid_data[dense_features] = valid_data[dense_features].fillna(0, )

        target = ['label']

        #处理稀疏Hot
        for feat in sparse_features:
            lbe = LabelEncoder()
            lbe.fit(pd.concat([train_data[feat],test_data[feat],valid_data[feat]]).unique())
            train_data[feat] = lbe.transform(train_data[feat])
            test_data[feat] = lbe.transform(test_data[feat])
            if args.data_params['use_valid']:
                valid_data[feat] = lbe.transform(valid_data[feat])
        
        if args.data_params['scaler'] == 'MinMaxScaler':
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            scaler = StandardScaler()
        scaler.fit(train_data[dense_features])
        train_data[dense_features] = scaler.transform(train_data[dense_features])
        test_data[dense_features] = scaler.transform(test_data[dense_features])
        if args.data_params['use_valid']:
            valid_data[dense_features] = scaler.transform(valid_data[dense_features])

        train_data.to_csv(args.data_params['data_root']+'train_.csv')
        test_data.to_csv(args.data_params['data_root']+'test_.csv')
        if args.data_params['use_valid']:
            valid_data.to_csv(args.data_params['data_root']+'valid_.csv')

    # 2.count #unique features for each sparse field,and record dense feature field name

    fix_SparseFeat = [SparseFeature(feat, len(pd.concat([train_data[feat],test_data[feat],valid_data[feat]]).unique()),embedding_dim=args.model_params['embedding_dim']) for feat in sparse_features]
    fix_DenseFeat = [DenseFeature(feat, args.data_params['dense_emb'], embedding_dim=args.model_params['embedding_dim']) for feat in dense_features]

    feature_columns = fix_SparseFeat + fix_DenseFeat

    feature_names = sparse_features + dense_features

    # 训练集
    train_data_dict = {name: train_data[name] for name in feature_names}
    test_data_dict = {name: test_data[name] for name in feature_names}

    if args.data_params['use_valid']:
        valid_data_dict = {name: valid_data[name] for name in feature_names}

    train_data_dict = [train_data_dict[feature] for feature in feature_names]
    train_label = train_data[args.data_params['label_col']['name']].values
    
    for i in range(len(train_data_dict)):
        if len(train_data_dict[i].shape) == 1:
            train_data_dict[i] = np.expand_dims(train_data_dict[i], axis=1)

    train_tensor_data = TensorDataset(
        torch.from_numpy(
            np.concatenate(train_data_dict, axis=-1),),
        torch.from_numpy(train_label))

    train_loader = DataLoader(
        dataset=train_tensor_data, shuffle=True, batch_size=args.model_params['batch_size'],num_workers = args.model_params['num_workers'],pin_memory=True,prefetch_factor=16)


    test_data_dict = [test_data_dict[feature] for feature in feature_names]
    test_label = test_data[args.data_params['label_col']['name']].values
    for i in range(len(test_data_dict)):
        if len(test_data_dict[i].shape) == 1:
            test_data_dict[i] = np.expand_dims(test_data_dict[i], axis=1)

    test_tensor_data = TensorDataset(
        torch.from_numpy(
            np.concatenate(test_data_dict, axis=-1)),
        torch.from_numpy(test_label))

    test_loader = DataLoader(
        dataset=test_tensor_data, shuffle=True, batch_size=args.model_params['batch_size'],num_workers = args.model_params['num_workers'],pin_memory=True,prefetch_factor=16)
    

    valid_data_dict = [valid_data_dict[feature] for feature in feature_names]
    valid_label = valid_data[args.data_params['label_col']['name']].values
    for i in range(len(valid_data_dict)):
        if len(valid_data_dict[i].shape) == 1:
            valid_data_dict[i] = np.expand_dims(valid_data_dict[i], axis=1)

    valid_tensor_data = TensorDataset(
        torch.from_numpy(
            np.concatenate(valid_data_dict, axis=-1)),
        torch.from_numpy(valid_label))

    valid_loader = DataLoader(
        dataset=valid_tensor_data, shuffle=True, batch_size=args.model_params['batch_size'],num_workers = args.model_params['num_workers'],pin_memory=True,prefetch_factor=16)
    
    return train_loader,test_loader,valid_loader,fix_SparseFeat,fix_DenseFeat