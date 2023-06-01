import os
import logging
import logging.config
import yaml
import glob
import json
from collections import OrderedDict
import h5py
import torch.nn as nn
def weight_init(net,xav_para = False):
    for layer in net:
        if isinstance(layer,nn.Linear):
            nn.init.xavier_normal_(layer.weight)
        elif isinstance(layer,nn.BatchNorm1d):
            nn.init.constant_(layer.weight,1)
            nn.init.constant_(layer.bias,0)
        elif isinstance(layer,nn.Conv1d):
            nn.init.kaiming_normal_(layer.weight)
        elif isinstance(layer,nn.Parameter):
            if xav_para == True:
                nn.init.xavier_normal_(layer)
            else:
                nn.init.zeros_(layer)

            

def load_config(args):
    params = load_model_config(args)
    data_params = load_dataset_config(args)
    return params,data_params

def load_model_config(args):
    model_configs = glob.glob(os.path.join(args.config_dir, "model_config.yaml"))  #读取config
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(args.config_dir))
    
    found_params = dict()
    config = model_configs[0]
    with open(config, 'r') as cfg:
        config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
        if args.model_setid in config_dict:
            found_params = config_dict[args.model_setid]
        else:
            assert f'expid={args.model_setid} is not valid in config.'

    found_params["model_id"] = args.model_setid
    return found_params

def load_dataset_config(args):
    params = {"data_setid": args.data_setid}
    dataset_configs = glob.glob(os.path.join(args.config_dir, "dataset_config.yaml"))
    dataset_config = dataset_configs[0]
    with open(dataset_config, "r") as cfg:
        config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
        if args.data_setid in config_dict:
            params.update(config_dict[args.data_setid])
            return params
    raise RuntimeError(f'dataset_id={args.data_setid} is not found in config.')