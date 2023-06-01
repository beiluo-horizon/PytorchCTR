from .MLP import FullMLP
from .WideDeep import WideDeep
from .FM import FM
from .DeepFM import DeepFM
from .xDeepFM import xDeepFM
from .DCN import DCN
from .AutoInt import AutoInt
from .AFN import AFN
from .GateNet import GateNet
from .FiBiNet import FiBiNet
from .FAT_DeepFFM import FATDeepFFM
from .FiBiNetPlus import FiBiNetPlus
from .ContextNet import ContextNet
from .DCNv2 import DCNv2
from .MaskNet import MaskNet 
from .FinalMLP import FinalMLP
import torch


def get_model(fix_SparseFeat, fix_DenseFeat,args):
    if args.model_params['model_name'] == 'FullMLP':
        model = FullMLP(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'WideDeep':
        model = WideDeep(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'FM':
        model = FM(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'DeepFM':
        model = DeepFM(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'xDeepFM':
        model = xDeepFM(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'DCN':
        model = DCN(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'AutoInt':
        model = AutoInt(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'AFN':
        model = AFN(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'GateNet':
        model = GateNet(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'FiBiNet':
        model = FiBiNet(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'FATDeepFFM':
        model = FATDeepFFM(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'FiBiNetPlus':
        model = FiBiNetPlus(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'ContextNet':
        model = ContextNet(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'DCNv2':
        model = DCNv2(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'MaskNet':
        model = MaskNet(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    elif args.model_params['model_name'] == 'FinalMLP':
        model = FinalMLP(fix_SparseFeat, fix_DenseFeat,args).to(args.device)
    else:
        raise ValueError('Unexpected model name')
    
    # model = torch.compile(model,mode="reduce-overhead")
    return model