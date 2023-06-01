
import torch
import torch.nn.functional as F
from torch import nn, optim
import time
class LF:
    def __init__(self,
                 recmodel,
                 args):
        self.model = recmodel
        self.lr = args.model_params['learning_rate']
        if args.model_params['optimizer'] == 'adam':
            self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        elif args.model_params['optimizer'] == 'SGD':
            self.opt = optim.SGD(self.parameters(), lr=self.lr)
        elif args.model_params['optimizer'] == "adagrad":
            self.opt = optim.Adagrad(self.parameters(),lr=self.lr)  # 0.01
        elif args.model_params['optimizer'] =="rmsprop":
            self.opt = optim.RMSprop(self.parameters(),lr=self.lr)
        else:
            raise NotImplementedError
        
        if args.model_params['loss'] == "binary_crossentropy":
            self.loss_func = F.binary_cross_entropy
        elif args.model_params['loss'] == "mse":
            self.loss_func = nn.MSELoss()
        elif args.model_params['loss'] == "mae":
            self.loss_func = nn.L1Loss()
        else:
            raise NotImplementedError
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,T_max = 200,eta_min=0.0001)

    def stageOne(self,x, y):
        batch_size = x.shape[0]
        y_pred,reg_loss = self.model(x)
        self.opt.zero_grad()
        loss = self.loss_func(y_pred.squeeze(),y.squeeze())     
        loss = loss + reg_loss
        loss.backward()
        self.opt.step()
        # self.scheduler.step()
        return loss.cpu().item()


