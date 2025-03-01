import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from misc.metric_tools import ConfuseMatrixMeter
from misc.torchutils import get_scheduler
from model.des_modules.descriptor_loss import DescriptorLoss
import numpy as np

logger = logging.getLogger('base')



class des(BaseModel):
    def __init__(self, opt):
        super(des, self).__init__(opt)
        # define network and load pretrained models
        self.netdes = self.set_device(networks.define_des(opt))

        # set loss and load resume state
        self.loss_type = opt['model_des']['loss_type']
        if self.loss_type == 'ce':
            #self.loss_func = nn.CrossEntropyLoss().to(self.device)
            self.loss_func =  DescriptorLoss().to(self.device)
        else:
            raise NotImplementedError()

        if self.opt['phase'] == 'train':
            self.netdes.train()
            # find the parameters to optimize
            optim_des_params = list(self.netdes.parameters())

            if opt['train']["optimizer"]["type"] == "adam":
                self.optdes = torch.optim.Adam(
                    optim_des_params, lr=opt['train']["optimizer"]["lr"])
            elif opt['train']["optimizer"]["type"] == "adamw":
                self.optdes = torch.optim.AdamW(
                    optim_des_params, lr=opt['train']["optimizer"]["lr"])
            else:
                raise NotImplementedError(
                    'Optimizer [{:s}] not implemented'.format(opt['train']["optimizer"]["type"]))
            self.log_dict = OrderedDict()

            # Define learning rate sheduler
            self.exp_lr_scheduler_netdes = get_scheduler(optimizer=self.optdes, args=opt['train'])
        else:
            self.netdes.eval()
            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()
        self.n = 0
        self.epoch_dis = 0

    # Feeding all data to the des model
    def feed_data(self, img1, img2, H, data):
        self.img1 = img1
        self.img2 = img2
        self.H = H
        self.data = self.set_device(data)
    # Optimize the parameters of the des model
    def optimize_parameters(self):
        self.optdes.zero_grad()
        self.img1 = self.img1.mean(1, keepdim=True)
        self.img2 = self.img2.mean(1, keepdim=True)
        self.pred_cm1 = self.netdes(self.img1)
        self.pred_cm2 = self.netdes(self.img2)
        l_des = self.loss_func(self.pred_cm1, self.pred_cm2,self.H)
        l_des.backward()
        self.optdes.step()
        self.log_dict['l_des'] = l_des.item()

    # Testing on given data
    def test(self):
        self.netdes.eval()
        self.img1 = self.img1.mean(1, keepdim=True)
        self.img2 = self.img2.mean(1, keepdim=True)
        with torch.no_grad():
            if isinstance(self.netdes, nn.DataParallel):
                self.pred_cm1 = self.netdes.module.forward(self.img1)
                self.pred_cm2 = self.netdes.module.forward(self.img2)
            else:
                self.pred_cm1 = self.netdes(self.img1)
                self.pred_cm2 = self.netdes(self.img2)
        l_des = self.loss_func(self.pred_cm1, self.pred_cm2,self.H)  # pm:b, 128, h, w; H:b, 3, 3
        self.log_dict['l_des'] = l_des.item()
        self.netdes.train()

    # Get current log
    def get_current_log(self):
        return self.log_dict
    def get_des(self,img):
        self.netdes.eval()
        img = img.mean(1, keepdim=True)
        with torch.no_grad():
            if isinstance(self.netdes, nn.DataParallel):
                des = self.netdes.module.forward(img)
            else:
                des = self.netdes(img)
        self.netdes.train()
        return des



    # Printing the des network
    def print_network(self):
        s, n = self.get_network_description(self.netdes)
        if isinstance(self.netdes, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netdes.__class__.__name__,
                                             self.netdes.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netdes.__class__.__name__)

        logger.info(
            'des Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    # Saving the network parameters
    def save_network(self, epoch, is_best_model=False):
        des_gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'des_model_E{}_gen.pth'.format(epoch))
        des_opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'des_model_E{}_opt.pth'.format(epoch))

        if is_best_model:
            best_des_gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_des_model_gen.pth'.format(epoch))
            best_des_opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_des_model_opt.pth'.format(epoch))

        # Save des model pareamters
        network = self.netdes
        if isinstance(self.netdes, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, des_gen_path)
        if is_best_model:
            torch.save(state_dict, best_des_gen_path)

        # Save des optimizer paramers
        opt_state = {'epoch': epoch,
                     'scheduler': None,
                     'optimizer': None}
        opt_state['optimizer'] = self.optdes.state_dict()
        torch.save(opt_state, des_opt_path)
        if is_best_model:
            torch.save(opt_state, best_des_opt_path)

        # Print info
        logger.info(
            'Saved current des model in [{:s}] ...'.format(des_gen_path))
        if is_best_model:
            logger.info(
                'Saved best des model in [{:s}] ...'.format(best_des_gen_path))

    # Loading pre-trained des network
    def load_network(self):
        load_path = self.opt['path_des']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for des model [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)

            # des model
            network = self.netdes
            if isinstance(self.netdes, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=True)

            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.optdes.load_state_dict(opt['optimizer'])
                self.begin_step = 0
                self.begin_epoch = opt['epoch']

    # Collecting status of the current running batch
    def _collect_running_batch_states(self):
        self.epoch_l += self.log_dict['l_des']
        self.n += 1

    # Collect the status of the epoch
    def _collect_epoch_states(self):
        self.log_dict['epoch_l_des'] = self.epoch_l / self.n

    # Rest all the performance metrics
    def _clear_cache(self):
        self.n = 0
        self.epoch_l = 0

    # Finctions related to learning rate sheduler
    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_netdes.step()