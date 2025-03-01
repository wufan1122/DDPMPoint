import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from misc.metric_tools import ConfuseMatrixMeter
from misc.torchutils import get_scheduler
logger = logging.getLogger('base')


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    minus_ten = torch.full_like(scores, -10)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, minus_ten, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, minus_ten)
def normalize_scores(scores):
    """
    对 scores 进行归一化处理，使其值在 [0, 1] 范围内。
    """
    min_val = scores.min()
    max_val = scores.max()
    if max_val - min_val != 0:
        normalized_scores = (scores - min_val) / (max_val - min_val)
    else:
        normalized_scores = scores - min_val  # 如果所有值相同，归一化为0
    return normalized_scores

def get_current_kpts(scores,k=1000):
    normalized_scores = normalize_scores(scores)
    
    
    pm = normalized_scores
    scores_flat = pm.view(-1) 
    sorted_scores, _ = torch.sort(scores_flat, descending=True)
    tenth_largest = sorted_scores[k]
    pm = simple_nms(pm, 4)
    tensor_bool = pm.squeeze() > tenth_largest
    kpts  = torch.nonzero(tensor_bool)
    return kpts

class Pdec(BaseModel):
    def __init__(self, opt):
        super(Pdec, self).__init__(opt)
        # define network and load pretrained models
        self.netPdec = self.set_device(networks.define_Pdec(opt))

        # set loss and load resume state
        self.loss_type = opt['model_pdec']['loss_type']
        if self.loss_type == 'ce':
            self.loss_func = nn.CrossEntropyLoss().to(self.device)
            #self.loss_func = detector_loss
        else:
            raise NotImplementedError()

        if self.opt['phase'] == 'train':
            self.netPdec.train()
            # find the parameters to optimize
            optim_Pdec_params = list(self.netPdec.parameters())

            if opt['train']["optimizer"]["type"] == "adam":
                self.optPdec = torch.optim.Adam(
                    optim_Pdec_params, lr=opt['train']["optimizer"]["lr"])
            elif opt['train']["optimizer"]["type"] == "adamw":
                self.optPdec = torch.optim.AdamW(
                    optim_Pdec_params, lr=opt['train']["optimizer"]["lr"])
            else:
                raise NotImplementedError(
                    'Optimizer [{:s}] not implemented'.format(opt['train']["optimizer"]["type"]))
            self.log_dict = OrderedDict()

            # Define learning rate sheduler
            self.exp_lr_scheduler_netPdec = get_scheduler(optimizer=self.optPdec, args=opt['train'])
        else:
            self.netPdec.eval()
            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()

        self.running_metric = ConfuseMatrixMeter(n_class=opt['model_pdec']['out_channels'])
        self.len_train_dataloader = opt["len_train_dataloader"]
        self.len_val_dataloader = opt["len_val_dataloader"]

    # Feeding all data to the Pdec model
    def feed_data(self, feats, data):
        self.feats= feats
        self.data = self.set_device(data)
    def feed_data2(self, f1, f2, data):
        self.f1 = f1
        self.f2 = f2
    # Optimize the parameters of the Pdec model
    def optimize_parameters(self):
        self.optPdec.zero_grad()
        self.pred_cm = self.netPdec(self.feats)
        l_Pdec = self.loss_func(self.pred_cm, self.data["labels"].long())
        l_Pdec.backward()
        self.optPdec.step()
        self.log_dict['l_Pdec'] = l_Pdec.item()

    # Testing on given data
    def test(self):
        self.netPdec.eval()
        with torch.no_grad():
            if isinstance(self.netPdec, nn.DataParallel):
                self.pred_cm = self.netPdec.module.forward(self.feats)
            else:
                self.pred_cm = self.netPdec(self.feats)
        self.netPdec.train()

    # Get current log
    def get_current_log(self):
        return self.log_dict

    # Get current visuals
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['pred_cm'] = torch.argmax(self.pred_cm, dim=1, keepdim=False)
        out_dict['gt_cm'] = self.data['labels']
        return out_dict
    def get_current_visuals_test(self):
        out_dict = OrderedDict()
        out_dict['pred_cm'] = torch.argmax(self.pred_cm, dim=1, keepdim=False)
        return out_dict

    def get_scores(self,feats):
        self.netPdec.eval()
        with torch.no_grad():
            if isinstance(self.netPdec, nn.DataParallel):
                scores_o = self.netPdec.module.forward(feats)
            else:
                scores_o = self.netPdec(feats)
            scores = scores_o[:, 1, :, :]
        self.netPdec.train()
        return scores
    def get_scores2(self,feats):
        self.netPdec.eval()
        with torch.no_grad():
            if isinstance(self.netPdec, nn.DataParallel):
                scores_o = self.netPdec.module.forward(feats)
            else:
                scores_o = self.netPdec(feats)
            scores = torch.argmax(scores_o, dim=1, keepdim=False)

        self.netPdec.train()
        return scores


    # Printing the Pdec network
    def print_network(self):
        s, n = self.get_network_description(self.netPdec)
        if isinstance(self.netPdec, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netPdec.__class__.__name__,
                                             self.netPdec.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netPdec.__class__.__name__)

        logger.info(
            'Pdec Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    # Saving the network parameters
    def save_network(self, epoch, is_best_model=False):
        Pdec_gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'Pdec_model_E{}_gen.pth'.format(epoch))
        Pdec_opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'Pdec_model_E{}_opt.pth'.format(epoch))

        if is_best_model:
            best_Pdec_gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_Pdec_model_gen.pth'.format(epoch))
            best_Pdec_opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_Pdec_model_opt.pth'.format(epoch))

        # Save Pdec model pareamters
        network = self.netPdec
        if isinstance(self.netPdec, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, Pdec_gen_path)
        if is_best_model:
            torch.save(state_dict, best_Pdec_gen_path)

        # Save Pdec optimizer paramers
        opt_state = {'epoch': epoch,
                     'scheduler': None,
                     'optimizer': None}
        opt_state['optimizer'] = self.optPdec.state_dict()
        torch.save(opt_state, Pdec_opt_path)
        if is_best_model:
            torch.save(opt_state, best_Pdec_opt_path)

        # Print info
        logger.info(
            'Saved current Pdec model in [{:s}] ...'.format(Pdec_gen_path))
        if is_best_model:
            logger.info(
                'Saved best Pdec model in [{:s}] ...'.format(best_Pdec_gen_path))

    # Loading pre-trained Pdec network
    def load_network(self):
        load_path = self.opt['path_Pdec']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for Pdec model [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)

            # Pdec model
            network = self.netPdec
            if isinstance(self.netPdec, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=True)

            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.optPdec.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

    # Functions related to computing performance metrics for Pdec
    def _update_metric(self):
        """
        update metric
        """
        G_pred = self.pred_cm.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=self.data['labels'].detach().cpu().numpy())
        return current_score

    # Collecting status of the current running batch
    def _collect_running_batch_states(self):
        self.running_acc = self._update_metric()
        self.log_dict['running_acc'] = self.running_acc.item()

    # Collect the status of the epoch
    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.log_dict['epoch_acc'] = self.epoch_acc.item()

        for k, v in scores.items():
            self.log_dict[k] = v
            # message += '%s: %.5f ' % (k, v)

    # Rest all the performance metrics
    def _clear_cache(self):
        self.running_metric.clear()

    # Finctions related to learning rate sheduler
    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_netPdec.step()



