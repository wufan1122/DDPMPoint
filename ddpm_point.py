from cProfile import label

import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/points.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    print(args)
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb

        print("Initializing wandblog.")
        wandb_logger = WandbLogger(opt)
        # Training log
        wandb.define_metric('epoch')
        wandb.define_metric('training/train_step')
        wandb.define_metric("training/*", step_metric="train_step")
        # Validation log
        wandb.define_metric('validation/val_step')
        wandb.define_metric("validation/*", step_metric="val_step")
        # Initialization
        train_step = 0
        val_step = 0
    else:
        wandb_logger = None

    # Loading point_detection datasets.
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            print("Creating [train] pdec dataloader.")
            train_set = Data.create_points_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)

        elif phase == 'val' and args.phase != 'test':
            print("Creating [val] pdec dataloader.")
            val_set = Data.create_points_dataset(dataset_opt, phase)
            val_loader = Data.create_points_dataloader(
                val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)

        elif phase == 'test' and args.phase == 'test':
            print("Creating [test] pdec dataloader.")
            print(phase)
            test_set = Data.create_points_dataset(dataset_opt, phase)
            test_loader = Data.create_points_dataloader(
                test_set, dataset_opt, phase)
            opt['len_test_dataloader'] = len(test_loader)

    logger.info('Initial Dataset Finished')

    # Loading diffusion model
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # Set noise schedule for the diffusion model
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    # Creating pdec model
    point_detection = Model.create_pdec_model(opt)

    #################
    # Training loop #
    #################
    begin_step = point_detection.begin_step
    begin_epoch = point_detection.begin_epoch
    print("begin_step:", begin_step, "begin_epoch:", begin_epoch)
    if opt['path_Pdec']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            begin_epoch, begin_step))
    n_epoch = opt['train']['n_epoch']
    best_mF1 = 0.0
    start_epoch = 0
    if opt['phase'] == 'train':
        for current_epoch in range(start_epoch, n_epoch):
            point_detection._clear_cache()
            train_result_path = '{}/train/{}'.format(opt['path']
                                                     ['results'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)

            ################
            ### training ###
            ################
            message = 'lr: %0.7f\n \n' % point_detection.optPdec.param_groups[0]['lr']
            logger.info(message)
            for current_step, train_data in enumerate(train_loader):
                # Feeding data to diffusion model and get features
                diffusion.feed_data(train_data)
                f = []
                for t in opt['model_pdec']['t']:
                    fe, fd= diffusion.get_feats(t=t)  # np.random.randint(low=2, high=8)
                    if opt['model_pdec']['feat_type'] == "dec":
                        f.append(fd)
                        del fe
                    else:
                        f.append(fe)
                        del fd

                # for i in range(0, len(fd_A)):
                #     print(fd_A[i].shape)
                """
                print(len(f))
                for i in range(0, len(f)):
                    for j in range(0, len(f[i])):
                        if i == 0:
                            print(f[i][j].shape)
                """
                # Feeding features from the diffusion model to the pdec model
                point_detection.feed_data(f, train_data)
                point_detection.optimize_parameters()
                point_detection._collect_running_batch_states()

                # log running batch status
                if current_step % opt['train']['train_print_freq'] == 0:
                    # message
                    logs = point_detection.get_current_log()
                    message = '[Training Pdec]. epoch: [%d/%d]. Itter: [%d/%d], Pdec_loss: %.5f, running_mf1: %.5f\n' % \
                              (current_epoch, n_epoch - 1, current_step, len(train_loader), logs['l_Pdec'],
                               logs['running_acc'])
                    logger.info(message)

                    # vissuals
                    visuals = point_detection.get_current_visuals()

                    img_mode = "grid"
                    if img_mode == "single":
                        # Converting to uint8
                        img = Metrics.tensor2img(train_data['img'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        gt_cm = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8,
                                                   min_max=(0, 1))  # uint8
                        pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                                     out_type=np.uint8, min_max=(0, 1))  # uint8

                        # save imgs
                        Metrics.save_img(
                            img, '{}/img_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            pred_cm, '{}/img_pred_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            gt_cm, '{}/img_gt_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                    else:
                        # grid img
                        visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
                        visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                        grid_img = torch.cat((train_data['img'],
                                              visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                              visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                             dim=0)
                        grid_img = Metrics.tensor2img(grid_img)  # uint8
                        Metrics.save_img(
                            grid_img,
                            '{}/img_pred_gt_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))

            ### log epoch status ###
            point_detection._collect_epoch_states()
            logs = point_detection.get_current_log()
            message = '[Training Pdec (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' % \
                      (current_epoch, n_epoch - 1, logs['epoch_acc'])
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
                tb_logger.add_scalar(k, v, current_step)
            message += '\n'
            logger.info(message)

            if wandb_logger:
                wandb_logger.log_metrics({
                    'training/mF1': logs['epoch_acc'],
                    'training/mIoU': logs['miou'],
                    'training/OA': logs['acc'],
                    'training/change-F1': logs['F1_1'],
                    'training/no-change-F1': logs['F1_0'],
                    'training/change-IoU': logs['iou_1'],
                    'training/no-change-IoU': logs['iou_0'],
                    'training/train_step': current_epoch
                })

            point_detection._clear_cache()
            point_detection._update_lr_schedulers()

            ##################
            ### validation ###
            ##################
            if current_epoch % opt['train']['val_freq'] == 0:
                val_result_path = '{}/val/{}'.format(opt['path']
                                                     ['results'], current_epoch)
                os.makedirs(val_result_path, exist_ok=True)

                for current_step, val_data in enumerate(val_loader):
                    # Feed data to diffusion model
                    diffusion.feed_data(val_data)
                    f = []
                    for t in opt['model_pdec']['t']:
                        fe, fd = diffusion.get_feats(t=t)  # np.random.randint(low=2, high=8)
                        if opt['model_pdec']['feat_type'] == "dec":
                            f.append(fd)
                            del fe
                        else:
                            f.append(fe)
                            del fd

                    # Feed data to pdec model
                    point_detection.feed_data(f, val_data)
                    point_detection.test()
                    point_detection._collect_running_batch_states()

                    # log running batch status for val data
                    if current_step % opt['train']['val_print_freq'] == 0:
                        # message
                        logs = point_detection.get_current_log()
                        message = '[Validation pdec]. epoch: [%d/%d]. Itter: [%d/%d], running_mf1: %.5f\n' % \
                                  (current_epoch, n_epoch - 1, current_step, len(val_loader), logs['running_acc'])
                        logger.info(message)

                        # vissuals
                        visuals = point_detection.get_current_visuals()

                        img_mode = "grid"
                        if img_mode == "single":
                            # Converting to uint8
                            img = Metrics.tensor2img(val_data['img'], out_type=np.uint8, min_max=(-1, 1))  # uint8

                            gt_cm = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                                       out_type=np.uint8, min_max=(0, 1))  # uint8
                            pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                                         out_type=np.uint8, min_max=(0, 1))  # uint8

                            # save imgs
                            Metrics.save_img(
                                img, '{}/img_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))

                            Metrics.save_img(
                                pred_cm, '{}/img_pred_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            Metrics.save_img(
                                gt_cm, '{}/img_gt_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        else:
                            # grid img
                            visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
                            visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                            grid_img = torch.cat((val_data['img'],
                                                  visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                                  visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                                 dim=0)
                            grid_img = Metrics.tensor2img(grid_img)  # uint8
                            Metrics.save_img(
                                grid_img,
                                '{}/img_pred_gt_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))

                point_detection._collect_epoch_states()
                logs = point_detection.get_current_log()
                message = '[Validation pdec (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' % \
                          (current_epoch, n_epoch - 1, logs['epoch_acc'])
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                message += '\n'
                logger.info(message)

                if wandb_logger:
                    wandb_logger.log_metrics({
                        'validation/mF1': logs['epoch_acc'],
                        'validation/mIoU': logs['miou'],
                        'validation/OA': logs['acc'],
                        'validation/change-F1': logs['F1_1'],
                        'validation/no-change-F1': logs['F1_0'],
                        'validation/change-IoU': logs['iou_1'],
                        'validation/no-change-IoU': logs['iou_0'],
                        'validation/val_step': current_epoch
                    })

                if logs['epoch_acc'] > best_mF1:
                    is_best_model = True
                    best_mF1 = logs['epoch_acc']
                    logger.info(
                        '[Validation pdec] Best model updated. Saving the models (current + best) and training states.')
                else:
                    is_best_model = False
                    logger.info('[Validation pdec]Saving the current Pdec model and training states.')
                logger.info('--- Proceed To The Next Epoch ----\n \n')

                point_detection.save_network(current_epoch+begin_epoch, is_best_model=is_best_model)
                point_detection._clear_cache()

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch - 1})

        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation (testing).')
        test_result_path = '{}/test/'.format(opt['path']
                                             ['results'])
        os.makedirs(test_result_path, exist_ok=True)
        logger_test = logging.getLogger('test')  # test logger
        point_detection._clear_cache()
        for current_step, test_data in enumerate(test_loader):
            # Feed data to diffusion model
            diffusion.feed_data(test_data)
            f = []
            for t in opt['model_pdec']['t']:
                fe, fd = diffusion.get_feats(t=t)  # np.random.randint(low=2, high=8)
                if opt['model_pdec']['feat_type'] == "dec":
                    f.append(fd)
                    del fe
                else:
                    f.append(fe)
                    del fd

            # Feed data to pdec model
            point_detection.feed_data(f, test_data)
            point_detection.test()
            point_detection._collect_running_batch_states()

            # Logs
            logs = point_detection.get_current_log()
            message = '[Testing pdec]. Itter: [%d/%d], running_mf1: %.5f\n' % \
                      (current_step, len(test_loader), logs['running_acc'])
            logger_test.info(message)

            # Vissuals
            visuals = point_detection.get_current_visuals()
            img_mode = 'single'
            if img_mode == 'single':
                # Converting to uint8
                visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
                visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                img = Metrics.tensor2img(test_data['img'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                gt_cm = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8,
                                           min_max=(0, 1))  # uint8
                pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8,
                                             min_max=(0, 1))  # uint8

                # Save imgs
                Metrics.save_img(
                    img, '{}/img_{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    pred_cm, '{}/img_pred_cm{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    gt_cm, '{}/img_gt_cm{}.png'.format(test_result_path, current_step))
            else:
                # grid img
                visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
                visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                grid_img = torch.cat((test_data['img'],
                                      visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                      visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                     dim=0)
                grid_img = Metrics.tensor2img(grid_img)  # uint8
                Metrics.save_img(
                    grid_img, '{}/img_pred_gt_{}.png'.format(test_result_path, current_step))

        point_detection._collect_epoch_states()
        logs = point_detection.get_current_log()
        message = '[Test pdec summary]: Test mF1=%.5f \n' % \
                  (logs['epoch_acc'])
        for k, v in logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
            message += '\n'
        logger_test.info(message)

        if wandb_logger:
            wandb_logger.log_metrics({
                'test/mF1': logs['epoch_acc'],
                'test/mIoU': logs['miou'],
                'test/OA': logs['acc'],
                'test/change-F1': logs['F1_1'],
                'test/no-change-F1': logs['F1_0'],
                'test/change-IoU': logs['iou_1'],
                'test/no-change-IoU': logs['iou_0'],
            })

        logger.info('End of testing...')
