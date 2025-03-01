import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter


from model.utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/des.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
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

    # Loading des_detection datasets.
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            print("Creating [train] des dataloader.")
            train_set = Data.create_des_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)

        elif phase == 'val' and args.phase != 'test':
            print("Creating [val] des dataloader.")
            val_set = Data.create_des_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)

        elif phase == 'test' and args.phase == 'test':
            print("Creating [test] des dataloader.")
            print(phase)
            test_set = Data.create_des_dataset(dataset_opt, phase)
            test_loader = Data.create_des_dataloader(
                test_set, dataset_opt, phase)
            opt['len_test_dataloader'] = len(test_loader)

    logger.info('Initial Dataset Finished')

    # Creating des model
    des_detection = Model.create_des_model(opt)

    #################
    # Training loop #
    #################
    begin_step = des_detection.begin_step
    begin_epoch = des_detection.begin_epoch
    print("begin_step:", begin_step, "begin_epoch:", begin_epoch)
    if opt['path_des']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            begin_epoch, begin_step))
    n_epoch = opt['train']['n_epoch']
    best_epoch_l_des = 0.0
    start_epoch = 0
    if opt['phase'] == 'train':
        for current_epoch in range(start_epoch, n_epoch):
            des_detection._clear_cache()
            train_result_path = '{}/train/{}'.format(opt['path']
                                                     ['results'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)

            ################
            ### training ###
            ################
            message = 'lr: %0.7f\n \n' % des_detection.optdes.param_groups[0]['lr']
            logger.info(message)
            for current_step, train_data in enumerate(train_loader):
                img1 = train_data["img1"].to(des_detection.device)
                with torch.no_grad():
                    H, H2 = sample_homography_np(np.array([img1.shape[3], img1.shape[2]]), np.array([img1.shape[3], img1.shape[2]]))
                    totensor = torchvision.transforms.ToTensor()
                    H = totensor(H).expand(img1.shape[0], -1, -1).to(img1.device)
                    img2, _ = apply_homography_vectorized(img1, H)
                # Feeding features to the des model
                des_detection.feed_data(img1, img2, H, train_data)
                des_detection.optimize_parameters()
                des_detection._collect_running_batch_states()

                # log running batch status
                if current_step % opt['train']['train_print_freq'] == 0:
                    # message
                    logs = des_detection.get_current_log()
                    message = '[Training des]. epoch: [%d/%d]. Itter: [%d/%d], des_loss: %.5f\n' % \
                              (current_epoch+begin_epoch, n_epoch - 1, current_step, len(train_loader), logs['l_des'])
                    logger.info(message)
                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'train/loss': logs['l_des']
                        })
            ### log epoch status ###
            des_detection._collect_epoch_states()
            logs = des_detection.get_current_log()
            message = '[Training des (epoch summary)]: epoch: [%d/%d]. epoch_l_des=%.5f \n' % \
                      (current_epoch+begin_epoch, n_epoch - 1, logs['epoch_l_des'])
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
                tb_logger.add_scalar(k, v, current_step)
            message += '\n'
            logger.info(message)
            if wandb_logger:
                wandb_logger.log_metrics({
                    'train/epoch_loss': logs['epoch_l_des']
                })


            des_detection._clear_cache()
            des_detection._update_lr_schedulers()

            ##################
            ### validation ###
            ##################
            if current_epoch % opt['train']['val_freq'] == 0:
                val_result_path = '{}/val/{}'.format(opt['path']
                                                     ['results'], current_epoch)
                os.makedirs(val_result_path, exist_ok=True)

                for current_step, val_data in enumerate(val_loader):
                    img1 = val_data["img1"].to(des_detection.device)
                    with torch.no_grad():
                        H, H2 = sample_homography_np(np.array([img1.shape[3], img1.shape[2]]), np.array([img1.shape[3], img1.shape[2]]))
                        totensor = torchvision.transforms.ToTensor()
                        H = totensor(H).expand(img1.shape[0], -1, -1).to(img1.device)
                        img2, _ = apply_homography_vectorized(img1, H)
                    # Feeding features to the des model
                    des_detection.feed_data(img1, img2, H, val_data)
                    des_detection.test()
                    des_detection._collect_running_batch_states()

                    # log running batch status for val data
                    if current_step % opt['train']['val_print_freq'] == 0:
                        # message
                        logs = des_detection.get_current_log()
                        message = '[Validation des]. epoch: [%d/%d]. Itter: [%d/%d], des_loss: %.5f\n' % \
                                  (current_epoch+begin_epoch, n_epoch - 1, current_step, len(val_loader), logs['l_des'])
                        logger.info(message)
                        if wandb_logger:
                            wandb_logger.log_metrics({
                                'validation/loss': logs['l_des']
                            })
                des_detection._collect_epoch_states()
                logs = des_detection.get_current_log()
                message = '[Validation des (epoch summary)]: epoch: [%d/%d]. epoch_l_des=%.5f \n' % \
                          (current_epoch+begin_epoch, n_epoch - 1, logs['epoch_l_des'])
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                message += '\n'
                logger.info(message)

                if wandb_logger:
                    wandb_logger.log_metrics({
                        'validation/epoch_loss': logs['epoch_l_des']
                    })

                if logs['epoch_l_des'] < best_epoch_l_des:
                    is_best_model = True
                    best_l_des = logs['epoch_l_des']
                    logger.info(
                        '[Validation des] Best model updated. Saving the models (current + best) and training states.')
                else:
                    is_best_model = False
                    logger.info('[Validation des]Saving the current des model and training states.')
                logger.info('--- Proceed To The Next Epoch ----\n \n')

                des_detection.save_network(current_epoch + begin_epoch, is_best_model=is_best_model)
                des_detection._clear_cache()

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch - 1})

        logger.info('End of training.')