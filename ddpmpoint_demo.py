import data as Data
import model as Model
import data.util as Util
import argparse
import logging
import core.logger as Logger

from tensorboardX import SummaryWriter

from model.pdec_model import get_current_kpts
from model.des_interpolation import InterpolateSparse2d
from misc import mnn

from model.utils import *
from PIL import Image
import torch.nn.functional as Ff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ptest.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='test')
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

    # Loading point_detection datasets.
    
    logger.info('Initial Dataset Finished')
    # Loading diffusion model
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # Set noise schedule for the diffusion model
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    # Creating pdec model
    point_detection = Model.create_pdec_model(opt)
    logger.info('Initial pdec Model Finished')
    # Creating des model
    des_detection = Model.create_des_model(opt)
    logger.info('Initial des Model Finished')

    logger.info('Begin Feature Points Detection.')
    test_result_path = '{}/test/'.format(opt['path']
                                         ['results'])
    os.makedirs(test_result_path, exist_ok=True)

    interpolator = InterpolateSparse2d('bicubic')
    path1 = '{}/1.png'.format(opt['datasets']['test']['dataroot'])
    path2 = '{}/2.png'.format(opt['datasets']['test']['dataroot'])
    img1 = Image.open(path1).convert("RGB")
    img2 = Image.open(path2).convert("RGB")

    img1 = img1.resize((640, 480), Image.Resampling.LANCZOS)
    img1 = Util.transform_augment_cd(img1, 'val', min_max=(-1, 1))
    img2 = img2.resize((640, 480), Image.Resampling.LANCZOS)
    img2 = Util.transform_augment_cd(img2, 'val', min_max=(-1, 1))
    img1 = img1.unsqueeze(0).to(diffusion.device)
    img2 = img2.unsqueeze(0).to(diffusion.device)
    f1 = []
    f2 = []
    for t in opt['model_pdec']['t']:
        f1e, f1d, f2e, f2d = diffusion.get_feats_demo(img1, img2, t=t)  # np.random.randint(low=2, high=8)
        if opt['model_pdec']['feat_type'] == "dec":
            f1.append(f1d)
            f2.append(f2d)

        else:
            f1.append(f1e)
            f2.append(f2e)

    # Feed data to pdec model
    point_detection.feed_data2(f1, f2, f2)

    # Vissuals
    scores1 = point_detection.get_scores(f1)
    scores2 = point_detection.get_scores(f2)
 
    des1 = des_detection.get_des(img1) # torch.Size( [1, 128, 60, 80])
    des1 = Ff.normalize(des1, dim=1)
    des2 = des_detection.get_des(img2)
    des2 = Ff.normalize(des2, dim=1)

    kpt1 = get_current_kpts(scores1,10000).unsqueeze(0)  # torch.Size([1000, 2])
    kpt2 = get_current_kpts(scores2,10000).unsqueeze(0)
    print(kpt1.shape)
    des1 = interpolator(des1, kpt1[..., [1, 0]], H=480, W=640).squeeze(0)
    des2 = interpolator(des2, kpt2[..., [1, 0]], H=480, W=640).squeeze(0)
    kpt1 = kpt1.squeeze(0)
    kpt2 = kpt2.squeeze(0)
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    print(img1.shape)

    img1 = cv2.resize(img1, (640, 480), interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, (640, 480), interpolation=cv2.INTER_LINEAR)

    matches = mnn.mutual_nearest_neighbor(des1, des2).cpu()
    kpts1_matched = kpt1[matches[:, 0]].cpu().numpy()
    kpts2_matched = kpt2[matches[:, 1]].cpu().numpy()

    H, mask = cv2.findHomography(kpts1_matched, kpts2_matched, cv2.FM_RANSAC)

    good_matches = matches[mask.ravel() == 1]

    kpts1_good = kpt1[good_matches[:, 0]].cpu().numpy()
    kpts2_good = kpt2[good_matches[:, 1]].cpu().numpy()


    h, w, c = img1.shape
    out = 255 * np.ones((h, w + 10 + w, c), np.uint8)
    out[:, :w, :] = img1

    # 将 img2 放在背景的右边，留出10个像素的间隔
    out[:, w + 10:, :] = img2


    def generate_colors(num_colors):
        colors = []
        for i in range(num_colors):
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            colors.append(color)
        return colors


    num_matches = len(kpts1_good)
    # print(num_matches)
    colors = generate_colors(num_matches)

    # 绘制匹配点和连接线
    for i, ((y0, x0), (y1, x1)) in enumerate(zip(kpts1_good, kpts2_good)):
        color = (0, 255, 0)
        cv2.line(out, (x0, y0), (x1 + 10 + w, y1), color=color, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + 10 + w, y1), 2, color, -1,
                   lineType=cv2.LINE_AA)

    filename1 = '{}/out.png'.format(test_result_path)
    cv2.imwrite(filename1, out)

    logger.info('End of dec...')


