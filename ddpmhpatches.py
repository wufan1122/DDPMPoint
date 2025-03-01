from cProfile import label
from fileinput import filename

import torch
import time
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
import cv2
from model.pdec_model import get_current_kpts
from model.des_interpolation import InterpolateSparse2d
from misc import mnn
import numpy as np
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
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test' and args.phase == 'test':
            print("Creating [test] pdec dataloader.")
            print(phase)
            test_set = Data.create_hpatches_dataset(dataset_opt, phase)
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
    logger.info('Initial pdec Model Finished')
    # Creating des model
    des_detection = Model.create_des_model(opt)
    logger.info('Initial des Model Finished')

    logger.info('Begin Feature Points Detection.')
    test_result_path = '{}/test/'.format(opt['path']
                                         ['results'])
    os.makedirs(test_result_path, exist_ok=True)
    interpolator = InterpolateSparse2d('bicubic')
    def compute_repeatability(m_points, n_points, H, img2_width, img2_height, epsilon=5.0):
	    """
	    计算特征点重复率。
	
	    参数:
	    - m_points (Tensor): 第一幅图像的特征点，形状为 (N, 2)，表示 N 个点的 (x, y) 坐标。
	    - n_points (Tensor): 第二幅图像的特征点，形状为 (M, 2)，表示 M 个点的 (x, y) 坐标。
	    - H (Tensor): 变换矩阵，形状为 (3, 3)。
	    - img2_width (int): 第二幅图像的宽度。
	    - img2_height (int): 第二幅图像的高度。
	    - epsilon (float): 距离阈值，用于判断两个点是否为重复点。
	
	    返回:
	    - repeatability (float): 特征点重复率。
	    """
	    
	    # 确保输入是浮点型
	    m_points = m_points.float()
	    n_points = n_points.float()
	    H = H.float()
	    
	    # 将 m_points 扩展为齐次坐标
	    ones = torch.ones(m_points.shape[0], 1, device=m_points.device)
	    m_homogeneous = torch.cat([m_points, ones], dim=1)  # 形状为 (N, 3)
	    
	    # 应用变换矩阵 H
	    m_transformed = torch.matmul(H, m_homogeneous.t()).t()  # 形状为 (N, 3)
	    
	    # 归一化到图像坐标
	    m_transformed[:, 0] /= m_transformed[:, 2]
	    m_transformed[:, 1] /= m_transformed[:, 2]
	    m_transformed = m_transformed[:, :2]
	    
	    # 范围约束筛选
	    mask = (m_transformed[:, 0] > 0) & (m_transformed[:, 0] < img2_width) & \
	            (m_transformed[:, 1] > 0) & (m_transformed[:, 1] < img2_height)
	    m_filtered = m_transformed[mask]  # 形状为 (K, 2)
	    
	    if m_filtered.numel() == 0:
	        return 0.0  # 如果没有匹配点，重复率为0
	    
	    # 计算欧式距离
	    # 扩展 m_filtered 和 n_points 以便广播
	    m_expanded = m_filtered.unsqueeze(1)  # 形状为 (K, 1, 2)
	    n_expanded = n_points.unsqueeze(0)    # 形状为 (1, M, 2)
	    
	    distances = torch.norm(m_expanded - n_expanded, dim=2)  # 形状为 (K, M)
	    
	    # 找到每个 m_filtered 点最近的 n_points 点的距离
	    min_distances, _ = torch.min(distances, dim=1)  # 形状为 (K,)
	    
	    # 计算重复率
	    repeatability = (min_distances <= epsilon).sum().item() / m_filtered.shape[0]
	    
	    return repeatability

	    
    def mean_matching_accuracy(kpts1, H1, H_GT, threshold=5.0):
	    """
	    计算 Mean Matching Accuracy (MMA)
	    
	    参数:
	    - kpts1: 图像 img1 中的特征点，形状为 [N, 2]，其中 N 是特征点数量
	    - H1: 估计的单应性矩阵，形状为 [3, 3]
	    - H_GT: 真实的单应性矩阵，形状为 [3, 3]
	    - threshold: 判断正确匹配的阈值，默认是 5 像素
	    
	    返回:
	    - mma: Mean Matching Accuracy
	    """
	    # 将 kpts1 转换为齐次坐标 [x, y, 1]
	    kpts1 = kpts1.float()
	    H1 = H1.float()
	    H_GT = H_GT.float()
	    N = kpts1.shape[0]
	    ones = torch.ones(N, 1, device=kpts1.device)
	    kpts1_homo = torch.cat([kpts1, ones], dim=1)  # [N, 3]
	    
	    # 使用真实单应性矩阵 H_GT 映射 kpts1 到 img2
	    kpts1_transformed_gt = torch.matmul(H_GT, kpts1_homo.t()).t()  # [N, 3]
	    kpts1_transformed_gt = kpts1_transformed_gt[:, :2] / kpts1_transformed_gt[:, 2:]  # 归一化 [N, 2]
	    
	    # 使用估计的单应性矩阵 H1 映射 kpts1 到 img2
	    kpts1_transformed_pred = torch.matmul(H1, kpts1_homo.t()).t()  # [N, 3]
	    kpts1_transformed_pred = kpts1_transformed_pred[:, :2] / kpts1_transformed_pred[:, 2:]  # 归一化 [N, 2]
	    
	    # 计算预测点与真实点之间的欧几里得距离
	    distances = torch.norm(kpts1_transformed_pred - kpts1_transformed_gt, dim=1)
	    
	    # 判断哪些匹配对的误差小于阈值
	    correct_matches = (distances < threshold).float()
	    
	    # 计算正确匹配的比例
	    mma = torch.mean(correct_matches)
	    
	    return mma.item()

    def homography_estimation_accuracy(H1, H_GT, image_size, corr_thres=None):
	    """
	    计算估计的单应性矩阵 H1 与真实单应性矩阵 H_GT 之间的误差。
	
	    参数:
	    H1 (torch.Tensor): 估计的单应性矩阵，形状为 (3, 3)。
	    H_GT (torch.Tensor): 真实的单应性矩阵，形状为 (3, 3)。
	    image_size (tuple): 图像的尺寸 (height, width)。
	    corr_thres (list, optional): 误差阈值列表，用于计算正确性。如果为 None，则只返回平均误差。
	
	    返回:
	    error (float): 估计的单应性矩阵与真实单应性矩阵之间的平均误差。
	    correctness (list): 在不同阈值下的正确性判断（仅当 corr_thres 不为 None 时返回）。
	    """
	    # 定义图像的四个角点
	    H1 = H1.float()
	    H_GT = H_GT.float()
	    
	    h, w = image_size
	    corners = torch.tensor([[0, 0],
	                            [0, w - 1],
	                            [h - 1, 0],
	                            [h - 1, w - 1]], device=H1.device).float()  # (4, 2)
	
	    # 将点转换为齐次坐标 (x, y, 1)
	    N = corners.shape[0]
	    ones = torch.ones(N, 1, device=corners.device)
	    points_homogeneous = torch.cat([corners, ones], dim=1)  # (4, 3)
	
	    # 使用估计的单应性矩阵 H1 变换点
	    transformed_points_H1 = torch.mm(H1, points_homogeneous.t()).t()  # (4, 3)
	    transformed_points_H1 = transformed_points_H1[:, :2] / transformed_points_H1[:, 2].unsqueeze(1)  # 归一化
	
	    # 使用真实的单应性矩阵 H_GT 变换点
	    transformed_points_H_GT = torch.mm(H_GT, points_homogeneous.t()).t()  # (4, 3)
	    transformed_points_H_GT = transformed_points_H_GT[:, :2] / transformed_points_H_GT[:, 2].unsqueeze(1)  # 归一化
	
	    # 计算误差（欧氏距离）
	    error = torch.norm(transformed_points_H1 - transformed_points_H_GT, dim=1).mean()
	
	    # 如果需要计算正确性
	    if corr_thres is not None:
	        correctness = [float(error <= cthr) for cthr in corr_thres]
	        return error.item(), correctness
	    else:
	        return error.item()
		    

	    
    def adjust_homography(H, original_size, new_size):
	    """
	    调整单应性矩阵以适应新的图像分辨率。
	
	    参数:
	    - H (torch.Tensor): 原始单应性矩阵，形状为 (3, 3)。
	    - original_size (tuple): 原始图像的分辨率 (width, height)。
	    - new_size (tuple): 新图像的分辨率 (width, height)。
	
	    返回:
	    - H_adjusted (torch.Tensor): 调整后的单应性矩阵，形状为 (3, 3)。
	    """
	    W1, H1 = original_size
	    W2, H2 = new_size
	
	    # 计算缩放比例
	    sx = W2 / W1
	    sy = H2 / H1
	
	    # 构造缩放矩阵 S 和其逆矩阵 S_inv
	    S = torch.tensor([[sx, 0, 0],
	                      [0, sy, 0],
	                      [0, 0, 1]], dtype=H.dtype, device=H.device)
	
	    S_inv = torch.tensor([[1/sx, 0, 0],
	                          [0, 1/sy, 0],
	                          [0, 0, 1]], dtype=H.dtype, device=H.device)
	
	    # 调整单应性矩阵
	    H_adjusted = S @ H @ S_inv
	
	    return H_adjusted
    rep_total = [0,0,0,0,0,0,0,0,0]
    mma_total = [0,0,0,0,0,0,0,0,0]
    mha_total = [0,0,0,0,0,0,0,0,0]
    for current_step, test_data in enumerate(tqdm(test_loader)):
        # Feed data to diffusion model
        diffusion.feed_data(test_data)
        for iiii in range(5):
            
            # ***********************************
            #
            #
            # ***********************************
            f1 = []
            f2 = []
            for t in opt['model_pdec']['t']:
                f1e, f1d, f2e, f2d, H1n = diffusion.get_feats3(iiii, t=t)  # np.random.randint(low=2, high=8)
                if opt['model_pdec']['feat_type'] == "dec":
                    f1.append(f1d)
                    f2.append(f2d)

                else:
                    f1.append(f1e)
                    f2.append(f2e)

            # ***********************************
            #
            #
            # ***********************************
            # Feed data to pdec model
            point_detection.feed_data2(f1, f2, test_data)

            # Vissuals
            scores1 = point_detection.get_scores(f1)
            scores2 = point_detection.get_scores(f2)

            des1 = des_detection.get_des(diffusion.imgs[0])
            des1 = Ff.normalize(des1, dim=1)
            des2 = des_detection.get_des(diffusion.imgs[iiii+1])
            des2 = Ff.normalize(des2, dim=1)

            
            
            test_data_subdir_cleaned = test_data["subdir"][0]
            img1_path = '{}/{}/{}/1.png'.format(opt['datasets']['test']['dataroot'], opt['datasets']['test']['folder_type'],test_data_subdir_cleaned)
            img2_path = '{}/{}/{}/{}.png'.format(opt['datasets']['test']['dataroot'], opt['datasets']['test']['folder_type'],test_data_subdir_cleaned,iiii+2)

            kpts1 = get_current_kpts(scores1, 10000).unsqueeze(0)  # torch.Size([563, 2])
            kpts2 = get_current_kpts(scores2, 10000).unsqueeze(0)
            des1 = interpolator(des1, kpts1[..., [1, 0]], H=480, W=640).squeeze(0)
            des2 = interpolator(des2, kpts2[..., [1, 0]], H=480, W=640).squeeze(0)

            kpts1 = kpts1.squeeze(0)
            kpts2 = kpts2.squeeze(0)
            #print(kpts1.shape,des1.shape,H1n.shape) #torch.Size([1682, 2]) torch.Size([1682, 128]) torch.Size([1, 3, 3])
            

            img1 = cv2.imread(img1_path)  # ,480, 640, 3
            img2 = cv2.imread(img2_path)
            w, h, _ = img1.shape


            img1 = cv2.resize(img1, (640, 480), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, (640, 480), interpolation=cv2.INTER_LINEAR)




            matches = mnn.mutual_nearest_neighbor(des1, des2).cpu()


            kpts1_matched = kpts1[matches[:, 0]].cpu().numpy()
            kpts2_matched = kpts2[matches[:, 1]].cpu().numpy()



            H, mask = cv2.findHomography(kpts1_matched[..., [1, 0]], kpts2_matched[..., [1, 0]], cv2.FM_RANSAC)
            good_matches = matches[mask.ravel() == 1]



            # 提取过滤后的匹配点
            kpts1_good = kpts1[good_matches[:, 0]]
            kpts2_good = kpts2[good_matches[:, 1]]

            
            H_adjusted = adjust_homography(H1n.squeeze(0), (h, w), (640, 480))
            H_pre = torch.from_numpy(H).to(kpts2_good.device)
            #print(H_adjusted)
            #print(H_pre)
            rep_p = []
            mma_p = []

            for eps in [1,2,3,4,5,6,7,8,9]:
                #rep
                repeatability = compute_repeatability(kpts1_good[..., [1, 0]], kpts2_good[..., [1, 0]], H_adjusted, 640, 480, eps) 
                rep_p.append(repeatability)
                #MMA
                mma = mean_matching_accuracy(kpts1_good[..., [1, 0]], H_pre, H_adjusted, eps)
                mma_p.append(mma)

            #MHA 
            error, correctness = homography_estimation_accuracy(H_pre, H_adjusted, (480,640), [1, 2, 3, 4, 5, 6, 7, 8, 9])
            print("error:", error)
            print("correctness:", correctness)
            

            

            kpts1_good = kpts1_good.cpu().numpy()
            kpts2_good = kpts2_good.cpu().numpy()

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


            num_matches = len(kpts1_matched)
            colors = generate_colors(num_matches)



            # 绘制匹配点和连接线
            for i, ((y0, x0), (y1, x1)) in enumerate(zip(kpts1_good, kpts2_good)):
                color = (0, 255, 0)
                cv2.line(out, (x0, y0), (x1 + 10 + w, y1), color=color, thickness=1, lineType=cv2.LINE_AA)
                # display line end-points as circles
                cv2.circle(out, (x0, y0), 2, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(out, (x1 + 10 + w, y1), 2, color, -1,
                           lineType=cv2.LINE_AA)



            filename1 = '{}/{}_out_{}.png'.format(test_result_path,test_data_subdir_cleaned,iiii+1)
            rep_total = [x + y for x, y in zip(rep_total, rep_p)]
            mma_total = [x + y for x, y in zip(mma_total, mma_p)]
            mha_total = [x + y for x, y in zip(mha_total, correctness)]

            cv2.imwrite(filename1, out)
    rep_avg = [x / 48 / 5 for x in rep_total]
    mma_avg = [x / 48 / 5 for x in mma_total]
    mhaa  = [x / 48 / 5 for x in mha_total]
    print("rep_avg:", rep_avg, "mma_avg:", mma_avg, "mha:", mhaa)
    logger.info('End of dec...')

