

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_embedding_loss


def warp_points(points, homographies, device='cuda'):

    homographies = homographies.float()
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    batch_size = homographies.shape[0]
    points = torch.cat((points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)
    #homographies = homographies.view(batch_size*3,3)
    homographies = homographies.reshape(batch_size * 3, 3)
    warped_points = homographies@points.transpose(0,1)

    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0,:,:] if no_batches else warped_points

class DescriptorLoss(nn.Module):
    def __init__(self):
        super(DescriptorLoss, self).__init__()

        self.margin = 1.0
    def forward(self, p1, p2, H):
        b, c, h, w = p1.shape
        s = 60 / 480
        S = torch.tensor([
                           [s, 0, 0],
                           [0, s, 0],
                           [0, 0, 1]
                       ], dtype=torch.float64)

        S_inv = torch.tensor([
                               [1/s, 0, 0],
                               [0, 1/s, 0],
                               [0, 0, 1]
                       ], dtype=torch.float64)

        S=S.to(p1.device)
        S_inv=S_inv.to(p1.device)
        H_prime = S @ H @ S_inv

        out = self.apply_homography_vectorized2(p1, H_prime)
        des1 = p1[:, :, out[:, 2].long(), out[:, 3].long()].view(b, c, -1)
        des2 = p2[:, :, out[:, 1].long(), out[:, 0].long()].view(b, c, -1)

        losses = 0
        for i in range(b):
            loss, conf = self.dual_softmax_loss(des1[i],des2[i])
            losses += loss
        losses = losses / b
        #loss = self.cosine_similarity_loss(p1v, p2v)
        return losses

    def dual_softmax_loss(self,X, Y, temp=0.2):
        if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
            raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

        dist_mat = (X.t() @ Y) * temp
        conf_matrix12 = F.log_softmax(dist_mat, dim=1)
        conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

        with torch.no_grad():
            conf12 = torch.exp(conf_matrix12).max(dim=-1)[0]
            conf21 = torch.exp(conf_matrix21).max(dim=-1)[0]
            conf = conf12 * conf21

        target = torch.arange(len(X[0]), device=X.device)

        loss = F.nll_loss(conf_matrix12, target) + \
               F.nll_loss(conf_matrix21, target)

        return loss, conf

    def apply_homography_vectorized2(self, p1, H):
        with torch.no_grad():
            b, c, h, w = p1.shape
            # 创建网格坐标
            H_inv = torch.inverse(H)
            grid = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), dim=2).to(p1.device)

            grid = grid.view([-1, 2])
            grid = torch.stack((grid[:, 1], grid[:, 0]), dim=1)
            grid_positive = grid.view(h, w, 2).unsqueeze(0)
            # 广播单应矩阵并应用变换
            grid_new = warp_points(grid, H_inv, p1.device)
            grid_new = torch.stack((grid_new[:, :, 1], grid_new[:, :, 0]), dim=2)#.squeeze(0)
            grid_new = grid_new.view(-1, h, w, 2)
            grid_new = grid_new.to(torch.int).float()


            out = torch.cat((grid_positive, grid_new), dim=3)
            out = out[(out[:, :, :, 2] >= 0) & (out[:, :, :, 2] <= 59) & (out[:, :, :, 3] >= 0) & (out[:, :, :, 3] <= 59)]
        return out
