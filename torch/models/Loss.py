import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunctions():
    def __init__(self, alpha=0.7):
        super(LossFunctions, self).__init__()
        self.alpha = alpha

    def correlation_coefficient(self, x, y):
        mask_value = -0.
        mask_x = (x != mask_value).type(torch.float32)
        mask_y = (y != mask_value).type(torch.float32)
        mask = mask_x * mask_y
        x_masked = x * mask
        y_masked = y * mask

        x_masked = (x_masked - (torch.sum(x_masked) / torch.sum(mask))) * mask
        y_masked = (y_masked - (torch.sum(y_masked) / torch.sum(mask))) * mask

        r_num = torch.sum(x_masked * y_masked)
        r_den = torch.sqrt(torch.sum(torch.square(x_masked))
                           * torch.sum(torch.square(y_masked)))

        r = 0.
        r = torch.where(r_den > 0, r_num / r_den, r + 0)
        return r

    def cosine_sim(self, x, y):
        mask_value = 0.
        mask_x = (x != mask_value).type(torch.float32)
        mask_y = (y != mask_value).type(torch.float32)
        mask = mask_x * mask_y
        x_masked = x * mask
        y_masked = y * mask

        x_norm = F.normalize(x_masked, p=2, dim=0) * mask
        y_norm = F.normalize(y_masked, p=2, dim=0) * mask

        cos_sim = torch.sum(torch.mul(x_norm, y_norm))
        return cos_sim

    def trait_sim_loss_function(self, real, pred):
        mask_value = -1
        mask = (real != mask_value).type(torch.float32)

        real_trans = (real * mask).t()
        pred_trans = (pred * mask).t()

        sim_loss = 0.0
        loss = 0.0
        cnt = 0.0
        trait_num = 9
        # print("trait num: ", trait_num)

        for i in range(1, trait_num):
            for j in range(i + 1, trait_num):
                corr = self.correlation_coefficient(
                    real_trans[i], real_trans[j])
                sim_loss = torch.where(
                    corr >= 0.7, sim_loss +
                    (1 -
                     self.cosine_sim(pred_trans[i], pred_trans[j])), sim_loss
                )
                cnt = torch.where(corr >= 0.7, cnt + 1, cnt)

        loss = torch.where(cnt > 0, sim_loss / cnt, loss)
        return loss

    def mse_loss_function(self, real, pred):
        mask_value = -1
        mask = (real != mask_value).type(torch.float32)
        mse = nn.MSELoss()
        return mse(real * mask, real * pred)

    def loss_function(self, real, pred):
        mse_loss = self.mse_loss_function(real, pred).float()
        ts_loss = self.trait_sim_loss_function(real, pred).float()
        loss = self.alpha * mse_loss + (1 - self.alpha) * ts_loss
        return loss.float()
