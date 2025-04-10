import torch
import torch.nn as nn
from pytorch_msssim import ssim


class CombinedMSESSIMLoss(nn.Module):
    def __init__(self, mse_weight, ssim_weight):
        """
        mse_weight: Weight for the MSE loss component.
        ssim_weight: Weight for the SSIM loss component.
        """
        super(CombinedMSESSIMLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight

    def forward(self, output, target):
        
        loss_mse = self.mse_loss(output, target)


        batch_size = output.shape[0]
        output_img = output.view(batch_size, 1, 28, 28)
        target_img = target.view(batch_size, 1, 28, 28)

        
        ssim_val = ssim(output_img, target_img, data_range=target.max() - target.min(), size_average=True)
        
        loss_ssim = 1 - ssim_val

        total_loss = (self.mse_weight * loss_mse) + (self.ssim_weight * loss_ssim)
        return total_loss
