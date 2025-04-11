import torch
import torch.nn as nn
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.filters import sobel
from scipy.stats import pearsonr



class CombinedMSESSIMLoss(nn.Module):
    def __init__(self, mse_weight, ssim_weight, epi_weight, psnr_weight):
        """
        mse_weight: Weight for the MSE loss component.
        ssim_weight: Weight for the SSIM loss component.
        """
        super(CombinedMSESSIMLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.epi_weight = epi_weight
        self.psnr_weight = psnr_weight

    def forward(self, output, target):
        
        loss_mse = self.mse_loss(output, target)


        batch_size = output.shape[0]
        output_img = output.view(batch_size, 1, 28, 28)
        target_img = target.view(batch_size, 1, 28, 28)


        # torch computation
        ssim_val = ssim(output_img, target_img, data_range=target.max() - target.min(), size_average=True) # ssim

        # convert to numpy objects for psnr and epi computation
        output_img = output_img.detach().cpu().numpy()
        target_img = target_img.detach().cpu().numpy()


        psnr_loss = psnr(output_img, target_img, data_range = 1.0) # psnr
        edge_output= sobel(output_img)
        edge_target = sobel(target_img)
        epi_loss, _ = pearsonr(edge_output.flatten(), edge_target.flatten())
        
        loss_ssim = 1 - ssim_val

        total_loss = (self.mse_weight * loss_mse) + (self.ssim_weight * loss_ssim) + (self.epi_weight * epi_loss) + (self.psnr_weight * psnr_loss)
        return total_loss
