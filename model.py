import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class inversionLayer(nn.Module):
    def __init__(self, args, psf):
        super().__init__()
        self.args = args
        self.device = args.device
        reg_param = torch.nn.Parameter(args.reg_param_init*torch.ones(1)).to(self.device)
        psf = torch.fft.fftshift(psf, dim = (2,3)).to(self.device)
        psf_fft = torch.fft.fft2(psf)
        wiener_filter = torch.conj(psf_fft)/(torch.abs(psf_fft)**2 + reg_param**2)
        self.wiener_filter = nn.Parameter(wiener_filter, requires_grad = True)

    def forward(self, meas):

        meas = torch.fft.fftshift(meas, dim = (2,3))
        meas_fft = torch.fft.fft2(meas)
        deconvolved = torch.fft.ifft2(self.wiener_filter*meas_fft)
        deconvolved = torch.fft.fftshift(torch.abs(deconvolved), dim = (2,3))
        # Center Crop
        shp = deconvolved.size()
        deconvolved =  deconvolved[:,:,int(shp[2]/2-160):int(shp[2]/2+160),int(shp[3]/2-160):int(shp[3]/2+160)]
        # Min Max Scaling
        for i in range(meas.size()[0]):
            deconvolved[i] = (deconvolved[i].clone() - torch.min(deconvolved[i].clone()))/(torch.max(deconvolved[i].clone())-torch.min(deconvolved[i].clone()))

        return deconvolved