import os, sys
import torch
import torch.nn as nn

import kornia
import numpy as np


class ProjectionFilter(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = torch.tensor([float(beta)])
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()

    def forward(self, x):
        return 0.5 * (torch.tanh(0.5 * self.beta) + torch.tanh(self.beta * (x - 0.5))) / torch.tanh(0.5 * self.beta)
    
    def update_params(self, scaler):
        self.beta = self.beta * scaler
    
    def reset_params(self, beta=1):
        self.beta = torch.tensor([float(beta)])
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()


class SmoothingFilter(nn.Module):
    def __init__(self, radius=1):
        super().__init__()
        self.radius = radius

    def forward(self, x):
        return kornia.box_blur(input=x.unsqueeze(0), kernel_size=(self.radius*2+1, self.radius*2+1), 
                               border_type='reflect', normalized=True).squeeze(0)

    def update_params(self, scaler):
        self.radius = self.radius * scaler
    
    def reset_params(self, radius=1):
        self.radius = radius


class GaussianSmoothingFilter(nn.Module):
    def __init__(self, sigma=1):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = self.compute_kernel_size(sigma)
        if sigma == 1:
            self.kernel_size = 5

    def forward(self, x):
        density = kornia.gaussian_blur2d(input=x.unsqueeze(0), kernel_size=(self.kernel_size, self.kernel_size),
                                         sigma=(self.sigma, self.sigma), border_type='reflect').squeeze(0)
        return density

    def update_params(self, scaler):
        self.sigma = self.sigma * scaler
        self.kernel_size = self.compute_kernel_size(self.sigma)
    
    def reset_params(self, sigma=1):
        self.sigma = sigma
        self.kernel_size = self.compute_kernel_size(sigma)
        
    def compute_kernel_size(self, sigma):
        kernel_size = np.floor(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size - 1
        return int(kernel_size)


def apply_filters_group(x, filters, configs):
    for filt, c in zip(filters, configs[1::2]):
        if c >= 0:
            x = filt(x)
    return x


def update_adaptive_filtering(iteration, filters, configs):
    beta_interval, beta_scaler, radius_interval, radius_scaler, gauss_interval, sigma_scaler = configs
    if (iteration % beta_interval) == 0 and (iteration != 0):
        for filt in filters:
            if isinstance(filt, ProjectionFilter):
                filt.update_params(scaler=beta_scaler)
                if beta_scaler != 1:
                    sys.stderr.write(" Update -> Projection Filter       (beta={:0.2f})\n".format(filt.beta.item()))

    if (iteration % radius_interval) == 0 and (iteration != 0):
        for filt in filters:
            if isinstance(filt, SmoothingFilter):
                filt.update_params(scaler=radius_scaler)
                if radius_scaler != 1:
                    sys.stderr.write(" Update -> Smoothing Filter       (radius={:0.2f})\n".format(filt.radius))
                    
    if (iteration % gauss_interval) == 0 and (iteration != 0):
        for filt in filters:
            if isinstance(filt, GaussianSmoothingFilter):
                filt.update_params(scaler=sigma_scaler)
                if sigma_scaler != 1:
                    sys.stderr.write(" Update -> Gaussian Smoothing Filter       (sigma={:0.2f})\n".format(filt.sigma))
