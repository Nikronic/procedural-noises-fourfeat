import torch
from torch.utils.data import Dataset

import numpy as np

from collections import namedtuple
import pandas as pd

import os, sys
import json
from datetime import datetime


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def get_mgrid(sidelen, domain, flatten=True):
    '''
    Generates a grid of nodes of elements in given ``domain`` range with ``sidelen`` nodes of that dim

    :param sidelen:  a 2D/3D tuple of number of nodes
    :param domain: a tuple of list of ranges of each dim corresponding to sidelen
    :param flatten: whether or not flatten the final grid (-1, 2/3)
    :return:
    '''

    sidelen = np.array(sidelen)
    tensors = []
    for d in range(len(sidelen)):
        tensors.append(torch.linspace(domain[d, 0], domain[d, 1], steps=sidelen[d]))
    tensors = tuple(tensors)
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    if flatten:
        mgrid = mgrid.reshape(-1, len(sidelen))
    return mgrid


class MeshGrid(Dataset):
    def __init__(self, sidelen, domain, flatten=True):
        """
        Generates a mesh grid matrix of equally distant coordinates

        :param sidelen: Grid dimensions (number of nodes along each dimension)
        :param domain: Domain boundry
        :param flatten: whether or not flatten the final grid (-1, 2 or 3)
        :return: Meshgrid of coordinates (elements, 2 or 3)
        """
        super().__init__()
        self.sidelen = sidelen
        self.domain = domain
        self.flatten = flatten

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return get_mgrid(self.sidelen, self.domain, self.flatten)


class SupervisedMeshGrid(Dataset):
    def __init__(self, sidelen, domain, gt_path, flatten=True):
        """
        Generates a mesh grid matrix of equally distant coordinates for a ground truth target with same grid size

        :param sidelen: Grid dimensions (number of nodes along each dimension)
        :param domain: Domain boundry
        :param gt_path: Path to the .npy saved ground truth densities of the same shape
        :param flatten: whether or not flatten the final grid (-1, 2 or 3)
        :return: Meshgrid of coordinates (elements, 2 or 3)
        """
        super().__init__()
        self.sidelen = sidelen
        self.domain = domain
        self.flatten = flatten
        self.gt_path = gt_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        
        # get saved ground truth
        gt_densities = np.load(self.gt_path).astype(np.float32)
        gt_densities = torch.as_tensor(gt_densities)
        gt_densities = gt_densities.permute(1, 0).unsqueeze(0)

        return get_mgrid(self.sidelen, self.domain, self.flatten), -gt_densities


class RandomField(Dataset):
    def __init__(self, latent, std=0.1, mean=0):
        """
        Generates a latent vector distributed from random normal

        :param latent: Latent vector size based on number of elements
        :param std: std of gaussian noise
        :param mean: mean of gaussian noise
        :return: A random tensor with size of latent
        """
        super().__init__()
        self.latent = latent
        self.std = std
        self.mean = mean

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        # latent size with one feature for each element in latent space
        return torch.randn(self.latent, 1) * self.std + self.mean


class NormalLatent(Dataset):
    def __init__(self, latent_size, std=1, mean=0):
        """
        Generates a latent vector distributed from random normal

        :param latent: Latent vector size based
        :param std: std of gaussian noise
        :param mean: mean of gaussian noise
        :return: A random tensor with size of latent
        """
        super().__init__()
        self.latent_size = latent_size
        self.std = std
        self.mean = mean

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return torch.normal(mean=self.mean, std=self.std, size=(self.latent_size, ))


# see issue #20: register_buffer is bugged in pytorch!
def save_weights(model, title, save=False, path=None):
    if path is None:
        path = 'tmp/'
    
    if save:
        d = {
            'scale': model.scale,
            'B': model.B,
            'model_state_dict': model.state_dict()
        }
        torch.save(d, path + title + '.pt')


def load_weights(model, path):
    d = torch.load(path)
    model.load_state_dict(d['model_state_dict'])
    model.B = d['B']
    model.scale = d['scale']
    sys.stderr.write('Weights, scale, and B  loaded.')
    

def save_densities(density, gridDimensions, title, save=False, prediciton=True, path=None):
    if path is None:
        path = 'tmp/'

    if save:
        if prediciton:
            if os.path.isfile(path + title + '_pred.npy'):
                title += str(int(datetime.timestamp(datetime.now())))
            with open(path + title + '_pred.npy', 'wb') as f:
                np.save(f, -density.view(gridDimensions).detach().cpu().numpy()[:, :].T)

        else:
            with open(path + title + '_gt.npy', 'wb') as f:
                np.save(f, -density.reshape(gridDimensions[0], gridDimensions[1]).T)
