import torch
from torch.utils.data import Dataset

import noises

import numpy as np
from collections import namedtuple

import os, sys
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


class PerlinMeshGrid(Dataset):
    def __init__(self, sidelen, domain, rho, n_octaves, stochasticity='stochastic', ratio=0.5, mode='octave_simplex', flatten=True):
        """
        Generates a mesh grid matrix of equally distant coordinates for a ground truth target with same grid size which is
        a Perlin noise where supports multi-octave and specified by its algorith 

        :param sidelen: Grid dimensions (number of nodes along each dimension)
        :param domain: Domain boundry
        :param rho: See ``noises.OctavePerlin`` class
        :param n_octaves: See ``noises.OctavePerlin`` class
        :param stochastic: Wether or not create a new for sampling everytime ``__getitem__`` is being called
        :param ratio: Mask ratio (stochastic - may not create the exact number of zero/ones in mask)
        :param mode: Mode of Perlin noise: 1. ``octave_simplex``
        :param flatten: whether or not flatten the final grid (-1, 2 or 3)
        :return: Meshgrid of coordinates (elements, 2 or 3)
        """
        super().__init__()
        self.coor_sidelen = sidelen[1]
        self.octave_sidelen = sidelen[0]
        self.domain = domain
        self.flatten = flatten
        self.mode = mode
        self.rho = rho
        self.n_octave = n_octaves
        self.stochasticity = stochasticity
        self.ratio = ratio
        self.input, self.output = self.populate_data()
        self.mask = self.compute_new_mask(ratio=self.ratio)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
    
        if self.stochasticity == 'stochastic':
            # uses a new mask for every call
            self.mask = self.compute_new_mask(ratio=self.ratio)
            return self.sample()
        elif self.stochasticity == 'deterministic_randinit':
            # uses the same mask all the time
            return self.sample()
        elif self.stochasticity == 'deterministic_gridinit':
            # uses a subgrid of original grid all the time
            step =  int(1 / self.ratio)
            output = self.output[::step, ::step]
            input_coords = get_mgrid(list(output.shape), self.domain, self.flatten) 
            return input_coords, output 
        else:
            raise ValueError('Stochasticity {} is not defined or implemented'.format(self.stochasticity))

    def sample(self):
        mask = self.mask

        output = self.output.flatten()
        output = output[mask]

        input_coords = self.input
        input_coords = input_coords[mask]
        
        return input_coords, output

    def populate_data(self):
        if self.mode == 'octave_simplex':
            octave_perlin = noises.OctavePerlin(height=self.octave_sidelen[0], width=self.octave_sidelen[1],
                                                rho=self.rho, n_octaves=self.n_octave, device=None)
            output_noise = octave_perlin()
        input_coords = get_mgrid(self.coor_sidelen, self.domain, self.flatten)
        return input_coords, output_noise
    
    def compute_new_mask(self, ratio=0.5):
        mask = torch.rand_like(self.input[..., 0]) > (1 - ratio)
        # with this mask, you may not get a grid shaped output, so we use flattened output to be masked
        ## this enables us to use any ratio for our mask without considering a **square** output grid
        return mask


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
