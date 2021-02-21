import torch
import torch.nn.functional as functional
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import os, sys

import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from networks import MLP
import visualizations
import utils

n_octaves = 3
rho = 2
sidelen = np.array([7, 7]).reshape(1, -1)
sidelen = np.concatenate([sidelen, ((sidelen*2)*(2**(n_octaves-1)))], axis=0)
domain = np.array([[0., 1.],[0., 1.]])

dataset = utils.PerlinMeshGrid(sidelen=sidelen, domain=domain, rho=rho, n_octaves=n_octaves, mode='octave_simplex')
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
model_input, ground_truth = next(iter(dataloader))
