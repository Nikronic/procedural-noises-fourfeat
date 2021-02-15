import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd.profiler as profiler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks import MLP
from utils import MeshGrid
import utils
import multires_utils
import visualizations

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from tqdm import tqdm
import json
import argparse
from datetime import datetime
import itertools
import copy


parser=argparse.ArgumentParser()
parser.add_argument('--expp', help='Path to the pretrained weights (result will be saved in "images" with same path)', 
                    default='fourfeat_cl/3921813.pt')
args=parser.parse_args()
experiment_path = args.expp
sys.stderr.write('Loading pretrained weight with experiment ID and path: {}\n'.format(experiment_path))

# PyTorch related global variables
torch.autograd.set_detect_anomaly(False)

# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_model = False
save_density = False
interpolation = False
fixed_scale = True

problem_path = 'problems/2d/mbb_beam.json'  # TODO
test_resolution = [1500, 500]  # TODO
max_resolution = [45, 15]  # TODO
interpolate = False  # TODO

mrconfprint = 'Testing pretrained model in: {} \n'.format(max_resolution)
mrconfprint += 'Interpolation: {}'.format(interpolate)
sys.stderr.write(mrconfprint)

# hyper parameter of positional encoding in NeRF
embedding_size = 256

# deep learning modules
nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=embedding_size,
                 scale=1., hidden_act=nn.ReLU(), output_act=None)
model = nerf_model

# load pretrained weights
weights_path = 'logs/weights/{}'.format(experiment_path)
images_path = weights_path.replace('weights', 'images')[:-3]+'/'
utils.load_weights(model, weights_path)
if torch.cuda.is_available():
    model.cuda()

mrconfprint = 'Testing pretrained model in: {} \n'.format(max_resolution)
mrconfprint += 'Interpolation: {}'.format(interpolate)
sys.stderr.write(mrconfprint)

# hyperparameters of the problem 
with open(problem_path, 'r') as j:
     configs = json.loads(j.read())

problem_name = configs['problem_name']
MATERIAL_PATH = configs['MATERIAL_PATH']
BC_PATH = configs['BC_PATH']
orderFEM = configs['orderFEM']
domainCorners = configs['domainCorners']
gridDimensions = configs['gridDimensions']
E0 = configs['E0']
Emin = configs['Emin']
SIMPExponent = configs['SIMPExponent']
maxVolume = torch.tensor(configs['maxVolume'])
seed = configs['seed']
sys.stderr.write('VoxelFEM problem configs: {}\n'.format(configs))

if max_resolution is None:
    max_resolution = gridDimensions

# reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# query for max resolution
domain = np.array([[0., 1.],[0., 1.]])
dataset = utils.MeshGrid(sidelen=max_resolution, domain=domain, flatten=False)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
model_input = next(iter(dataloader))
if torch.cuda.is_available():
    model_input = model_input.cuda()
    maxVolume = maxVolume.cuda()

with torch.no_grad():
    model.eval()
    density = model(model_input)
    density = density.permute(0, 3, 1, 2)
    density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                            mode='constrained_sigmoid')

# now query for test resolution
if test_resolution is None:
    test_resolution = max_resolution
if interpolate:    
    density = torch.nn.functional.interpolate(density, size=tuple(test_resolution), mode='bilinear')

# topopt (via VoxelFEM-Optimization-Problem)
constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
filters = []  # type: ignore
uniformDensity = maxVolume
tps = initializeTensorProductSimulator(
    orderFEM, domainCorners, test_resolution, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH
)                                                                                  
objective = pyVoxelFEM.ComplianceObjective(tps)                                    
top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, filters)
if torch.cuda.is_available():
    maxVolume = maxVolume.cuda()
voxelfem_engine = fem.VoxelFEMFunction.apply
density = density.cpu()
compliance_loss = voxelfem_engine(density.flatten(), top)
binary_compliance_loss = utils.compute_binary_compliance_loss(density=density, top=top,
                                                              loss_engine=voxelfem_engine)

# visualization
grid_title = ''.join(str(i)+'x' for i in test_resolution)[:-1]
maxVolume_np = maxVolume.detach().cpu().numpy()
expp_title = args.expp[args.expp.rfind('/')+1:-3]
title = 'testPretrained-{}_s{}_{}_Vol{}_intpol-{}_'.format(expp_title, model.scale, grid_title, maxVolume_np, interpolate)
visualizations.density_vis(density, compliance_loss, test_resolution, title, True, visualize, True,
                            binary_loss=binary_compliance_loss, path=images_path)
sys.stderr.write('Test image saved to: {}\n'.format(images_path))

# utils.save_densities(density, gridDimensions, title, save_density, True, path='logs/densities/fourfeat_multires/')

