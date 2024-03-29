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
from math import pi

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from tqdm import tqdm
import json, time
import argparse
from datetime import datetime
import itertools
import copy


parser=argparse.ArgumentParser()
parser.add_argument('--jid', help='Slurm job id to name experiment dirs with it')
args=parser.parse_args()
experiment_id = args.jid
sys.stderr.write('Experiment ID: {}\n'.format(experiment_id))

# PyTorch related global variables
torch.autograd.set_detect_anomaly(False)

# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_model = True
save_density = False
fixed_scale = True

# record runtime
start_time = time.perf_counter()

problem_path = 'problems/2d/mbb_beam.json'  # TODO
max_resolution = [495, 165]  # TODO

# multires hyperparameters 
weight_decay = 0  # TODO
use_scheduler = False
forgetting_weights = 'orthogonal'  # TODO
forgetting_activations = None  # TODO
rate = 0.7  # TODO
res_order = 'ftc'
repeat_res = 5  # TODO
epoch_mode = 'constant'
resolutions = multires_utils.prepare_resolutions(interval=2, start=0, end=10, order=res_order, repeat_res=repeat_res)  # TODO
epoch_sizes = multires_utils.prepare_epoch_sizes(n_resolutions=len(resolutions), # TODO
                                                 start=800, end=1500, 
                                                 mode=epoch_mode, constant_value=1500)
mrconfprint = 'resolution order: {}, epoch mode: {}, '.format(res_order, epoch_mode)
mrconfprint += 'forgetting_weights: {}, forgetting_activations: {}, rate: {}, '.format(forgetting_weights,
                                                                                       forgetting_activations,
                                                                                       rate)
mrconfprint += 'repeat resolutions: {} times, '.format(repeat_res)
sys.stderr.write(mrconfprint)

# create experiments folders for each run  
log_base_path = 'logs/'
log_image_path = '{}images/fourfeat_cl/'.format(log_base_path)
log_loss_path =  '{}loss/fourfeat_cl/'.format(log_base_path)
log_weights_path =  '{}weights/fourfeat_cl/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, None, 
                                                experiment_id=args.jid)
log_image_path = '{}images/fourfeat_cl/{}'.format(log_base_path, append_path)
log_loss_path =  '{}loss/fourfeat_cl/{}'.format(log_base_path, append_path)
sys.stderr.write('image path: {}, loss path: {}\n'.format(log_image_path, log_loss_path))

# hyper parameter of positional encoding in NeRF
if not fixed_scale:
    interval_scale = 0.5
    scale = np.arange(60) * interval_scale + interval_scale
else:
    scale = [46.0]  # TODO

embedding_size = 256
sys.stderr.write('scale: {}, fourier embedding size: {}\n'.format(scale, embedding_size))

with open(problem_path, 'r') as j:
     configs = json.loads(j.read())

# hyperparameters of the problem 
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
adaptive_filtering = configs['adaptive_filtering']
seed = configs['seed']
sys.stderr.write('VoxelFEM problem configs: {}\n'.format(configs))

gridDimensions_ = copy.deepcopy(gridDimensions)

if max_resolution is None:
    max_resolution = gridDimensions

# reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    maxVolume = maxVolume.cuda()

domain = np.array([[0., 1.],[0., 1.]])

# deep learning modules
nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=embedding_size,
                 scale=scale[0], hidden_act=nn.ReLU(), output_act=None)              
model = nerf_model
sys.stderr.write('Deep learning model config: {}\n'.format(model))

if torch.cuda.is_available():
    model.cuda()

if weight_decay > 0:
    learning_rate = 1e-3
else:
    learning_rate = 1e-4

optim = torch.optim.Adam(lr=learning_rate, params=itertools.chain(list(model.parameters())), weight_decay=weight_decay)
# reduce on plateau
scheduler = None
if use_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='min', patience=20)
sys.stderr.write('DL optim: {}, LR scheduler: {}\n'.format(optim, scheduler))
sys.stderr.write('L2 Regularization: {}\n'.format(weight_decay))

# training
batch_size = 1
actual_steps = 0  # overall number of iterations over all resolutions
compliance_loss_array = []

for idx, res in enumerate(resolutions):
    for s in scale:

        model.train()  # in case of getting overriden by disabling activation forgetting

        gridDimensions = tuple(np.array(gridDimensions_) + res * np.array(domainCorners[1]))
        sys.stderr.write('New resolution within multires loop: {}\n'.format(gridDimensions))

        if torch.cuda.is_available():
            maxVolume = maxVolume.cuda()

        # deep learning modules
        dataset = MeshGrid(sidelen=gridDimensions, domain=domain, flatten=False)
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
        model_input = next(iter(dataloader))

        # we dont want to update input in nerf so dont enable grads here
        if torch.cuda.is_available():
            model_input = model_input.cuda()

        # topopt (via VoxelFEM-Optimization-Problem)
        constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
        filters = []  # type: ignore
        uniformDensity = maxVolume
        tps = initializeTensorProductSimulator(
            orderFEM, domainCorners, gridDimensions, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH
        )                                                                                  
        objective = pyVoxelFEM.ComplianceObjective(tps)                                    
        top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, filters) 

        # instantiate autograd.Function for VoxelFEM engine
        voxelfem_engine = VoxelFEMFunction.apply

        # save loss values for plotting
        compliance_loss_array_res = []

        # apply 'fogetting' for weights
        if forgetting_weights is not None:
            multires_utils.forget_weights(model=model, rate=rate, mode=forgetting_weights, n_neurons=256, embedding_size=embedding_size)
            sys.stderr.write('Weight forgetting has been applied. \n')
        
        # apply 'forgetting' for activations
        if forgetting_activations is not None:
            model.register_gated_activations(model_input, rate=rate)
            sys.stderr.write('Activation forgetting has been applied. \n')

        # training of xPhys
        for step in tqdm(range(epoch_sizes[idx]), desc='Training: '):

            def closure():
                optim.zero_grad()
                # aka x
                density = model(model_input)
                density = density.view(gridDimensions)
                # aka xPhys
                density = physical_density(density, maxVolume)
                # gaussian smoothing
                density = kornia.gaussian_blur2d(density.unsqueeze(0).unsqueeze(0),
                                                 (5, 5), (1., 1.), border_type='reflect').squeeze(0)
                # compliance for predicted xPhys
                if torch.cuda.is_available():
                    density = density.cpu()
                compliance_loss = voxelfem_engine(density.flatten(), top)
                if torch.cuda.is_available():
                    compliance_loss.cuda()

                global actual_steps
                actual_steps += 1

                # reduce LR if no reach plateau
                if use_scheduler:
                    scheduler.step(compliance_loss)

                compliance_loss.backward()

                # save loss values for plotting
                compliance_loss_array_res.append(compliance_loss.detach().item())
                sys.stderr.write("Total Steps: %d, Resolution Steps: %d, Compliance loss %0.6f" % (actual_steps,
                                                                                                   step, compliance_loss))
                return compliance_loss

            optim.step(closure)

        compliance_loss_array.extend(compliance_loss_array_res)

        # test model with FORGETTING ENABLED
        density = model(model_input)
        density = physical_density(density, maxVolume)

        # loss of conversion to binary by thresholding
        binary_compliance_loss = utils.compute_binary_compliance_loss(density=density, top=top,
                                                                      loss_engine=voxelfem_engine)

        # visualization and saving model
        grid_title = ''.join(str(i)+'x' for i in gridDimensions)[:-1]
        model_title = model.__class__.__name__
        optim_title = optim.__class__.__name__
        maxVolume_np = maxVolume.detach().cpu().numpy()
        if forgetting_weights is not None:
            forgetting_title = 'forget_W{}'.format(forgetting_weights)
        if forgetting_activations is not None:
            forgetting_title = 'forget_A{}'.format(forgetting_activations)

        title = 'fourFeat(wFgt)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(idx+1)+'x'+str(actual_steps)+'_'+model_title+'_'+optim_title+'_dec'+str(weight_decay)+'_'+problem_name+'_Vol'+str(maxVolume_np)+'_gauss5x1_'
        title = visualizations.loss_vis(compliance_loss_array_res, title, True, path=log_loss_path)
        visualizations.density_vis(density, compliance_loss_array_res[-1], gridDimensions, title, True, visualize, True,
                                   binary_loss=binary_compliance_loss, path=log_image_path)

        # test model with activation FORGETTING DISABLED
        if forgetting_activations is not None:
            model.eval()  
        
        # test model with weight FORGETTING DISABLED (INVALID operation! omitted)
        if forgetting_weights is not None:
            sys.stderr.write('Disabling forgetting and density plot have been omitted (does not make sense) \n')
            sys.stderr.write('Instead, we will plot densities queried for max resolution.\n')

        density = model(model_input)
        density = physical_density(density, maxVolume)

        # loss of conversion to binary by thresholding
        binary_compliance_loss = utils.compute_binary_compliance_loss(density=density, top=top,
                                                                      loss_engine=voxelfem_engine)

        if forgetting_activations is not None:
            title = 'fourFeat(woFgt)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(idx+1)+'x'+str(actual_steps)+'_'+model_title+'_'+optim_title+'_dec'+str(weight_decay)+'_'+problem_name+'_Vol'+str(maxVolume_np)+'_gauss5x1_'
            title = visualizations.loss_vis(compliance_loss_array_res, title, True, path=log_loss_path)
            visualizations.density_vis(density, compliance_loss_array_res[-1], gridDimensions, title, True, visualize, True,
                                       binary_loss=binary_compliance_loss, path=log_image_path)

        if forgetting_weights is not None:
            # now query for max resolution
            test_resolution = max_resolution
            dataset = MeshGrid(sidelen=test_resolution, domain=domain, flatten=False)
            dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
            model_input = next(iter(dataloader))
            if torch.cuda.is_available():
                model_input = model_input.cuda()

            # topopt (via VoxelFEM-Optimization-Problem)
            constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
            filters = []
            uniformDensity = maxVolume
            tps = initializeTensorProductSimulator(
                orderFEM, domainCorners, test_resolution, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH
            )                                                                                  
            objective = pyVoxelFEM.ComplianceObjective(tps)                                    
            top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, filters) 
            
            if torch.cuda.is_available():
                maxVolume = maxVolume.cuda()
            voxelfem_engine = VoxelFEMFunction.apply

            with torch.no_grad():
                model.eval()
                density = model(model_input)
                density = physical_density(density, maxVolume)

            # loss of conversion to binary by thresholding
            binary_compliance_loss = utils.compute_binary_compliance_loss(density=density, top=top,
                                                                          loss_engine=voxelfem_engine)
            if torch.cuda.is_available():
                density = density.cpu()
            compliance_loss = voxelfem_engine(density.flatten(), top)

            title = 'fourFeat(wFgt_max)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(idx+1)+'x'+str(actual_steps)+'_'+model_title+'_'+optim_title+'_dec'+str(weight_decay)+'_'+problem_name+'_Vol'+str(maxVolume_np)+'_gauss5x1_'
            visualizations.density_vis(density, compliance_loss, max_resolution, title, True, visualize, True,
                                       binary_loss=binary_compliance_loss, path=log_image_path)

# recording run time
execution_time = time.perf_counter() - start_time
sys.stderr.write('Overall runtime: {}\n'.format(execution_time))

# now query for max resolution after training finished
test_resolution = max_resolution
dataset = MeshGrid(sidelen=test_resolution, domain=domain, flatten=False)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
model_input = next(iter(dataloader))
if torch.cuda.is_available():
    model_input = model_input.cuda()

# topopt (via VoxelFEM-Optimization-Problem)
constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
filters = []
uniformDensity = maxVolume
tps = initializeTensorProductSimulator(
    orderFEM, domainCorners, test_resolution, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH
)                                                                                  
objective = pyVoxelFEM.ComplianceObjective(tps)                                    
top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, filters) 

if torch.cuda.is_available():
    maxVolume = maxVolume.cuda()
voxelfem_engine = VoxelFEMFunction.apply

# test model without activation FORGETTING ENABLED
if forgetting_activations is not None:
    pass

# test model without weight FORGETTING ENABLED
if forgetting_weights is not None:
    pass

with torch.no_grad():
    model.eval()
    density = model(model_input)
    density = physical_density(density, maxVolume)

# loss of conversion to binary by thresholding
binary_compliance_loss = utils.compute_binary_compliance_loss(density=density, top=top,
                                                              loss_engine=voxelfem_engine)
if torch.cuda.is_available():
    density = density.cpu()
compliance_loss = voxelfem_engine(density.flatten(), top)
maxVolume_np = maxVolume.detach().cpu().numpy()
grid_title = ''.join(str(i)+'x' for i in test_resolution)[:-1]
title = 'fourFeat(woFgt_test)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(actual_steps)+'_'+model_title+'_'+optim_title+'_dec'+str(weight_decay)+'_'+problem_name+'_Vol'+str(maxVolume_np)+'_gauss5x1_'
visualizations.density_vis(density, compliance_loss, max_resolution, title, True, visualize, True,
                           binary_loss=binary_compliance_loss, path=log_image_path)
title = 'fourFeat(woFgt_overall)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(actual_steps)+'_'+model_title+'_'+optim_title+'_dec'+str(weight_decay)+'_'+problem_name+'_Vol'+str(maxVolume_np)+'_gauss5x1_'
compliance_loss_array.append(compliance_loss)
title = visualizations.loss_vis(compliance_loss_array, title, True, path=log_loss_path)

utils.save_weights(model, append_path[:-1] if args.jid is None else args.jid, save_model, path=log_weights_path)

# utils.save_densities(density, gridDimensions, title, save_density, True, path='logs/densities/fourfeat_multires/')


