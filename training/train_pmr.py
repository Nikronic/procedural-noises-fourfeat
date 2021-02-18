# PMR: Parallel Multi-Resolution
import os, sys
import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
from torch.utils.data import DataLoader
import torch.autograd.profiler as profiler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks import MLP
import utils
import filtering
import multires_utils
import visualizations

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from tqdm import tqdm
import argparse
from datetime import datetime
import itertools, copy, time


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
interpolation = False
fixed_scale = True
fixed_resolution = False

problem_name = 'OctaveSimplexPerlin'  # TODO
n_octaves = 3  # TODO
rho = 2  # TODO
max_resolution = np.array([18, 18]).reshape(1, -1)  # TODO
max_resolution = np.concatenate([max_resolution, ((max_resolution*2)*(2**(n_octaves-1)))], axis=0)
gridDimensions = np.array([12, 12]).reshape(1, -1)  # TODO
gridDimensions = np.concatenate([gridDimensions, ((gridDimensions*2)*(2**(n_octaves-1)))], axis=0)

# record runtime
start_time = time.perf_counter()

# multires hyperparameters
adaptive_filtering_configs = [0.1, -1, 0.1, -1, 0.1, 1]  # (0<...<1, -x) means (no update, no usage) filters respectively
weight_decay = 0  # TODO
use_scheduler = False
forgetting_weights = None  # TODO
forgetting_activations = 'dropout'  # TODO
rate = 0.2  # TODO: 1.0 disables weight forgetting (similar to p=0.0 in dropout, p=0.2 == rate=0.8)
res_order = 'ftc'
repeat_res = 1  # TODO
epoch_mode = 'constant'
resolutions = multires_utils.prepare_resolutions(interval=1, start=0, end=5, order=res_order, repeat_res=repeat_res)[:-1]  # TODO
epoch_sizes = multires_utils.prepare_epoch_sizes(n_resolutions=len(resolutions), # TODO
                                                 start=300, end=2500, 
                                                 mode=epoch_mode, constant_value=1200)
mrconfprint = 'resolution order: {}, epoch mode: {}\n'.format(res_order, epoch_mode)
mrconfprint += 'forgetting_weights: {}, forgetting_activations: {}, rate: {}\n'.format(forgetting_weights,
                                                                                       forgetting_activations,
                                                                                       rate)
mrconfprint += 'repeat resolutions: {} times \n'.format(repeat_res)
mrconfprint += 'adaptive filtering configs: {} \n'.format(adaptive_filtering_configs)
sys.stderr.write(mrconfprint)

# for plotting purposes
if forgetting_weights is not None:
    forgetting_title = 'forget_W{}'.format(forgetting_weights)
if forgetting_activations is not None:
    forgetting_title = 'forget_A{}'.format(forgetting_activations)

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
    scale = [100.0]  # TODO

embedding_size = 256
sys.stderr.write('scale: {}, fourier embedding size: {}\n'.format(scale, embedding_size))

gridDimensions_ = copy.deepcopy(gridDimensions)
if max_resolution is None:
    max_resolution = gridDimensions

# reproducibility
seed = 8
torch.manual_seed(seed)
np.random.seed(seed)

domain = np.array([[0., 1.],[0., 1.]])

# deep learning modules
if forgetting_activations == 'dropout':
    nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=embedding_size,
                        scale=scale[0], dropout_rate=rate, hidden_act=nn.ReLU(), output_act=nn.Sigmoid())
else:
    nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=embedding_size,
                        scale=scale[0], hidden_act=nn.ReLU(), output_act=nn.Sigmoid())
model = nerf_model
if torch.cuda.is_available():
    model.cuda()
sys.stderr.write('Deep learning model config: {}\n'.format(model))

# filtering
projection_filter = filtering.ProjectionFilter(beta=1)
smoothing_filter = filtering.SmoothingFilter(radius=1)
gauss_smoothing_filter = filtering.GaussianSmoothingFilter(sigma=1)
filters = [projection_filter, smoothing_filter, gauss_smoothing_filter]

if weight_decay > 0:
    learning_rate = 1e-3
else:
    learning_rate = 1e-4

optim = torch.optim.Adam(lr=learning_rate, params=itertools.chain(list(model.parameters())), weight_decay=weight_decay)
criterion = nn.L1Loss(reduction='sum')
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

# prepare inputs
model_inputs = []
ground_truths = []
resolutions = list(resolutions)
for idx, res in enumerate(resolutions):
    gridDimensions = np.array(gridDimensions_[0]) + res
    gridDimensions = gridDimensions.reshape(1, -1)
    gridDimensions = np.concatenate([gridDimensions, ((gridDimensions*2)*(2**(n_octaves-1)))], axis=0)
    resolutions[idx] = gridDimensions

    dataset = utils.PerlinMeshGrid(sidelen=gridDimensions, domain=domain, 
                                   rho=rho, n_octaves=n_octaves, mode='octave_simplex')
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
    model_input, ground_truth = next(iter(dataloader))
    if torch.cuda.is_available():
        model_input = model_input.cuda()
        ground_truth = ground_truth.cuda()

    model_inputs.append(model_input)
    ground_truths.append(ground_truth)
sys.stderr.write('Model inputs (coordinates) and ground truths (noises) have been constructed.\n')

# training
for idx, res in enumerate(resolutions):
    for s in scale:
        # in case of getting overriden by disabling activation forgetting
        model.train()

        # save loss values for plotting
        compliance_loss_array_res = []  # type: ignore

        # apply 'fogetting' for weights
        if forgetting_weights is not None:
            multires_utils.forget_weights(model=model, rate=rate, mode=forgetting_weights,
                                          n_neurons=256, embedding_size=embedding_size)
            sys.stderr.write('Weight forgetting has been applied. \n')
        
        # apply 'forgetting' for activations
        if forgetting_activations is not None:
            multires_utils.forget_activations(model=model, model_input=model_input, mode=forgetting_activations,
                                              rate=rate)

        # reset adaptive filtering
        for filt in filters:
            filt.reset_params()  # type: ignore
        sys.stderr.write('Adaptive filtering has been reset to their defaults. \n')

        # training of xPhys
        for step in tqdm(range(epoch_sizes[idx]), desc='Training: '):
            
            def closure():
                optim.zero_grad()

                global actual_steps
                actual_steps += 1

                # overall compliance over n resolution as the weighted sum with equal weights
                compliance_loss = 0 

                # aka x
                for i in range(len(model_inputs)):
                    
                    density = model(model_inputs[i])
                    ground_truth = ground_truths[i]
                    density = density.view(ground_truth.shape)

                    single_res_compliance = criterion(density, ground_truth)

                    # adaptive filtering
                    if adaptive_filtering_configs is not None:
                        density = filtering.apply_filters_group(x=density, filters=filters, configs=adaptive_filtering_configs)
                        filtering.update_adaptive_filtering(iteration=step, filters=filters, configs=adaptive_filtering_configs)

                    # sum of compliances over all resolutions
                    compliance_loss = compliance_loss + single_res_compliance

                    # test model with FORGETTING DISABLED at every resolution
                    if (step+1) % (epoch_sizes[idx]) == 0:
                        model.eval()  # disable forgetting (eg dropout)
                        sys.stderr.write('Visualizing (wo forgetting) output at iteration: "{}" for resolution: "{}"'.format(actual_steps,
                                                                                                                             resolutions[i][1]))

                        density = model(model_inputs[i])
                        density = density.view(ground_truth.shape)

                        # visualization
                        grid_title = ''.join(str(i)+'x' for i in resolutions[i][1])[:-1]
                        adaptive_filtering_configs_title = ''.join(str(i)+'|' for i in adaptive_filtering_configs)[:-1]
                        title = 'FF(woFgt)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(idx+1)+'x'+str(actual_steps)
                        title += '_dec'+str(weight_decay)+'_'+problem_name+'_F'+adaptive_filtering_configs_title
                        visualizations.density_vis(density, single_res_compliance, tuple(resolutions[i][1]), title, True, visualize, True,
                                                   binary_loss=None, path=log_image_path)
                        model.train()  # enable forgetting (eg dropout)

                # reduce LR if no reach plateau
                if use_scheduler:
                    scheduler.step(compliance_loss)

                compliance_loss.backward()

                # save loss values for plotting
                compliance_loss_array_res.append(compliance_loss.detach().item())
                sys.stderr.write("Total Steps: %d, Resolution Steps: %d, MAE sum loss %0.6f" % (actual_steps,
                                                                                                step, compliance_loss))
                return compliance_loss

            optim.step(closure)

        compliance_loss_array.extend(compliance_loss_array_res)


# recording run time
execution_time = time.perf_counter() - start_time
sys.stderr.write('Overall runtime: {}\n'.format(execution_time))

# now query for max resolution after training finished
test_resolution = max_resolution
dataset = utils.PerlinMeshGrid(sidelen=test_resolution, domain=domain, 
                               rho=rho, n_octaves=n_octaves, mode='octave_simplex')
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
model_input, ground_truth = next(iter(dataloader))
if torch.cuda.is_available():
    model_input = model_input.cuda()
    ground_truth = ground_truth.cuda()

# test model without activation FORGETTING ENABLED
if forgetting_activations is not None:
    pass

# test model without weight FORGETTING ENABLED
if forgetting_weights is not None:
    pass

with torch.no_grad():
    model.eval()
    density = model(model_input)
    density = density.view(ground_truth.shape)

compliance_loss = criterion(density, ground_truth)

grid_title = ''.join(str(i)+'x' for i in test_resolution[1])[:-1]
adaptive_filtering_configs_title = ''.join(str(i)+'|' for i in adaptive_filtering_configs)[:-1]
title = 'FF(woFgt_test)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(actual_steps)
title += '_dec'+str(weight_decay)+'_'+problem_name+'_F'+adaptive_filtering_configs_title
visualizations.density_vis(density, compliance_loss, tuple(test_resolution[1]), title, True, visualize, True,
                           binary_loss=None, path=log_image_path)
title = 'FF(woFgt_overall)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(actual_steps)
title += '_dec'+str(weight_decay)+'_'+problem_name+'_F'+adaptive_filtering_configs_title
compliance_loss_array.append(compliance_loss)
title = visualizations.loss_vis(compliance_loss_array, title, True, path=log_loss_path)

utils.save_weights(model, append_path[:-1] if args.jid is None else args.jid, save_model, path=log_weights_path)

# utils.save_densities(density, gridDimensions, title, save_density, True, path='logs/densities/fourfeat_multires/')