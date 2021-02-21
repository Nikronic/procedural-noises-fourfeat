import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd.profiler as profiler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks import MLP
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
import time
import argparse
import itertools


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
fixed_resolution = True

problem_name = 'OctaveSimplexPerlin'  # TODO
n_octaves = 4  # TODO
rho = 2  # TODO
sampling_rate = 1.0  # TODO
sampling_mode = 'deterministic_gridinit'  # TODO
max_resolution = np.array([18, 18]).reshape(1, -1)  # TODO
max_resolution = np.concatenate([max_resolution, ((max_resolution*2)*(2**(n_octaves-1)))], axis=0)

pconfprint = 'Problem configs: \n'
pconfprint += 'Problem Name: {}, Number of Octaves: {}, Rho: {}\n'.format(problem_name, n_octaves, rho)
pconfprint += 'Sampling Rate: {} using {} Sampling Method\n'.format(sampling_rate, sampling_mode)
pconfprint += 'Perlin Noise Input Grid: {}, Output Grid: {}'.format(max_resolution[0], max_resolution[1])
sys.stderr.write(pconfprint)

# record runtime
start_time = time.perf_counter()

# multires hyperparameters for single res
weight_decay = 0  # TODO
use_scheduler = False
forgetting_weights = None  # TODO
forgetting_activations = 'dropout'  # TODO
rate = 0.0  # TODO
res_order = 'ftc'
repeat_res = 5  # TODO
epoch_mode = 'constant'
resolutions = multires_utils.prepare_resolutions(interval=0, start=0, end=1, order=res_order, repeat_res=repeat_res)[:-1]  # TODO
epoch_sizes = multires_utils.prepare_epoch_sizes(n_resolutions=len(resolutions), # TODO
                                                 start=300, end=2500, 
                                                 mode=epoch_mode, constant_value=1500)
mrconfprint = 'resolution order: {}, epoch mode: {}\n'.format(res_order, epoch_mode)
mrconfprint += 'forgetting_weights: {}, forgetting_activations: {}, rate: {}\n'.format(forgetting_weights,
                                                                                       forgetting_activations,
                                                                                       rate)
mrconfprint += 'repeat resolutions: {} times \n'.format(repeat_res)
sys.stderr.write(mrconfprint)

# for plotting purposes
if forgetting_weights is not None:
    forgetting_title = 'forget_W{}'.format(forgetting_weights)
if forgetting_activations is not None:
    forgetting_title = 'forget_A{}'.format(forgetting_activations)

# create experiments folders for each run  
log_base_path = 'logs/'
log_image_path = '{}images/fourfeat/'.format(log_base_path)
log_loss_path =  '{}loss/fourfeat/'.format(log_base_path)
log_weights_path =  '{}weights/fourfeat/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, None, 
                                                experiment_id=args.jid)
log_image_path = '{}images/fourfeat/{}'.format(log_base_path, append_path)
log_loss_path =  '{}loss/fourfeat/{}'.format(log_base_path, append_path)
sys.stderr.write('image path: {}, loss path: {}\n'.format(log_image_path, log_loss_path))

# hyper parameter of positional encoding in NeRF
if not fixed_scale:
    interval_scale = 0.5
    scale = np.arange(60) * interval_scale + interval_scale
else:
    scale = [5.0]  # TODO

embedding_size = 256
sys.stderr.write('scale: {}, fourier embedding size: {}\n'.format(scale, embedding_size))

# reproducibility
seed = 8
torch.manual_seed(seed)
np.random.seed(seed)

# deep learning modules
if forgetting_activations == 'dropout':
    nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=embedding_size,
                     scale=scale[0], dropout_rate=rate, hidden_act=nn.ReLU(), output_act=None)
else:
    nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=embedding_size,
                     scale=scale[0], hidden_act=nn.ReLU(), output_act=None)
model = nerf_model
if torch.cuda.is_available():
    model.cuda()
sys.stderr.write('Deep learning model config: {}\n'.format(model))

if weight_decay > 0:
    learning_rate = 1e-3
else:
    learning_rate = 1e-4

optim = torch.optim.Adam(lr=learning_rate, params=itertools.chain(list(model.parameters())), weight_decay=weight_decay)
criterion = nn.MSELoss(reduction='sum')
# reduce on plateau
scheduler = None
if use_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='min', patience=20)
sys.stderr.write('Criterion: {}, DL optim: {}, LR scheduler: {}\n'.format(criterion, optim, scheduler))
sys.stderr.write('L2 Regularization: {}\n'.format(weight_decay))

# prepare dataset
domain = np.array([[0., 1.],[0., 1.]])
dataset = utils.PerlinMeshGrid(sidelen=max_resolution, domain=domain, rho=rho, n_octaves=n_octaves,
                               stochasticity=sampling_mode, ratio=sampling_rate,
                               mode='octave_simplex')
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
max_resolution_model_input, max_resolution_ground_truth = dataset.input, dataset.output
if torch.cuda.is_available():
    max_resolution_model_input = max_resolution_model_input.cuda()
    max_resolution_ground_truth = max_resolution_ground_truth.cuda()

# training
batch_size = 1
actual_steps = 0  # overall number of iterations over all resolutions
compliance_loss_array = []

for idx, res in enumerate(resolutions):
    for s in scale:
        # in case of getting overriden by disabling activation forgetting
        model.train()

        # save loss values for plotting
        compliance_loss_array_res = []

        # apply 'fogetting' for weights
        if forgetting_weights is not None:
            multires_utils.forget_weights(model=model, rate=rate, mode=forgetting_weights,
                                          n_neurons=256, embedding_size=embedding_size)
            sys.stderr.write('Weight forgetting has been applied. \n')
        
        # apply 'forgetting' for activations
        if forgetting_activations is not None:
            # TODO: gated_activation does not work (needs dynamic input)
            multires_utils.forget_activations(model=model, model_input=None, mode=forgetting_activations,
                                              rate=rate)

        # training of xPhys
        for step in tqdm(range(epoch_sizes[idx]), desc='Training: '):

            def closure():
                optim.zero_grad()

                global actual_steps
                actual_steps += 1

                model_input, ground_truth = next(iter(dataloader))
                if torch.cuda.is_available():
                    model_input = model_input.cuda()
                    ground_truth = ground_truth.cuda()

                density = model(model_input)
                density = density.view(ground_truth.shape)

                compliance_loss = criterion(density, ground_truth)
                compliance_loss.backward()

                # reduce LR if no reach plateau
                if use_scheduler:
                    scheduler.step(compliance_loss)

                # test model with FORGETTING DISABLED at every resolution
                if (step+1) % (epoch_sizes[idx]) == 0:
                    with torch.no_grad():
                        model.eval()  # disable forgetting (eg dropout)
                        sys.stderr.write('Visualizing (wo forgetting) output at iteration: "{}" for resolution: "{}"\n'.format(actual_steps,
                                                                                                                            density.shape))

                        density = model(max_resolution_model_input)
                        density = density.view(max_resolution_ground_truth.shape)
                        t_loss = criterion(density, max_resolution_ground_truth)

                        # visualization
                        grid_title = ''.join(str(i)+'x' for i in density.shape)[:-1]
                        title = 'FF(woFgt)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(idx+1)+'x'+str(actual_steps)
                        title +=  '_'+problem_name+'_octRho'+str(n_octaves)+'x'+str(rho)+'_'+sampling_mode+'x'+str(sampling_rate)
                        visualizations.density_vis(density, t_loss, tuple(density.shape), title, True, visualize, True,
                                                binary_loss=None, path=log_image_path)
                        model.train()  # enable forgetting (eg dropout)

                # save loss values for plotting
                compliance_loss_array_res.append(compliance_loss.detach().item())
                sys.stderr.write("Total Steps: %d, Resolution Steps: %d, Compliance loss %0.6f \n" % (actual_steps, step, compliance_loss))

                return compliance_loss

            optim.step(closure)

        compliance_loss_array.extend(compliance_loss_array_res)


# recording run time
execution_time = time.perf_counter() - start_time
sys.stderr.write('Overall runtime: {}\n'.format(execution_time))

# now query for max resolution after training finished
test_resolution = max_resolution
if torch.cuda.is_available():
    max_resolution_model_input = max_resolution_model_input.cuda()
    max_resolution_ground_truth = max_resolution_ground_truth.cuda()

# test model without activation FORGETTING ENABLED
if forgetting_activations is not None:
    pass

# test model without weight FORGETTING ENABLED
if forgetting_weights is not None:
    pass

with torch.no_grad():
    model.eval()
    density = model(max_resolution_model_input)
    density = density.view(max_resolution_ground_truth.shape)

compliance_loss = criterion(density, max_resolution_ground_truth)

grid_title = ''.join(str(i)+'x' for i in test_resolution[1])[:-1]
title = 'FF(woFgt_test)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(actual_steps)
title +=  '_'+problem_name+'_octRho'+str(n_octaves)+'x'+str(rho)+'_'+sampling_mode+'x'+str(sampling_rate)
visualizations.density_vis(density, compliance_loss, tuple(test_resolution[1]), title, True, visualize, True,
                           binary_loss=None, path=log_image_path)
title = 'FF(woFgt_overall)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(actual_steps)
title +=  '_'+problem_name+'_octRho'+str(n_octaves)+'x'+str(rho)+'_'+sampling_mode+'x'+str(sampling_rate)
compliance_loss_array.append(compliance_loss)
title = visualizations.loss_vis(compliance_loss_array, title, True, path=log_loss_path)

title = 'GT_'+problem_name+'_octRho'+str(n_octaves)+'x'+str(rho)+'_'+sampling_mode+'x'+str(sampling_rate)
visualizations.density_vis(max_resolution_ground_truth.cpu(), 0.0, max_resolution_ground_truth.shape, title, False, True, True,
                           None, log_image_path)

utils.save_weights(model, append_path[:-1] if args.jid is None else args.jid, save_model, path=log_weights_path)
