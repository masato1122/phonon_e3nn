import sys
import os
from typing import Dict, Union
import copy
import pandas as pd

import torch
from torch_geometric.data import Data
from torch_cluster import radius_graph

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from e3nn.nn.models.gate_points_2101 import Convolution, smooth_cutoff, tp_path_exists

import math
import time
from tqdm import tqdm

# #--- for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phonon_e3nn.mpl.initialize import (set_matplot, set_axis, set_legend)

# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'

# standard formatting for plots
# fontsize = 16
# textsize = 14
# font_family = 'sans-serif'
# sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

class CustomCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x


class Network(torch.nn.Module):
    r"""equivariant neural network
    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features
    irreps_hidden : `e3nn.o3.Irreps`
        representation of the hidden features
    irreps_out : `e3nn.o3.Irreps`
        representation of the output features
    irreps_node_attr : `e3nn.o3.Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes
    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes
        the edge attributes are :math:`h(r) Y(\vec r / r)`
        where :math:`h` is a smooth function that goes to zero at ``max_radius``
        and :math:`Y` are the spherical harmonics polynomials
    layers : int
        number of gates (non linearities)
    max_radius : float
        maximum radius for the convolution
    number_of_basis : int
        number of basis on which the edge length are projected
    radial_layers : int
        number of hidden layers in the radial fully connected network
    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network
    num_neighbors : float
        typical number of nodes at a distance ``max_radius``
    num_nodes : float
        typical number of nodes in a graph
    """
    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_node_attr,
        layers,
        mul,
        lmax,
        max_radius,
        number_of_basis=10,
        radial_layers=1,
        radial_neurons=100,
        num_neighbors=1.,
        num_nodes=1.,
        reduce_output=True,
    ) -> None:
        super().__init__()
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
            irreps = gate.irreps_out
            self.layers.append(CustomCompose(conv, gate))

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
        )

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        if 'edge_index' in data:
            edge_src = data['edge_index'][0]  # edge source
            edge_dst = data['edge_index'][1]  # edge destination
            edge_vec = data['edge_vec']
        
        else:
            edge_index = radius_graph(data['pos'], self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]

        return batch, edge_src, edge_dst, edge_vec

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """evaluate the network
        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        batch, edge_src, edge_dst, edge_vec = self.preprocess(data)
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='gaussian',
            cutoff=False
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and 'x' in data:
            assert self.irreps_in is not None
            x = data['x']
        else:
            assert self.irreps_in is None
            x = data['pos'].new_ones((data['pos'].shape[0], 1))

        if self.input_has_node_attr and 'z' in data:
            z = data['z']
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = data['pos'].new_ones((data['pos'].shape[0], 1))

        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
            if torch.isnan(x).any():
                print("NaN detected in layer output")
                sys.exit()
        
        if self.reduce_output:
            x = torch.scatter(x, batch, dim=0).div(self.num_nodes**0.5)
        
        # Ensure target tensor is reshaped correctly
        if x.shape[-1] == 1 and len(data['target'].shape) == 1:
            data['target'] = data['target'].unsqueeze(-1)
        
        # Debugging: Check for NaN values in the output
        if torch.isnan(x).any():
            print("NaN detected in model output")
            sys.exit()
        
        return x


def visualize_layers(model, textsize=12):
    
    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['gate', 'tp', 'lin2', 'gate']))
    try: layers = model.mp.layers
    except: layers = model.layers

    num_layers = len(layers)
    num_ops = max([len([k for k in list(layers[i].first._modules.keys()) if k not in ['fc', 'alpha']])
                   for i in range(num_layers-1)])

    fig, ax = plt.subplots(num_layers, num_ops, figsize=(14,3.5*num_layers))
    for i in range(num_layers - 1):
        ops = layers[i].first._modules.copy()
        ops.pop('fc', None); ops.pop('alpha', None)
        for j, (k, v) in enumerate(ops.items()):
            ax[i,j].set_title(k, fontsize=textsize)
            v.cpu().visualize(ax=ax[i,j])
            ax[i,j].text(0.7,-0.15,'--> to ' + layer_dst[k], fontsize=textsize-2, transform=ax[i,j].transAxes)

    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['output', 'tp', 'lin2', 'output']))
    ops = layers[-1]._modules.copy()
    ops.pop('fc', None); ops.pop('alpha', None)
    for j, (k, v) in enumerate(ops.items()):
        ax[-1,j].set_title(k, fontsize=textsize)
        v.cpu().visualize(ax=ax[-1,j])
        ax[-1,j].text(0.7,-0.15,'--> to ' + layer_dst[k], fontsize=textsize-2, transform=ax[-1,j].transAxes)

    fig.subplots_adjust(wspace=0.3, hspace=0.5)


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))

    
def evaluate(model, dataloader, loss_funcs, device, alpha=None):
    
    model.eval()
    losses_cumu = {'mse': 0., 'mae': 0., 'custom': 0.}
    # start_time = time.time()
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(d)
            
            # Debugging: Check for NaN values in the target
            if torch.isnan(d.target).any():
                print("NaN detected in target")
            
            if len(output.shape) == 1 and len(d.target.shape) == 2:
                output = output.unsqueeze(-1)
                
            losses = {}
            losses['custom'] = loss_funcs['custom'](output, d.target, alpha=alpha).cpu()
            losses['mse'] = loss_funcs['mse'](output, d.target).cpu()
            losses['mae'] = loss_funcs['mae'](output, d.target).cpu()
            
            for key in losses_cumu:
                losses_cumu[key] += losses[key].detach().item()
    
    for key in losses_cumu:
        losses_cumu[key] /= len(dataloader)
    
    return losses_cumu

def train(model, optimizer, 
          dataloader_train, dataloader_valid, dataloader_test,
          loss_funcs, 
          file_model=None, 
          mono_increase=False, outdir=None,
          num_epochs=101, num_epochs_limit=None, patience=50,
          lr_min=None,
          scheduler=None, device="cpu"):
    """ Train a model using custom loss function
    """
    model.to(device)

    logfile = outdir + '/log.csv'
    if os.path.exists(logfile):
        df_log = pd.read_csv(logfile)
    else:
        df_log = pd.DataFrame()
    
    # checkpoint_generator = loglinspace(0.3, 5)
    checkpoint_generator = loglinspace(0.3, 1)
    # checkpoint = next(checkpoint_generator)
    start_time = time.time()
    
    loss_fn_mse = loss_funcs['mse']
    loss_fn_mae = loss_funcs['mae']
    loss_fn_custom = loss_funcs['custom']
    alpha = loss_funcs['grad_weight']
    # adaptive_loss = loss_funcs['adaptive']
    
    try:
        # models = torch.load(run_name + '.torch', weights_only=True)
        # model.load_state_dict(torch.load(run_name + '.torch', weights_only=True)['state'])
        # model.load_state_dict(torch.load(file_model, weights_only=True)['best'])
        results = torch.load(file_model, weights_only=True)    
    except:
        results = {}
        history = []
        s0 = 0
        total_step = 0
        prev_valid_loss = float('inf')
        best_model = None
        loss_increasing_counter = 0
    else:
        try:
            model.load_state_dict(results['best'])
            print("\nLoaded the best model:", file_model)
        except:
            model.load_state_dict(results['state'])
            print('\nLoaded a model:', file_model)
        
        best_model = copy.deepcopy(model)
        
        ### Get info related to the best model
        history = results['history']
        ibest = min(range(len(history)), key=lambda i: history[i]['valid']['loss'])
        total_step = history[-1]['total_step'] + 1
        prev_valid_loss = history[ibest]['valid']['loss']
        try:
            loss_increasing_counter = history[-1]['loss_increase']
        except:
            loss_increasing_counter = 0
    
    # loss_funcs = {'custom': loss_fn_custom, 'mse': loss_fn_mse, 'mae': loss_fn_mae}
    
    print()
    for step in range(num_epochs):
        
        model.train()
        
        ## Adjust learning rate
        lr_now = optimizer.param_groups[0]['lr']
        if lr_min is not None:
            if lr_now < lr_min:
                scheduler.gamma = 1.0
                optimizer.param_groups[0]['lr'] = lr_min
            
        # for j, d in tqdm(enumerate(dataloader_train), total=len(dataloader_train), bar_format=bar_format):
        for j, d in enumerate(dataloader_train):
            
            d.to(device)
            output = model(d)
            
            if len(output.shape) == 1 and len(d.target.shape) == 2:
                output = output.unsqueeze(-1)
            
            losses = {}
            losses['mse'] = loss_fn_mse(output, d.target).cpu()
            losses['mae'] = loss_fn_mae(output, d.target).cpu()
            
            ## If mono_increase is False, 'custom' loss function is not used.
            if mono_increase:
                ### ver.1
                # adaptive_loss.update_a(losses['mse'])
                # losses['custom'] = loss_fn_custom(output, d.target, alpha=adaptive_loss.a).cpu()
                ### ver.2
                losses['custom'] = loss_fn_custom(output, d.target, alpha=alpha).cpu()
            else:
                losses['custom'] = losses['mse']
            
            optimizer.zero_grad()
            losses['custom'].backward()
            
            optimizer.step()
        
        end_time = time.time()
        wall = end_time - start_time
        
        ### ver. original
        # if step == checkpoint:
        #     checkpoint = next(checkpoint_generator)
        #     assert checkpoint > step
        ### ver. modified
        if step >= 0:
            
            valid_avg_loss = evaluate(model, dataloader_valid, loss_funcs, device, alpha=alpha)
            train_avg_loss = evaluate(model, dataloader_train, loss_funcs, device, alpha=alpha)
            test_avg_loss = evaluate(model, dataloader_test, loss_funcs, device, alpha=alpha)
            
            current_result = {
                'total_step': total_step,
                'step': step,
                'loss_increase': loss_increasing_counter,
                'wall': wall,
                # 'batch': {
                #     'loss': losses['custom'].item(),
                #     'mse':  losses['mse'].item(),
                #     'mae':  losses['mae'].item(),
                # },
                'valid': {
                    'loss': valid_avg_loss['custom'],
                    'mse': valid_avg_loss['mse'],
                    'mae': valid_avg_loss['mae'],
                },
                'train': {
                    'loss': train_avg_loss['custom'],
                    'mse': train_avg_loss['mse'],
                    'mae': train_avg_loss['mae'],
                },
                'test': {
                    'loss': test_avg_loss['custom'],
                    'mse': test_avg_loss['mse'],
                    'mae': test_avg_loss['mae'],
                },
            }
            
            history.append(current_result)
            
            results = {
                'history': history,
                'state': model.state_dict()
            }
            if best_model is None:
                results['best'] = model.state_dict()
            else:
                results['best'] = best_model.state_dict()
            
            # print(history[-1]['batch']['loss'], history[-1]['valid']['loss'], history[-1]['train']['loss'])
            
            print(f"Step {total_step+1:3d} ({step+1:2d}) " +
                  f"lr = {lr_now: .2e} " +
                  f"increasing {loss_increasing_counter:2d} " +
                  f"train loss = {train_avg_loss['custom']:8.4f} (" +
                  f"custom: {train_avg_loss['custom'] - train_avg_loss['mse']: .3e} " +
                  f"alpha: {alpha: .2e} " +
                  f"MSE: {train_avg_loss['mse']: .3f} " +
                  f"MAE: {train_avg_loss['mae']: .3f}) " +
                  f"valid loss = {valid_avg_loss['custom']:8.4f} (" +
                  f"MAE: {valid_avg_loss['mae']: .3f}) " + 
                  f"test loss = {test_avg_loss['custom']:8.4f} (" +
                  f"MAE: {test_avg_loss['mae']: .3f}) "
                  #   f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}"
                  )
            
            with open(file_model, 'wb') as f:
                torch.save(results, f)
            
            if (num_epochs_limit is not None and total_step >= num_epochs_limit):
                print("\nReached maximum number of epochs. Exiting training.")
                break
            
            #### Save log file
            if step % 10 == 0:
                output_figure = True
            else:
                output_figure = False
            
            try:
                df_log = save_log(df_log, current_result, filename=logfile, output_figure=output_figure)
            except:
                pass
        
        ## Early stopping
        if valid_avg_loss['custom'] > prev_valid_loss:
            loss_increasing_counter += 1
        else:
            loss_increasing_counter = 0
            prev_valid_loss = valid_avg_loss['custom']
            best_model = copy.deepcopy(model)
            print('Update the best model')
        
        if loss_increasing_counter == patience:
            print(f"\nValidation loss has been increasing for {patience} times. Exiting training.")
            break
        
        if scheduler is not None:
            scheduler.step()

        total_step += 1


def save_log(df_log, current_result, filename='log.csv', output_figure=False):
    """ Save log for each step """
    dump = {}
    for key in current_result:
        if isinstance(current_result[key], dict):
            for key2 in current_result[key]:
                if isinstance(current_result[key][key2], float):
                    key_new = key + '_' + key2
                    val = current_result[key][key2]
                    dump[key_new] = val
                else:
                    print("Error: Nested dictionary is not supported.")
                    sys.exit()
        else:
            val = current_result[key]
            key_new = key
            dump[key_new] = val
    
    out = pd.DataFrame([dump])
    df_log = pd.concat([df_log, out], ignore_index=True)
    
    with open(filename, 'w') as f:
        df_log.to_csv(f, index=False)
        f.close()
    
    from phonon_e3nn.utils.plotter import plot_loss_history
    if output_figure:
        xdat = df_log['total_step'].values
        loss_train = df_log['train_mae'].values
        loss_valid = df_log['valid_mae'].values
        plot_loss_history(
            xdat, loss_train, loss_valid, 
            figname=filename.replace('.csv', '.png'))
    
    return df_log
