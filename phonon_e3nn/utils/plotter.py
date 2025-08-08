#
# Original code: https://github.com/ninarina12/phononDoS_tutorial
# Modified by M. Ohnishi
# Modified on February 06, 2025
#
# The code is partially originated from https://github.com/ninarina12/phononDoS_tutorial
# 
# MIT License
# 
# Copyright (c) 2024 Masato Ohnishi at The Institute of Statistical Mathematics
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import numpy as np
import pandas as pd

import torch_geometric as tg

import networkx as nx
from ase import Atoms
from ase.visualize.plot import plot_atoms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phonon_e3nn.mpl.initialize import set_matplot, set_axis, set_legend
from phonon_e3nn.utils.utils_data import element_representation, split_subplot

font_family = 'sans-serif'

def plot_prediction_parity(
    df, indices, target=None, figname='fig_parity.png', loss_type='mae',
    dpi=600, fontsize=7, fig_width=2.8, aspect=0.9, lw=0.5, ms=2.0):

    cmap = plt.get_cmap('tab10')
    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width))
    
    ax = plt.subplot()
    ax.set_title(target)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    
    vmin = 100
    vmax = -100
    # markers = ['x', '^', 's', 'D', 'v', '<', '>', 'p', 'P', '*', 'X']
    for i, (kind, idx) in enumerate(indices.items()):
        
        data = df.iloc[idx]
        xdat = np.asarray(data[target])
        ydat = np.asarray(data[target+'_pred'])
        
        alpha = 1.0
        if i == 0:
            c = 'blue'
            alpha = 1.0
            marker = 'x'
            lw = 0.2
        elif i == 1:
            c = cmap(2)
            marker = '^'
            lw = 0.3
        elif i == 2:
            c = 'red'
            marker = 'o'
            lw = 0.5
        
        if loss_type == 'mse':
            error = np.mean((xdat - ydat)**2)
        elif loss_type == 'mae':
            error = np.mean(np.abs(xdat - ydat))
        else:
            raise ValueError('Unknown loss type')
        label = f'{kind} ({len(xdat)} data, {loss_type.upper()}: {error:.3f})'
        
        ax.plot(xdat, ydat, linestyle='None', lw=lw, 
                marker=marker, markersize=ms, alpha=alpha,
                mfc='none', mew=lw, mec=c, label=label)
        
        vmin_now = min(xdat.min(), ydat.min())
        vmax_now = max(xdat.max(), ydat.max())
        vmin = min(vmin, vmin_now)
        vmax = max(vmax, vmax_now)
    
    ax.plot([vmin, vmax], [vmin, vmax], linestyle='--', lw=lw, color='grey', zorder=-1)
    
    set_axis(ax)
    set_legend(ax, fs=6, alpha=0.5, loc='upper left')
    
    fig.savefig(figname, dpi=dpi, bbox_inches='tight')
    print(" Output", figname)
    return fig

def plot_prediction_single(ax, data, lw=0.8, marker='none', ms=2.0, 
                           ylim=[-0.05, 1.05], col_pred='blue', loss_type='mae',
                           fs_title=6):
    # loc_legend='upper right', fs_legend=5,
    """ plot the prediction of a single material (parameter dependent property)
    
    Args:
        ax: matplotlib axis
        data: pandas df.Series
    """
    if isinstance(data, pd.DataFrame):
        data = data.iloc[0]
    
    ## Preset
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    cmap = plt.get_cmap("tab10")
    xticks = None; mxticks = None
    yticks = None; myticks = None
    
    ## Get the target column
    target = [col for col in data.index if '_pred' in col]
    if len(target) != 1:
        print(target, 'were found.')
        raise ValueError('No or too many columns with "_pred"')
    target = target[0].replace('_pred', '')
    
    ## formula
    formula = data['formula'].translate(sub)
    ydat1 = np.asarray(data[target])
    ydat2 = np.asarray(data[target+'_pred'])
    
    ## MSE
    if loss_type == 'mse':
        error = np.mean((ydat1 - ydat2)**2)
    elif loss_type == 'mae':
        error = np.mean(np.abs(ydat1 - ydat2))
    else:
        raise ValueError('Unknown loss type')
    title = f"{formula} ({loss_type.upper()}: {error:.3f})"
    ax.set_title(title, fontsize=fs_title, pad=5)
    
    ## Get the x column name
    xscale = 'linear'
    if target.endswith('_freq'):
        xcol = 'phfreq'
        xlabel = '${\\rm \\omega / \\omega_{max}}$'
        xticks = 1; mxticks = 2
        
        xdat = data[xcol]
        xdat /= np.max(xdat)
        
    elif target.endswith('_mfp'):
        
        xcol = 'log_mfp'
        
        ### ver.1
        xticks = 2; mxticks = 2
        xlabel = '$\\log_{10}[{\\rm MFP (nm)}]$'
        xdat = data[xcol]
        ### ver.2
        # xticks = None; mxticks = None
        # xlabel = 'MFP (nm)'
        # xdat = np.power(10, data[xcol])
        # xscale = 'log'
    else:
        print(target)
        print(data.columns)
        raise ValueError('Unknown target column. Please add the definition.')
    
    ## y-label
    if 'kspec' in target:
        ylabel = '\\kappa_{spec}'
    elif 'kcumu' in target:
        ylabel = '\\kappa_{cumul}'
    if '_norm' in target:
        ylabel += '^{\\rm norm}'
        yticks = 1; myticks = 4
    
    ylabel = "${\\rm %s}$" % ylabel
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ## Get the x and y data
    for i, ydat in enumerate([ydat1, ydat2]):
        
        if i == 0:
            color = 'grey'
            lw_scale = 1.0
            label = 'True'
        else:
            color = col_pred
            lw_scale = 1.0
            label = 'Predicted'
        
        ax.plot(xdat, ydat, linestyle='-',
                marker=marker, ms=ms,
                c=color, lw=lw*lw_scale, label=label)
    
    ax.set_ylim(ylim)
    set_axis(ax, xscale=xscale, 
             xticks=xticks, mxticks=mxticks, 
             yticks=yticks, myticks=myticks)
    


def plot_element_representation(
    stats, idx_train, idx_valid, idx_test, datasets, species, 
    figname='fig_element_representation.png', 
    fig_width=6.0, aspect=0.5, fontsize=6, dpi=500, lw=0.8):
    """ plot element representation in each dataset
    
    Original code : https://github.com/ninarina12/phononDoS_tutorial
    Modified by M. Ohnishi
    
    """
    # plot element representation in each dataset
    stats['train'] = stats['data'].map(lambda x: element_representation(x, np.sort(idx_train)))
    stats['valid'] = stats['data'].map(lambda x: element_representation(x, np.sort(idx_valid)))
    stats['test'] = stats['data'].map(lambda x: element_representation(x, np.sort(idx_test)))
    stats = stats.sort_values('symbol')
    
    ### original
    # fig, ax = plt.subplots(2,1, figsize=(fig_width, aspect*fig_width))
    
    ### modified
    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width))
    fig.subplots_adjust(hspace=0.2)
    
    ax = [plt.subplot(2,1,1), plt.subplot(2,1,2)]
    
    b0, b1 = 0., 0.
    for i, dataset in enumerate(datasets):
        split_subplot(ax[0], stats[:len(stats)//2], species[:len(stats)//2], dataset, bottom=b0, lw=lw, legend=True)
        split_subplot(ax[1], stats[len(stats)//2:], species[len(stats)//2:], dataset, bottom=b1, lw=lw)
        b0 += stats.iloc[:len(stats)//2][dataset].values
        b1 += stats.iloc[len(stats)//2:][dataset].values
    
    set_legend(ax[0], fs=6, loc='lower right')
    fig.savefig(figname, dpi=dpi, bbox_inches='tight')
    print(' Output', figname)
    
def plot_loss_history(steps, loss_train, loss_valid, 
                      figname='fig_loss.png', 
                      fig_width=2.3, aspect=0.9, fontsize=7, 
                      ms=1.2, lw=1.0, dpi=300):
    
    cmap = plt.get_cmap("tab10")
    
    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width))
    ax = plt.subplot()
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    
    ax.plot(steps, loss_train, 'o-', color=cmap(0), lw=lw, ms=ms, label="Training")
    ax.plot(steps, loss_valid, '^-', color=cmap(1), lw=lw, ms=ms, label="Validation")
    
    set_axis(ax)
    set_legend(ax, fs=6, loc='upper right')
    fig.savefig(figname, dpi=dpi, bbox_inches='tight')
    print(' Output', figname)

def get_lattice_parameters(df):
    """ lattice parameter statistics
    
    Original code : https://github.com/ninarina12/phononDoS_tutorial
    Modified by M. Ohnishi
    
    """
    a = []
    for entry in df.itertuples():
        a.append(entry.structure.cell.cellpar()[:3])
    return np.stack(a)

def plot_structure(structure, figname='fig_structure.png', 
                   rotation=('10x, 70y, 10z'),
                   fontsize=7, fig_width=2.5, aspect=0.9, dpi=300):
    """ Plot the structure of a given entry in the dataframe
    
    Original code : https://github.com/ninarina12/phononDoS_tutorial
    Modified by M. Ohnishi
    
    Args:
        structure (Atoms): ASE Atoms object representing the crystal structure
    
    """
    from ase.visualize.plot import plot_atoms
    cmap = plt.get_cmap('terrain')
    
    symbols = np.unique(list(structure.symbols))
    z = dict(zip(symbols, range(len(symbols))))
    
    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width))
    ax = plt.subplot()
    
    ax.set_xlabel("$x_1 ({\\rm \\AA})$")
    ax.set_ylabel("$x_2 ({\\rm \\AA})$")
    
    norm = plt.Normalize(vmin=0, vmax=len(symbols)-1)
    color = [matplotlib.colors.to_hex(k) for k in cmap(norm([z[j] for j in list(structure.symbols)]))]
    plot_atoms(structure, ax, radii=0.25, colors=color, rotation=rotation)
    
    fig.savefig(figname, dpi=300, bbox_inches='tight')
    print(' Output', figname)

def plot_lattice_parameters(
    df, figname='fig_lattice.png', fig_width=2.5, aspect=0.7, fontsize=7, dpi=300):
    """ plot lattice parameter statistics
    
    Original code : https://github.com/ninarina12/phononDoS_tutorial
    Modified by M. Ohnishi
    
    """
    cmap = plt.get_cmap('tab10')
    a = get_lattice_parameters(df)
    
    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width))
    
    ax = plt.subplot()
    ax.set_xlabel('Lattice parameter')
    ax.set_ylabel('Number of examples')
    
    b = 0.
    bins = 50
    for i, (d, n) in enumerate(zip(['a', 'b', 'c'], [a[:,0], a[:,1], a[:,2]])):
        color = [cmap(i)[k] for k in range(3)]  ## adjusted
        y, bins, _, = ax.hist(n, bins=bins, fc=color+[0.7], ec=color, bottom=b, label=d,
                              linewidth=0.5)
        b += y
    
    set_axis(ax)
    set_legend(ax, fs=6, loc='upper right')
    fig.savefig(figname, dpi=dpi, bbox_inches='tight')
    print(' Output', figname)
    print('average lattice parameter (a/b/c):', a[:,0].mean(), '/', a[:,1].mean(), '/', a[:,2].mean())

def plot_example(
    df, i=12, label_edges=False, 
    fontsize=6, fig_width=6.0, aspect=0.5, dpi=400, 
    figname='fig_example.png'):
    """ plot an example crystal structure and graph 
    
    Original code : https://github.com/ninarina12/phononDoS_tutorial
    Modified by M. Ohnishi
    
    """
    
    cmap = plt.get_cmap('tab10')
    
    # plot an example crystal structure and graph
    entry = df.iloc[i]['data']

    # get graph with node and edge attributes
    g = tg.utils.to_networkx(entry, node_attrs=['symbol'], edge_attrs=['edge_len'], to_undirected=True)

    # remove self-loop edges for plotting
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    node_labels = dict(zip([k[0] for k in g.nodes.data()], [k[1]['symbol'] for k in g.nodes.data()]))
    edge_labels = dict(zip([(k[0], k[1]) for k in g.edges.data()], [k[2]['edge_len'] for k in g.edges.data()]))

    # project positions of nodes to 2D for plotting
    pos = dict(zip(list(g.nodes), [np.roll(k,2)[:-1][::-1] for k in entry.pos.numpy()]))
    
    # plot unit cell
    set_matplot(fontsize=fontsize)
    fig, ax = plt.subplots(
        1, 2, figsize=(fig_width, aspect*fig_width), 
        gridspec_kw={'width_ratios': [2,3]})
    
    atoms = Atoms(
        symbols=entry.symbol, 
        positions=entry.pos.numpy(), 
        cell=entry.lattice.squeeze().numpy(), 
        pbc=True)
    
    symbols = np.unique(entry.symbol)
    z = dict(zip(symbols, range(len(symbols))))
    norm = plt.Normalize(vmin=0, vmax=len(symbols)-1)
    color = [matplotlib.colors.to_hex(k) for k in cmap(norm([z[j] for j in entry.symbol]))]
    plot_atoms(atoms, ax[0], radii=0.5, scale=0.5, colors=color, rotation=('30x, 30y, 0z'))

    # plot graph
    nx.draw_networkx(
        g, ax=ax[1], labels=node_labels, pos=pos, 
        font_family=font_family, font_size=fontsize, 
        linewidths=0.5, node_size=70, node_color=color, 
        edge_color='gray')
    
    if label_edges:
        try:
            nx.draw_networkx_edge_labels(
                g, ax=ax[1], edge_labels=edge_labels, pos=pos, 
                font_family=font_family, font_size=fontsize,
                label_pos=0.5,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.0))
        except Exception as e:
            print("Error drawing edge labels:", e)
    
    # format axes
    ax[0].set_xlabel(r'$x_1\ (\AA)$')
    ax[0].set_ylabel(r'$x_2\ (\AA)$')
    ax[0].set_title('Crystal structure', fontsize=fontsize)
    ax[1].set_aspect('equal')
    ax[1].axis('off')
    ax[1].set_title('Crystal graph', fontsize=fontsize)
    pad = np.array([-0.5, 0.5])
    ax[1].set_xlim(np.array(ax[1].get_xlim()) + pad)
    ax[1].set_ylim(np.array(ax[1].get_ylim()) + pad)
    
    fig.savefig(figname, dpi=dpi, bbox_inches='tight')
    print(' Output', figname)    
    return fig

def visualize_layers(
    model, figname='fig_model.png', 
    fig_width=14, h_each=3.5, wspace=0.3, hspace=0.5,
    dpi=300, fontsize=10):
    """ Visualize the layers of the model
    
    Original code : https://github.com/ninarina12/phononDoS_tutorial
    Modified by M. Ohnishi
    
    """
    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['gate', 'tp', 'lin2', 'gate']))
    try:
        layers = model.mp.layers
    except:
        layers = model.layers

    num_layers = len(layers)
    num_ops = max([len([k for k in list(layers[i].first._modules.keys()) if k not in ['fc', 'alpha']])
                   for i in range(num_layers-1)])
    
    set_matplot(fontsize=fontsize)
    fig, ax = plt.subplots(num_layers, num_ops, figsize=(fig_width, h_each*num_layers))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    
    for i in range(num_layers - 1):
        ops = layers[i].first._modules.copy()
        ops.pop('fc', None); ops.pop('alpha', None)
        for j, (k, v) in enumerate(ops.items()):
            ax[i,j].set_title(k, fontsize=fontsize)
            v.cpu().visualize(ax=ax[i,j])
            ax[i,j].text(0.7, -0.15, '--> to ' + layer_dst[k], 
                         fontsize=fontsize-2, transform=ax[i,j].transAxes)

    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['output', 'tp', 'lin2', 'output']))
    ops = layers[-1]._modules.copy()
    ops.pop('fc', None); ops.pop('alpha', None)
    for j, (k, v) in enumerate(ops.items()):
        ax[-1,j].set_title(k, fontsize=fontsize)
        v.cpu().visualize(ax=ax[-1,j])
        ax[-1,j].text(0.7, -0.15,'--> to ' + layer_dst[k], 
                      fontsize=fontsize-2, transform=ax[-1,j].transAxes)

    ### adjust the fontsize
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            for text in ax[i,j].texts:
                text.set_fontsize(fontsize)
    
    fig.savefig(figname, dpi=dpi, bbox_inches='tight')
    print(" Output", figname)
    return fig
